"""Transformers generation runner for the Qwen3-Omni text MMLU experiment.

Qwen3-Omni is not supported by the vLLM version used by the existing SAR
queues.  This runner keeps the same MMLU snapshot, rollout seeds, decoding
parameters, output schema, and full-response trajectory policy while loading
one Omni checkpoint across an explicitly selected set of GPUs with
``device_map=auto``.

The runner is intentionally text-only: Qwen3-Omni's Talker is disabled when
present and ``return_audio=False`` is used.  It writes each completed row
incrementally so a long run leaves recoverable trajectories on disk.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "VL_MMLU" / "test.jsonl"
DEFAULT_SEEDS = (101, 202, 303, 404)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _float_token(value: float | int) -> str:
    return str(float(value))


def _load_samples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError(f"{path}:{line_number} lacks question/answer")
            samples.append(
                {
                    "example_id": row.get("example_id", len(samples)),
                    "source_id": row.get("source_id"),
                    "question": question,
                    "answer": answer,
                    "subject": row.get("subject"),
                }
            )
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"no samples found in {path}")
    return samples


class _PresencePenalty:
    """Transformers LogitsProcessor equivalent to vLLM presence_penalty."""

    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = int(prompt_length)
        self.penalty = float(penalty)
        self._processed_length: int | None = None
        self._seen: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        # The previous implementation recomputed torch.unique() over the
        # entire generated prefix on every decoding step.  That becomes
        # quadratic in sequence length and is especially costly for a batch.
        # Keep a per-row vocabulary mask and only add the newly appended
        # tokens; this is mathematically the same one-time presence penalty.
        if self._seen is None:
            self._seen = torch.zeros(
                (input_ids.shape[0], scores.shape[-1]),
                dtype=torch.bool,
                device=scores.device,
            )
            self._processed_length = self.prompt_length

        current_length = int(input_ids.shape[1])
        processed_length = int(self._processed_length or self.prompt_length)
        if current_length > processed_length:
            new_tokens = input_ids[:, processed_length:current_length]
            for row_index, token_row in enumerate(new_tokens):
                self._seen[row_index].scatter_(
                    0,
                    token_row.to(device=self._seen.device, dtype=torch.long),
                    True,
                )
            self._processed_length = current_length

        scores -= self._seen.to(dtype=scores.dtype) * self.penalty
        return scores


def _input_device(model: Any):
    """Return the device on which the first thinker input should be placed."""
    thinker = getattr(model, "thinker", None)
    modules = [thinker, model] if thinker is not None else [model]
    for module in modules:
        try:
            for parameter in module.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        except StopIteration:
            continue
    raise RuntimeError("could not find a non-meta model parameter device")


def _move_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        result[key] = value.to(device) if hasattr(value, "to") else value
    return result


def _load_model(
    model_path: str,
    gpu_ids: list[int],
    gpu_memory_utilization: float,
    attn_implementation: str,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
):
    # This must happen before importing torch/Transformers so CUDA sees the
    # requested physical cards as a compact local device group.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Omni has 15/16 safetensors shards.  Transformers otherwise opens them
    # serially, which is particularly slow on the local multi-disk mount.
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after setting CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != len(gpu_ids):
        raise RuntimeError(
            f"expected {len(gpu_ids)} visible GPUs ({gpu_ids}), "
            f"found {torch.cuda.device_count()}"
        )

    max_memory = {}
    for local_id in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(local_id).total_memory // (1024**3)
        # The same 0.95 memory target requested for the vLLM math runs.  Keep
        # a small margin for CUDA context and generation buffers.
        max_memory[local_id] = f"{max(1, int(total_gib * gpu_memory_utilization))}GiB"

    dtype = torch.bfloat16
    print(
        f"[omni] loading {model_path} on physical GPUs={gpu_ids}; "
        f"max_memory={max_memory}; dtype=bfloat16",
        flush=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        # ``torch_dtype`` works in both the current Transformers release and
        # the 4.57.x build used by the existing vLLM queues.
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation=attn_implementation,
    )
    if bool(getattr(model, "has_talker", False)):
        model.disable_talker()
        print("[omni] Talker disabled; text-only generation", flush=True)
    model.eval()
    # Decoder-only batched generation is most reliable with left padding.  It
    # also makes the generated suffix begin at one common sequence position,
    # which is required by the per-row presence-penalty processor below.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        print(f"[omni] tokenizer padding_side={tokenizer.padding_side}", flush=True)

    if compile_model:
        thinker = getattr(model, "thinker", None)
        if thinker is None:
            raise RuntimeError("--compile requested but the Omni model has no thinker module")
        print(
            f"[omni] compiling thinker with torch.compile(mode={compile_mode}, dynamic=True); "
            "the first batch will include compilation overhead",
            flush=True,
        )
        model.thinker = torch.compile(thinker, mode=compile_mode, dynamic=True)
    return torch, processor, model, _input_device(model)


def _build_output_name(args: argparse.Namespace, seed_count: int) -> str:
    return (
        "mmlu"
        f"_t{_float_token(args.temperature)}"
        f"_p{_float_token(args.top_p)}"
        f"_k{args.top_k}"
        f"_rp{_float_token(args.repetition_penalty)}"
        f"_pp{_float_token(args.presence_penalty)}"
        f"_n{seed_count}-MNT{args.max_tokens}.jsonl"
    )


def _read_existing(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return existing
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["example_id"]), int(row["seed"]))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid existing row {path}:{line_number}") from exc
            existing[key] = row
    return existing


def _trim_existing_file(path: Path, keep_rows: int) -> None:
    """Safely remove trailing rows before regenerating a bad partial batch."""
    if keep_rows < 0:
        raise ValueError("keep_rows must be non-negative")
    with path.open("rb") as handle:
        lines = handle.readlines()
    if len(lines) <= keep_rows:
        return
    backup = path.with_suffix(path.suffix + f".pretrim-{int(time.time())}")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".trim-tmp")
    try:
        with temporary.open("wb") as handle:
            handle.writelines(lines[:keep_rows])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"[omni] trimmed {len(lines) - keep_rows} trailing rows from {path}; "
        f"backup={backup}",
        flush=True,
    )


def _sequences_from_generation(generated: Any) -> Any:
    """Normalize the thinker generation result across Transformers revisions."""
    if hasattr(generated, "sequences"):
        return generated.sequences
    if isinstance(generated, (tuple, list)) and generated:
        first = generated[0]
        return first.sequences if hasattr(first, "sequences") else first
    raise TypeError(
        "Qwen3-Omni generate() returned an unsupported result "
        f"type: {type(generated)!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3-Omni text MMLU trajectories")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--gpus", type=_ints, default=list((3, 8)))
    parser.add_argument("--seeds", type=_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit examples for a smoke test")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of text questions generated together (left-padded)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the thinker; the first batch is slower while compiling",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument(
        "--trim-existing-to",
        type=int,
        default=None,
        help="safely trim an existing JSONL to this many rows before resuming",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if len(args.gpus) < 2:
        parser.error("Qwen3-Omni-30B requires at least two GPUs for this run")
    if args.n is not None:
        if not 0 < args.n <= len(args.seeds):
            parser.error("--n must be between 1 and the number of seeds")
        args.seeds = args.seeds[: args.n]
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < -1:
        parser.error("invalid sampling parameters")
    if args.repetition_penalty <= 0 or args.max_tokens <= 0 or args.max_model_len <= 0:
        parser.error("invalid generation length/repetition parameters")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.trim_existing_to is not None and args.trim_existing_to < 0:
        parser.error("trim-existing-to must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(data_path, args.limit)
    output_file = output_dir / _build_output_name(args, len(args.seeds))
    manifest_file = output_dir / "generation_manifest.json"

    if args.trim_existing_to is not None and output_file.exists():
        _trim_existing_file(output_file, args.trim_existing_to)

    if args.overwrite and output_file.exists():
        # Keep the old file recoverable instead of unlinking it.  A fresh
        # timestamped backup makes an explicit overwrite non-destructive.
        backup = output_file.with_suffix(output_file.suffix + f".bak-{int(time.time())}")
        output_file.rename(backup)
        print(f"[omni] moved previous output to {backup}", flush=True)
    existing = {} if args.no_resume else _read_existing(output_file)
    expected = {(str(sample["example_id"]), int(seed)) for sample in samples for seed in args.seeds}
    if existing and expected.issubset(existing):
        print(f"[omni] all {len(expected)} rows already exist in {output_file}; nothing to do")
        return

    manifest = {
        "schema_version": 2,
        "model": str(model_path),
        "model_name": "Qwen3-Omni-30B-A3B",
        "variant": output_dir.parent.name,
        "modality": "OMNI",
        "benchmark": "MMLU",
        "data_path": str(data_path),
        "num_examples": len(samples),
        "seeds": args.seeds,
        "gpus": args.gpus,
        "backend": "transformers",
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "presence_penalty_impl": "incremental_seen_mask",
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size,
            "torch_compile": args.compile,
            "torch_compile_mode": args.compile_mode if args.compile else None,
            "response_mode": "full",
            "return_audio": False,
        },
        "output_file": str(output_file),
        "runner": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "transformers": _version("transformers"),
            "torch": _version("torch"),
        },
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    torch, processor, model, input_device = _load_model(
        str(model_path),
        args.gpus,
        args.gpu_memory_utilization,
        args.attn_implementation,
        args.compile,
        args.compile_mode,
    )
    from transformers import LogitsProcessorList

    start_time = time.time()
    output_mode = "a" if output_file.exists() else "w"
    with output_file.open(output_mode, encoding="utf-8") as handle:
        generated_rows = 0
        for seed in args.seeds:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            pending = [
                sample
                for sample in samples
                if (str(sample["example_id"]), int(seed)) not in existing
            ]
            print(
                f"[omni] start seed={seed} pending={len(pending)} batch_size={args.batch_size}",
                flush=True,
            )
            for batch_start in range(0, len(pending), args.batch_size):
                batch = pending[batch_start : batch_start + args.batch_size]
                prompt_texts = [
                    processor.apply_chat_template(
                        [{"role": "user", "content": sample["question"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for sample in batch
                ]
                inputs = processor(text=prompt_texts, return_tensors="pt", padding=True)
                inputs = _move_inputs(inputs, input_device)
                # Left padding gives every row the same input width.  The
                # generated suffix therefore starts at this exact position.
                prompt_length = int(inputs["input_ids"].shape[1])
                processors = LogitsProcessorList()
                if args.presence_penalty:
                    processors.append(_PresencePenalty(prompt_length, args.presence_penalty))
                generation_kwargs = {
                    "return_audio": False,
                    "thinker_max_new_tokens": args.max_tokens,
                    "thinker_do_sample": args.temperature > 0,
                    "thinker_temperature": args.temperature,
                    "thinker_top_p": args.top_p,
                    "thinker_top_k": args.top_k,
                    "thinker_repetition_penalty": args.repetition_penalty,
                    "thinker_return_dict_in_generate": True,
                    "thinker_logits_processor": processors,
                }
                pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
                if pad_token_id is not None:
                    generation_kwargs["thinker_pad_token_id"] = int(pad_token_id)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_kwargs)
                sequences = _sequences_from_generation(generated)
                if getattr(sequences, "ndim", None) != 2 or sequences.shape[0] != len(batch):
                    raise RuntimeError(
                        "unexpected batched generation shape: "
                        f"{getattr(sequences, 'shape', None)} for batch={len(batch)}"
                    )
                eos_token_id = int(getattr(model.config, "im_end_token_id", 151645))
                for row_index, sample in enumerate(batch):
                    key = (str(sample["example_id"]), int(seed))
                    completion_ids = sequences[row_index, prompt_length:]
                    # Batched generation right-pads shorter rows.  Exclude
                    # that padding so token usage and finish_reason retain
                    # the same meaning as in the original batch=1 runner.
                    eos_positions = (completion_ids == eos_token_id).nonzero(as_tuple=False)
                    if eos_positions.numel():
                        completion_ids = completion_ids[: int(eos_positions[0, 0]) + 1]
                    elif pad_token_id is not None:
                        non_pad_positions = (completion_ids != int(pad_token_id)).nonzero(as_tuple=False)
                        if non_pad_positions.numel():
                            completion_ids = completion_ids[: int(non_pad_positions[-1, 0]) + 1]
                        else:
                            completion_ids = completion_ids[:0]
                    raw_response = processor.decode(
                        completion_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    token_length = int(completion_ids.numel())
                    finish_reason = "length" if token_length >= args.max_tokens else "stop"
                    row = {
                        "example_id": sample["example_id"],
                        "source_id": sample.get("source_id"),
                        "question": sample["question"],
                        "prompt": sample["question"],
                        "answer": sample["answer"],
                        "subject": sample.get("subject"),
                        "seed": int(seed),
                        "raw_response": None,
                        "response": raw_response,
                        "token_length": token_length,
                        "finish_reason": finish_reason,
                        "truncated": finish_reason == "length",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[key] = row
                    generated_rows += 1
                if generated_rows == len(batch) or generated_rows % 10 < len(batch):
                    elapsed = time.time() - start_time
                    last_tokens = [
                        int(existing[(str(sample["example_id"]), int(seed))]["token_length"])
                        for sample in batch
                    ]
                    print(
                        f"[omni] seed={seed} examples={batch_start + 1}-"
                        f"{batch_start + len(batch)}/{len(samples)} "
                        f"batch={len(batch)} new_rows={generated_rows} "
                        f"tokens={last_tokens} elapsed={elapsed:.0f}s",
                        flush=True,
                    )
            print(f"[omni] finished seed={seed}", flush=True)

    manifest["num_rows"] = len(existing)
    manifest["rows_per_example"] = {
        str(sample["example_id"]): sum(
            (str(sample["example_id"]), int(seed)) in existing for seed in args.seeds
        )
        for sample in samples
    }
    manifest["completed_at"] = time.time()
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[omni] saved {len(existing)} rows -> {output_file}", flush=True)


if __name__ == "__main__":
    main()
