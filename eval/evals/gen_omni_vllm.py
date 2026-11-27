"""Resume Qwen3-Omni text benchmark generation with the vLLM backend.

The existing Omni JSONL is append-only from this runner's point of view:
rows already present are treated as immutable trajectories, and only missing
``(example_id, seed)`` pairs are submitted to vLLM.  Requests are submitted
in bounded chunks so completed chunks are flushed to disk before the next
chunk starts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "VL_MMLU" / "test.jsonl"
DEFAULT_SEEDS = (101, 202, 303, 404)


def _configure_local_caches() -> None:
    """Keep local checkpoint/processor cache access off the mergerfs /data mount."""
    cache_pairs = {
        "HF_HOME": "/mnt/disk3/hongbo/hf_cache",
        "HF_HUB_CACHE": "/mnt/disk3/hongbo/hf_cache/hub",
        "TRANSFORMERS_CACHE": "/mnt/disk3/hongbo/hf_cache/hub",
        "VLLM_CACHE_DIR": "/mnt/disk3/hongbo/vllm_cache",
        "TORCH_COMPILE_CACHE": "/mnt/disk3/hongbo/vllm_cache/torch_compile_cache",
    }
    for name, local_path in cache_pairs.items():
        if Path(local_path).parent.is_dir():
            os.environ[name] = local_path


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers: {value!r}"
        ) from exc
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


def _output_name(args: argparse.Namespace) -> str:
    return (
        args.benchmark.lower() + f"_t{_float_token(args.temperature)}"
        f"_p{_float_token(args.top_p)}"
        f"_k{args.top_k}"
        f"_rp{_float_token(args.repetition_penalty)}"
        f"_pp{_float_token(args.presence_penalty)}"
        f"_n{len(args.seeds)}-MNT{args.max_tokens}.jsonl"
    )


def _register_qwen3_omni() -> None:
    """Register the Qwen branch implementation in this vLLM process."""

    # The plugin import patches the Transformers 4.57/vLLM 0.9.2 duplicate
    # config registration before it imports vLLM.
    plugin_root = str(Path(__file__).resolve().parent)
    if plugin_root not in sys.path:
        sys.path.insert(0, plugin_root)
    os.environ["PYTHONPATH"] = plugin_root + os.pathsep + os.environ.get(
        "PYTHONPATH", ""
    )
    from qwen3_omni_vllm_plugin import (  # noqa: F401
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )
    from vllm.model_executor.models import ModelRegistry

    implementation = (
        "qwen3_omni_vllm_plugin:Qwen3OmniMoeThinkerForConditionalGeneration"
    )
    for architecture in (
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeModel",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ):
        ModelRegistry.register_model(architecture, implementation)


def _build_requests(processor: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for sample in batch:
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": sample["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": {},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
        )
    return requests


def _load_engine(args: argparse.Namespace):
    # Set visibility before importing torch/vLLM.  The command-line GPU ids
    # remain physical ids for logging; vLLM sees a compact local group.
    _configure_local_caches()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

    _register_qwen3_omni()
    from qwen3_omni_vllm_plugin import Qwen3OmniMoeThinkerForConditionalGeneration  # noqa: F401
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after setting CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != len(args.gpus):
        raise RuntimeError(
            f"expected {len(args.gpus)} visible GPUs ({args.gpus}), "
            f"found {torch.cuda.device_count()}"
        )
    print(
        f"[omni-vllm] loading {args.model} on physical GPUs={args.gpus}; "
        f"tp={len(args.gpus)} max_model_len={args.max_model_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=len(args.gpus),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        mm_processor_kwargs={"use_audio_in_video": False},
    )
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = 151645  # Qwen3-Omni's <|im_end|> / im_end_token_id.
    if isinstance(eos_token_id, (list, tuple, set)):
        stop_token_ids = [int(token_id) for token_id in eos_token_id]
    else:
        stop_token_ids = [int(eos_token_id)]
    print(
        f"[omni-vllm] loaded; vllm={_version('vllm')} "
        f"transformers={_version('transformers')} torch={_version('torch')} "
        f"stop_token_ids={stop_token_ids}",
        flush=True,
    )
    return processor, llm, stop_token_ids


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    data_path: Path,
    output_file: Path,
    samples: list[dict[str, Any]],
    existing: dict[tuple[str, int], dict[str, Any]],
) -> None:
    manifest = {
        "schema_version": 3,
        "model": str(Path(args.model).expanduser().resolve()),
        "model_name": "Qwen3-Omni-30B-A3B",
        "variant": args.variant,
        "modality": "OMNI",
        "benchmark": args.benchmark,
        "data_path": str(data_path.resolve()),
        "num_examples": len(samples),
        "seeds": args.seeds,
        "gpus": args.gpus,
        "tensor_parallel_size": len(args.gpus),
        "backend": "vllm",
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "request_chunk_size": args.request_chunk_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "response_mode": "full",
            "return_audio": False,
            "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
            "stop_token_ids": getattr(args, "stop_token_ids", [151645]),
        },
        "output_file": str(output_file),
        "runner": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "vllm": _version("vllm"),
            "transformers": _version("transformers"),
            "torch": _version("torch"),
            "qwen_vllm_branch": "ca66cbff08eda045240804b847cca712c80f9208",
            "trajectory_policy": "append_missing_pairs_only",
        },
        "num_rows": len(existing),
        "rows_per_example": {
            str(sample["example_id"]): sum(
                (str(sample["example_id"]), int(seed)) in existing
                for seed in args.seeds
            )
            for sample in samples
        },
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume Qwen3-Omni text benchmark generation with vLLM"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        choices=("MMLU", "GSM8K"),
        default="MMLU",
        help="benchmark label written to the manifest and output filename",
    )
    parser.add_argument("--variant", default="thinking")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--gpus", type=_ints, default=[3, 8])
    parser.add_argument("--seeds", type=_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--request-chunk-size", type=int, default=16)
    args = parser.parse_args()

    if len(args.gpus) < 2:
        parser.error("Qwen3-Omni-30B requires at least two GPUs")
    if args.n is not None:
        if not 0 < args.n <= len(args.seeds):
            parser.error("--n must be between 1 and the number of seeds")
        args.seeds = args.seeds[: args.n]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.request_chunk_size <= 0 or args.max_num_seqs <= 0:
        parser.error("request chunk size and max-num-seqs must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")

    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(data_path, args.limit)
    output_file = output_dir / _output_name(args)
    manifest_file = output_dir / "generation_manifest.json"
    existing = _read_existing(output_file)
    expected = {
        (str(sample["example_id"]), int(seed))
        for sample in samples
        for seed in args.seeds
    }
    missing = expected.difference(existing)
    print(
        f"[omni-vllm] existing_rows={len(existing)} expected_rows={len(expected)} "
        f"missing_rows={len(missing)} output={output_file}",
        flush=True,
    )
    if not missing:
        _write_manifest(manifest_file, args, data_path, output_file, samples, existing)
        print("[omni-vllm] nothing to generate", flush=True)
        return

    # Write the migration-aware manifest before loading the model so a crash
    # during a long load still records which backend was selected.
    _write_manifest(manifest_file, args, data_path, output_file, samples, existing)
    processor, llm, stop_token_ids = _load_engine(args)
    from vllm import SamplingParams

    started = time.time()
    output_mode = "a" if output_file.exists() else "w"
    with output_file.open(output_mode, encoding="utf-8") as handle:
        for seed in args.seeds:
            pending = [
                sample
                for sample in samples
                if (str(sample["example_id"]), int(seed)) not in existing
            ]
            print(
                f"[omni-vllm] start seed={seed} pending={len(pending)} "
                f"chunk={args.request_chunk_size} max_num_seqs={args.max_num_seqs}",
                flush=True,
            )
            for batch_start in range(0, len(pending), args.request_chunk_size):
                batch = pending[batch_start : batch_start + args.request_chunk_size]
                requests = _build_requests(processor, batch)
                sampling = SamplingParams(
                    n=1,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    max_tokens=args.max_tokens,
                    seed=int(seed),
                    stop_token_ids=stop_token_ids,
                )
                outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(batch)} requests"
                    )
                for sample, output in zip(batch, outputs):
                    completion = output.outputs[0]
                    raw_response = str(completion.text)
                    token_ids = getattr(completion, "token_ids", None) or []
                    token_length = len(token_ids)
                    finish_reason = getattr(completion, "finish_reason", None)
                    if finish_reason is None:
                        finish_reason = (
                            "length" if token_length >= args.max_tokens else "stop"
                        )
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
                        "truncated": bool(
                            finish_reason == "length" or token_length >= args.max_tokens
                        ),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[(str(sample["example_id"]), int(seed))] = row
                elapsed = time.time() - started
                tokens = [
                    int(existing[(str(sample["example_id"]), int(seed))]["token_length"])
                    for sample in batch
                ]
                print(
                    f"[omni-vllm] seed={seed} examples={batch_start + 1}-"
                    f"{batch_start + len(batch)}/{len(samples)} rows={len(existing)} "
                    f"tokens={tokens} elapsed={elapsed:.0f}s",
                    flush=True,
                )
            print(f"[omni-vllm] finished seed={seed}", flush=True)

    _write_manifest(manifest_file, args, data_path, output_file, samples, existing)
    print(f"[omni-vllm] saved {len(existing)} rows -> {output_file}", flush=True)


if __name__ == "__main__":
    main()
