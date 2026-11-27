"""Generate Qwen3-Omni audio-benchmark trajectories with vLLM.

The input is the normalized JSONL produced by
``prepare_omni_audio_data.py``.  Output is append-only at the
``(example_id, seed)`` level: a stopped run can be resumed without replacing
any already-written response.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEEDS = (101, 202, 303, 404)
# ``/data`` is a mergerfs mount on this host.  The Omni checkpoints and
# prepared audio currently live on disk3; resolving through the underlying
# filesystem avoids mergerfs stalls while keeping the canonical paths in the
# input files usable.
UNDERLYING_DATA_ROOTS = (
    Path("/mnt/disk3"),
    Path("/mnt/disk2"),
    Path("/mnt/disk1"),
)


def _resolve_access_path(path: Path, directory: bool = False) -> Path:
    """Resolve a /data path through an underlying local disk when possible."""
    raw = path.expanduser()
    candidates: list[Path] = []
    if raw.is_absolute() and len(raw.parts) >= 3 and raw.parts[1] == "data":
        relative = Path(*raw.parts[2:])
        candidates.extend(root / relative for root in UNDERLYING_DATA_ROOTS)
    else:
        candidates.append(raw)
    for candidate in candidates:
        try:
            valid = candidate.is_dir() if directory else candidate.is_file()
        except OSError:
            valid = False
        if valid:
            return candidate.resolve()
    return raw


def _configure_local_caches() -> None:
    """Redirect host-level /data caches to their local disk3 locations."""
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
            choices = [str(choice).strip() for choice in row.get("choices", [])]
            audio_path = _resolve_access_path(
                Path(str(row.get("audio_path", ""))), directory=False
            )
            if not question or not choices or not audio_path.is_file():
                raise ValueError(
                    f"{path}:{line_number} lacks question/choices/audio: {row.get('example_id')}"
                )
            if row.get("answer_index") is None:
                raise ValueError(f"{path}:{line_number} has no answer_index")
            samples.append(
                {
                    **row,
                    "example_id": row.get("example_id", len(samples)),
                    "source_id": row.get("source_id"),
                    "audio_path": str(audio_path.resolve()),
                    "question": question,
                    "choices": choices,
                    "answer_index": int(row["answer_index"]),
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
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = (str(row["example_id"]), int(row["seed"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid existing row {path}:{line_number}") from exc
                existing[key] = row
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return existing


def _output_name(args: argparse.Namespace) -> str:
    return (
        args.benchmark.lower()
        + f"_t{_float_token(args.temperature)}"
        + f"_p{_float_token(args.top_p)}"
        + f"_k{args.top_k}"
        + f"_rp{_float_token(args.repetition_penalty)}"
        + f"_pp{_float_token(args.presence_penalty)}"
        + f"_n{len(args.seeds)}-MNT{args.max_tokens}.jsonl"
    )


def _register_qwen3_omni() -> None:
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


class AudioReader:
    """Decode local audio to mono 16 kHz float32 arrays using ffmpeg."""

    def __init__(self, max_cache_items: int = 32):
        self.max_cache_items = max_cache_items
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def read(self, path: str) -> np.ndarray:
        cached = self.cache.pop(path, None)
        if cached is not None:
            self.cache[path] = cached
            return cached
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if audio.size == 0:
            raise RuntimeError(f"ffmpeg returned no samples for {path}")
        self.cache[path] = audio
        while len(self.cache) > self.max_cache_items:
            self.cache.popitem(last=False)
        return audio


def _question_text(sample: dict[str, Any]) -> str:
    options = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(sample["choices"])
    )
    return (
        "Listen carefully to the audio and answer the question. Reason about "
        "the audio when needed, then give your final choice clearly.\n\n"
        f"Question: {sample['question']}\n\n{options}\n\n"
        "End your response with the final answer as one option letter (A, B, C, ...)."
    )


def _build_requests(
    processor: Any,
    batch: list[dict[str, Any]],
    reader: AudioReader,
    official_settings: bool = False,
) -> list[dict[str, Any]]:
    """Build vLLM requests.

    In official mode the multimodal payload follows Qwen's published
    ``process_mm_info`` path and uses the same ``use_audio_in_video=True``
    processor option.  The local ffmpeg reader is still used as a cache: the
    official utility accepts an already-decoded 16 kHz mono numpy array, so
    repeated seeds do not decode the same audio four times.
    """
    if official_settings:
        from qwen_omni_utils import process_mm_info

    requests: list[dict[str, Any]] = []
    for sample in batch:
        text = _question_text(sample)
        template_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": sample["audio_path"]},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if official_settings:
            # ``process_mm_info`` is the official utility used by the Qwen3-
            # Omni example.  Supplying its supported ndarray form preserves
            # the official post-processing while avoiding repeated decoding.
            media_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": reader.read(sample["audio_path"])},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            audios, images, videos = process_mm_info(
                media_messages, use_audio_in_video=True
            )
            multi_modal_data: dict[str, Any] = {}
            if images is not None:
                multi_modal_data["image"] = images
            if videos is not None:
                multi_modal_data["video"] = videos
            if audios is not None:
                multi_modal_data["audio"] = audios
            requests.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": multi_modal_data,
                    "mm_processor_kwargs": {"use_audio_in_video": True},
                }
            )
            continue
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": reader.read(sample["audio_path"])},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
        )
    return requests


def _load_engine(args: argparse.Namespace):
    _configure_local_caches()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

    args.model = str(_resolve_access_path(Path(args.model), directory=True))
    _register_qwen3_omni()
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
    pipeline_parallel_size = args.pipeline_parallel_size
    if pipeline_parallel_size <= 0 or len(args.gpus) % pipeline_parallel_size:
        raise ValueError(
            "pipeline parallel size must be positive and divide the number of GPUs"
        )
    tensor_parallel_size = len(args.gpus) // pipeline_parallel_size
    print(
        f"[omni-audio-vllm] loading model={args.model} physical_gpus={args.gpus} "
        f"tp={tensor_parallel_size} pp={pipeline_parallel_size} "
        f"max_model_len={args.max_model_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} "
        f"max_num_seqs={args.max_num_seqs}",
        flush=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": (
            {"image": 0, "video": 0, "audio": 1}
            if args.official_settings
            else {"image": 0, "video": 0, "audio": 1}
        ),
        "mm_processor_kwargs": {
            "use_audio_in_video": bool(args.official_settings)
        },
    }
    if args.official_settings:
        # Match the official example.  Its tensor_parallel_size is the number
        # of visible GPUs; this runner computes the same value after optional
        # pipeline parallel partitioning (the queue uses pp=1).
        llm_kwargs["seed"] = 1234
    else:
        llm_kwargs.update(dtype="bfloat16", enforce_eager=True)
    llm = LLM(**llm_kwargs)
    if args.official_settings:
        stop_token_ids = None
    else:
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = 151645
        if isinstance(eos_token_id, (list, tuple, set)):
            stop_token_ids = [int(token_id) for token_id in eos_token_id]
        else:
            stop_token_ids = [int(eos_token_id)]
    print(
        f"[omni-audio-vllm] loaded vllm={_version('vllm')} "
        f"transformers={_version('transformers')} torch={_version('torch')} "
        f"qwen_omni_utils={_version('qwen-omni-utils')} "
        f"official_settings={args.official_settings} "
        f"stop_token_ids={stop_token_ids}",
        flush=True,
    )
    return processor, llm, stop_token_ids


def _dataset_manifest(data_path: Path) -> dict[str, Any]:
    path = data_path.with_name("dataset_manifest.json")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    data_path: Path,
    output_file: Path,
    samples: list[dict[str, Any]],
    existing: dict[tuple[str, int], dict[str, Any]],
    stop_token_ids: list[int] | None = None,
) -> None:
    # A manifest can be written by multiple disjoint seed shards.  Refresh
    # the row count under the JSONL's shared lock so the last writer does not
    # publish a stale in-memory count from its own shard.
    manifest_existing = _read_existing(output_file)
    dataset_manifest = _dataset_manifest(data_path)
    dataset_info = dataset_manifest.get("dataset", {})
    manifest = {
        "schema_version": 4,
        "model": str(Path(args.model).expanduser().resolve()),
        "model_name": "Qwen3-Omni-30B-A3B",
        "variant": args.variant,
        "modality": "OMNI_AUDIO",
        "benchmark": args.benchmark,
        "data_path": str(data_path.resolve()),
        "dataset": dataset_info,
        "num_examples": len(samples),
        "seeds": args.seeds,
        # ``run_seeds`` may be a disjoint subset when several vLLM workers
        # append to the same canonical file.  The manifest describes the
        # canonical result, so retain the complete requested seed set rather
        # than whichever shard happened to write last.
        "run_seeds": args.seeds,
        "gpus": args.gpus,
        "tensor_parallel_size": len(args.gpus) // args.pipeline_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "backend": "vllm",
        "audio": {
            "input_sample_rate": 16000,
            "channels": 1,
            "decoder": "ffmpeg",
            "process_mm_info": bool(args.official_settings),
            "use_audio_in_video": bool(args.official_settings),
            "limit_mm_per_prompt": (
                {"audio": 1, "image": 0, "video": 0}
            ),
        },
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
            "dtype": "auto" if args.official_settings else "bfloat16",
            "response_mode": "full",
            "return_audio": False,
            "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
            "engine_seed": 1234 if args.official_settings else None,
            "stop_token_ids": stop_token_ids,
            "official_settings": bool(args.official_settings),
        },
        "output_file": str(output_file),
        "runner": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "vllm": _version("vllm"),
            "transformers": _version("transformers"),
            "torch": _version("torch"),
            "qwen_omni_utils": _version("qwen-omni-utils"),
            "qwen_vllm_branch": "ca66cbff08eda045240804b847cca712c80f9208",
            "trajectory_policy": "append_missing_pairs_only",
        },
        "num_rows": len(manifest_existing),
        "updated_at": time.time(),
    }
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni audio generation with vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        choices=("MMAU", "MMAU_testmini", "MMAR", "MMSU"),
        required=True,
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--gpus", type=_ints, required=True)
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="pipeline parallel degree; tensor parallel degree is len(gpus)//this",
    )
    parser.add_argument("--seeds", type=_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--run-seeds",
        type=_ints,
        default=None,
        help=(
            "Optional subset of --seeds to generate in this process. The full "
            "--seeds list is retained for the canonical output filename and manifest."
        ),
    )
    parser.add_argument(
        "--example-start",
        type=int,
        default=1,
        help="1-based inclusive first sample to generate; the default is the full dataset.",
    )
    parser.add_argument(
        "--example-stop",
        type=int,
        default=None,
        help="1-based inclusive last sample to generate; the default is the dataset end.",
    )
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
    parser.add_argument("--request-chunk-size", type=int, default=8)
    parser.add_argument(
        "--official-settings",
        action="store_true",
        help="Use the official Qwen3-Omni vLLM and sampling settings.",
    )
    args = parser.parse_args()

    if args.official_settings:
        # Keep this mode unambiguous even if a caller inherited old queue
        # flags.  Four seeds remain an experiment-level extension; the engine
        # seed stays at the official example's 1234.
        args.temperature = 0.6
        args.top_p = 0.95
        args.top_k = 20
        args.repetition_penalty = 1.0
        args.presence_penalty = 0.0
        args.max_tokens = 16384
        args.max_model_len = 32768
        args.max_num_seqs = 8
        args.request_chunk_size = 8
        # The official example uses one request.  For long benchmark runs we
        # keep its engine/sampling/multimodal settings but may submit a larger
        # request window so vLLM's continuous batcher can replace short
        # requests while another request is still decoding to max_tokens.
        # This is intentionally opt-in and does not change max_num_seqs.
        official_chunk_override = os.environ.get("OMNI_OFFICIAL_REQUEST_CHUNK_SIZE")
        if official_chunk_override:
            args.request_chunk_size = int(official_chunk_override)

    if len(args.gpus) < 2:
        parser.error("Qwen3-Omni-30B requires at least two GPUs")
    if args.pipeline_parallel_size <= 0 or len(args.gpus) % args.pipeline_parallel_size:
        parser.error("pipeline parallel size must divide the number of GPUs")
    if args.n is not None:
        if not 0 < args.n <= len(args.seeds):
            parser.error("--n must be between 1 and the number of seeds")
        args.seeds = args.seeds[: args.n]
    if args.run_seeds is None:
        args.run_seeds = list(args.seeds)
    elif any(seed not in args.seeds for seed in args.run_seeds):
        parser.error("--run-seeds must be a subset of --seeds")
    if not args.run_seeds:
        parser.error("--run-seeds must contain at least one seed")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.example_start <= 0:
        parser.error("--example-start must be positive")
    if args.example_stop is not None and args.example_stop < args.example_start:
        parser.error("--example-stop must be greater than or equal to --example-start")
    if args.request_chunk_size <= 0 or args.max_num_seqs <= 0:
        parser.error("request chunk size and max-num-seqs must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")

    data_path = _resolve_access_path(Path(args.data), directory=False)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(data_path, args.limit)
    if args.example_start > len(samples):
        parser.error(
            f"--example-start={args.example_start} exceeds dataset size {len(samples)}"
        )
    example_stop = len(samples) if args.example_stop is None else min(
        args.example_stop, len(samples)
    )
    generation_samples = samples[args.example_start - 1 : example_stop]
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
        f"[omni-audio-vllm] benchmark={args.benchmark} existing_rows={len(existing)} "
        f"expected_rows={len(expected)} missing_rows={len(missing)} "
        f"output={output_file}",
        flush=True,
    )
    if not missing:
        _write_manifest(manifest_file, args, data_path, output_file, samples, existing)
        print("[omni-audio-vllm] nothing to generate", flush=True)
        return

    _write_manifest(manifest_file, args, data_path, output_file, samples, existing)
    processor, llm, stop_token_ids = _load_engine(args)
    _write_manifest(
        manifest_file, args, data_path, output_file, samples, existing, stop_token_ids
    )
    from vllm import SamplingParams

    reader = AudioReader()
    started = time.time()
    # Always append.  Two seed shards can start concurrently when the
    # canonical file does not exist yet; checking exists() before opening
    # would let the second shard truncate the first shard's trajectories.
    output_mode = "a"
    with output_file.open(output_mode, encoding="utf-8") as handle:
        for seed in args.run_seeds:
            # Another vLLM worker may be filling a disjoint seed partition in
            # the same canonical file. Refresh before each seed so this
            # process does not regenerate rows that appeared since startup.
            existing = _read_existing(output_file)
            pending = [
                sample
                for sample in generation_samples
                if (str(sample["example_id"]), int(seed)) not in existing
            ]
            print(
                f"[omni-audio-vllm] start benchmark={args.benchmark} seed={seed} "
                f"pending={len(pending)} chunk={args.request_chunk_size} "
                f"max_num_seqs={args.max_num_seqs}",
                flush=True,
            )
            for batch_start in range(0, len(pending), args.request_chunk_size):
                batch = pending[batch_start : batch_start + args.request_chunk_size]
                requests = _build_requests(
                    processor,
                    batch,
                    reader,
                    official_settings=args.official_settings,
                )
                sampling_kwargs: dict[str, Any] = {
                    "n": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "presence_penalty": args.presence_penalty,
                    "max_tokens": args.max_tokens,
                    "seed": int(seed),
                }
                if not args.official_settings:
                    sampling_kwargs["stop_token_ids"] = stop_token_ids
                sampling = SamplingParams(**sampling_kwargs)
                outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(batch)} requests"
                    )
                for sample, output in zip(batch, outputs):
                    completion = output.outputs[0]
                    response = str(completion.text)
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
                        "audio_path": sample["audio_path"],
                        "question": sample["question"],
                        "choices": sample["choices"],
                        "answer": sample.get("answer"),
                        "answer_index": sample["answer_index"],
                        "answer_letter": sample.get("answer_letter"),
                        "dataset": sample.get("dataset"),
                        "dataset_split": sample.get("dataset_split"),
                        "benchmark": args.benchmark,
                        "seed": int(seed),
                        "raw_response": None,
                        "response": response,
                        "token_length": token_length,
                        "finish_reason": finish_reason,
                        "truncated": bool(
                            finish_reason == "length" or token_length >= args.max_tokens
                        ),
                    }
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    existing[(str(sample["example_id"]), int(seed))] = row
                elapsed = time.time() - started
                tokens = [
                    int(existing[(str(sample["example_id"]), int(seed))]["token_length"])
                    for sample in batch
                ]
                print(
                    f"[omni-audio-vllm] benchmark={args.benchmark} seed={seed} "
                    f"examples={args.example_start + batch_start}-"
                    f"{args.example_start + batch_start + len(batch) - 1}/{len(samples)} "
                    f"rows={len(existing)} tokens={tokens} elapsed={elapsed:.0f}s",
                    flush=True,
                )
            print(f"[omni-audio-vllm] finished seed={seed}", flush=True)

    _write_manifest(
        manifest_file, args, data_path, output_file, samples, existing, stop_token_ids
    )
    print(f"[omni-audio-vllm] saved {len(existing)} rows -> {output_file}", flush=True)


if __name__ == "__main__":
    main()
