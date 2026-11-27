"""vLLM/Transformers generation runner for the SAR evaluation matrix.

The runner writes one benchmark directory per experiment, for example::

    results/TEXT/Qwen3-4B/instruct/HMMT25/

It uses four reproducible rollouts by default (seeds 101/202/303/404), keeps
the full response by default, records truncation and finish reasons, and
writes a ``generation_manifest.json`` next to the JSONL output.  The old
``<task>_t0.6_p0.95_n4-MNT32768.jsonl`` naming convention is available with
``--legacy-filename`` for reproducing the archived HMMT25 example.

The vLLM and Transformers imports are intentionally lazy: ``--help`` and
data-loader checks can run on a login node without initializing CUDA.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal environments
    def tqdm(values, **_: Any):
        return values


TASK_REGISTRY = {
    "AIME24": "AIME24/test.parquet",
    "AIME25": "AIME25/test.parquet",
    "AIME26": "AIME26/test.parquet",
    "AMC23": "AMC23/test.parquet",
    "MATH-500": "MATH-500/test.parquet",
    "Minerva": "Minerva/test.parquet",
    "Olympiad-Bench": "Olympiad-Bench/test.parquet",
    "MMLU-Pro": "MMLU-Pro/test.jsonl",
    "ARC-C": "ARC-C/test.jsonl",
    "BRUMO25": "BRUMO25/test.parquet",
    "CMIMC25": "CMIMC25/test.parquet",
    "HMMT25": "HMMT25/test.parquet",
    # These are normally supplied with --task-path because their official
    # runners use benchmark-specific formats.
    "LiveCodeBench": None,
    "BFCL": None,
    "MMLU": None,
    "GSM8K": None,
    "MMMU": None,
    "MathVista-testmini": "VL_MathVista/testmini.jsonl",
    "ARKitScenes": "VL_ARKitScenes/test.jsonl",
}

MATH_TASKS = {
    "AIME24",
    "AIME25",
    "AIME26",
    "AMC23",
    "MATH-500",
    "Minerva",
    "Olympiad-Bench",
    "BRUMO25",
    "CMIMC25",
    "HMMT25",
}
VL_VISUAL_TASKS = {"MMMU", "MathVista-testmini", "ARKitScenes"}
DEFAULT_SEEDS = (101, 202, 303, 404)
# Current operator authorization: GPU4 is ERROR; all other visible cards may
# be used.  Launchers may still pass a smaller explicit subset when cards are
# occupied by unrelated jobs.
ALLOWED_GPU_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 9}
MATH_PROMPT_SUFFIX = (
    " Please reason step by step, and put your final answer within \\boxed{}."
)
BFCL_PROMPT_SUFFIX = (
    "\n\nFor BFCL function calling, output exactly one JSON array with one element "
    "per user turn, in turn order. Each turn element is an array of function-call "
    'objects of the form {"function": "Tool.name", '
    '"parameters": {}}; use [] when a turn has no call. Do not include '
    "markdown fences, explanations, or any text outside this JSON array."
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_backend(model: str | Path, requested: str) -> str:
    """Select an inference backend without importing CUDA-heavy libraries.

    vLLM 0.8.x in the evaluation environment does not implement Qwen3-VL or
    Mellum2.  ``auto`` therefore uses the native Transformers implementation
    for those architectures and keeps vLLM for the text-only Qwen3/OLMO
    checkpoints.  An explicit backend remains available for debugging and
    future vLLM upgrades.
    """
    requested = str(requested).lower()
    if requested not in {"auto", "vllm", "transformers"}:
        raise ValueError(f"Unknown backend {requested!r}; use auto, vllm or transformers")
    if requested != "auto":
        return requested
    config_path = Path(model).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "vllm"
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(str(x).lower() for x in config.get("architectures", []))
    if model_type in {"qwen3_vl", "mellum"} or "qwen3vl" in architectures:
        return "transformers"
    return "vllm"


def _canonical_task(name: str) -> str:
    """Resolve a task name case-insensitively while retaining its spelling."""
    for task in TASK_REGISTRY:
        if task.lower() == name.lower():
            return task
    return name


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalar containers into JSON/pickle-safe values."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _content_to_text(content: Any) -> str:
    content = _to_builtin(content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text_parts.append(str(item.get("text", "")))
        return "\n".join(x for x in text_parts if x)
    return "" if content is None else str(content)


def _normalize_messages(raw: Any) -> list[dict[str, Any]] | None:
    """Normalize common parquet/JSONL prompt formats to chat messages."""
    raw = _to_builtin(raw)
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "messages" in raw:
            return _normalize_messages(raw["messages"])
        if "role" in raw and "content" in raw:
            return [raw]
        if "content" in raw:
            return [{"role": "user", "content": raw["content"]}]
        return None
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if isinstance(raw, list):
        if not raw:
            return None
        if all(isinstance(x, dict) and "role" in x for x in raw):
            return [dict(x) for x in raw]
        if len(raw) == 1:
            return _normalize_messages(raw[0])
        if all(isinstance(x, str) for x in raw):
            return [{"role": "user", "content": "\n".join(raw)}]
    return None


def _extract_question(messages: list[dict[str, Any]] | None, row: dict[str, Any]) -> str:
    for key in ("query", "question", "problem", "input"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if messages:
        user_text = []
        for message in messages:
            if message.get("role") in {"user", "human"}:
                text = _content_to_text(message.get("content"))
                if text:
                    user_text.append(text)
        if user_text:
            return "\n".join(user_text).strip()
        return _content_to_text(messages[-1].get("content")).strip()
    for key in ("prompt", "text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "gold_answer", "gold", "ground_truth", "target", "label"):
        value = row.get(key)
        if value is not None:
            if isinstance(value, (dict, list)):
                return json.dumps(_to_builtin(value), ensure_ascii=False, sort_keys=True)
            return str(value).strip()
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth") or reward_model.get("answer")
        if value is not None:
            return str(value).strip()
    return ""


def _coerce_example_id(value: Any, fallback: int) -> int | str:
    """Keep benchmark string IDs while normalizing numeric IDs for sorting."""
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or fallback


def _iter_rows(filepath: Path) -> Iterable[dict[str, Any]]:
    suffix = filepath.suffix.lower()
    if suffix == ".parquet":
        # Prefer the Arrow reader.  It avoids importing pandas (and its optional
        # NumPy extensions), which is important on the shared evaluation
        # environment where pandas/numexpr may be ABI-incompatible with the
        # active NumPy.  ``ParquetFile.iter_batches`` also preserves nested
        # prompt/reward-model structs as ordinary Python dictionaries.
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    yield {str(k): _to_builtin(v) for k, v in row.items()}
            return
        except Exception as arrow_exc:  # pragma: no cover - fallback depends on ABI
            # Keep compatibility with installations that have pandas but no
            # pyarrow (or an older Arrow reader that cannot decode the file).
            try:
                import pandas as pd
            except Exception as pandas_exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "Reading parquet requires a working pyarrow or pandas install; "
                    "use JSONL/JSON or repair the parquet Python environment."
                ) from pandas_exc
            try:
                frame = pd.read_parquet(filepath)
            except Exception as pandas_read_exc:
                raise RuntimeError(
                    f"Unable to read parquet file {filepath} with pyarrow or pandas"
                ) from pandas_read_exc
            # Keep the Arrow failure attached for debugging without exposing a
            # noisy traceback during normal CLI use.
            del arrow_exc
            for i in range(len(frame)):
                yield {str(k): _to_builtin(v) for k, v in frame.iloc[i].to_dict().items()}
        return
    if suffix in {".jsonl", ".json"}:
        with filepath.open(encoding="utf-8") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                if isinstance(payload, dict):
                    payload = payload.get("data", payload.get("examples", [payload]))
                if isinstance(payload, dict):
                    payload = [payload]
                if not isinstance(payload, list):
                    raise ValueError(
                        f"Expected a JSON object or list of objects in {filepath}"
                    )
                for row in payload:
                    if isinstance(row, dict):
                        yield row
                return
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{filepath}:{line_number} is not a JSON object")
                yield row
        return
    raise ValueError(f"Unsupported data format: {filepath} (use parquet/jsonl/json)")


def load_samples(filepath: str | Path, task_name: str | None = None) -> list[dict[str, Any]]:
    """Load parquet/JSONL rows into a common sample representation.

    The loader understands both JustRL parquet rows (``prompt`` plus
    ``reward_model.ground_truth``) and olympiad parquet rows (``problem`` plus
    ``answer``).  Preformatted ``messages``/multimodal content is preserved.
    """
    filepath = Path(filepath)
    task_name = _canonical_task(task_name or filepath.parent.name)
    samples = []
    for index, row in enumerate(_iter_rows(filepath)):
        raw_messages = row.get("messages", row.get("conversations"))
        if raw_messages is None:
            raw_messages = row.get("prompt")
        messages = _normalize_messages(raw_messages)
        question = _extract_question(messages, row)
        answer = _extract_answer(row)
        source_id = None
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            source_id = extra_info.get("index")
        if source_id is None:
            source_id = row.get("source_id")
        if source_id is None:
            source_id = row.get("id")
        if source_id is None:
            source_id = row.get("problem_idx")
        samples.append(
            {
                "example_id": _coerce_example_id(row.get("example_id"), index),
                "question": question,
                "answer": answer,
                "messages": messages,
                "source_id": str(source_id) if source_id is not None else None,
                "task": task_name,
                "choices": _to_builtin(row.get("choices")),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
                "precision": row.get("precision"),
                "metadata": _to_builtin(row.get("metadata")),
                "image_path": row.get("image_path"),
            }
        )
    if not samples:
        raise ValueError(f"No examples found in {filepath}")
    return samples


def _apply_prompt(sample: dict[str, Any], task_name: str, append_boxed: bool) -> list[dict[str, Any]]:
    """Create the chat messages sent to vLLM without mutating the sample."""
    messages = copy.deepcopy(sample.get("messages"))
    if not messages:
        messages = [{"role": "user", "content": sample["question"]}]
    if append_boxed and task_name in MATH_TASKS:
        # Math parquet data has one user message.  For a multimodal content
        # list, append a text item instead of converting away the image.
        target = None
        for message in reversed(messages):
            if message.get("role") in {"user", "human"}:
                target = message
                break
        if target is not None:
            content = target.get("content", "")
            if isinstance(content, list):
                content.append({"type": "text", "text": MATH_PROMPT_SUFFIX})
            else:
                target["content"] = f"{content}{MATH_PROMPT_SUFFIX}"
    if task_name == "BFCL":
        target = None
        for message in reversed(messages):
            if message.get("role") in {"user", "human"}:
                target = message
                break
        if target is not None:
            content = target.get("content", "")
            if isinstance(content, list):
                content.append({"type": "text", "text": BFCL_PROMPT_SUFFIX})
            else:
                target["content"] = f"{content}{BFCL_PROMPT_SUFFIX}"
    if task_name == "ARKitScenes":
        # The official Qwen3-VL Omni3D prompt is already stored in the
        # prepared sample.  Do not append the generic math instruction.
        pass
    return messages


def _parse_task_paths(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--task-path must be TASK=PATH, got {value!r}")
        task, path = value.split("=", 1)
        result[_canonical_task(task)] = Path(path).expanduser()
    return result


def resolve_task_path(
    task_name: str,
    data_dir: str | Path,
    overrides: dict[str, Path] | None = None,
) -> Path:
    task_name = _canonical_task(task_name)
    overrides = overrides or {}
    if task_name in overrides:
        path = overrides[task_name]
        if path.is_file():
            return path
        raise FileNotFoundError(f"--task-path for {task_name} does not exist: {path}")

    root = Path(data_dir).expanduser()
    relative = TASK_REGISTRY.get(task_name)
    candidates = []
    if relative:
        candidates.append(root / relative)
    # The local experiment snapshot uses ``VL_*`` directory names for the
    # text benchmarks evaluated with Qwen3-VL, while AIME26 is kept in the
    # shared MultiLingual data directory as ``aime2026.jsonl``.  Keep these
    # aliases here so a direct invocation with ``--data-dir`` behaves the same
    # as the dated wrappers (which pass an explicit ``--task-path``).
    local_aliases = {
        "AIME26": ("aime26.jsonl", "aime2026.jsonl"),
        "MMLU": ("VL_MMLU/test.jsonl", "VL_MMLU.jsonl"),
        "GSM8K": ("VL_GSM8K/test.jsonl", "VL_GSM8K.jsonl"),
        "MMMU": ("VL_MMMU/test.jsonl", "VL_MMMU.jsonl"),
        "MathVista-testmini": (
            "VL_MathVista/testmini.jsonl",
            "VL_MathVista.jsonl",
        ),
        "ARKitScenes": (
            "VL_ARKitScenes/test.jsonl",
            "VL_ARKitScenes.jsonl",
        ),
    }
    candidates.extend(root / alias for alias in local_aliases.get(task_name, ()))
    # AIME26 is maintained in the shared multilingual benchmark checkout
    # rather than in this repository's compact ``eval/data`` snapshot.  Keep
    # the absolute fallback here for direct CLI use; dated experiment wrappers
    # may still pass ``--task-path`` explicitly to make provenance obvious.
    if task_name == "AIME26":
        candidates.extend(
            Path(path)
            for path in (
                "/home/hongbo/MultiLingual/data/aime2026.jsonl",
                "/data/hongbo/MultiLingual/data/aime2026.jsonl",
            )
        )
    for suffix in (".parquet", ".jsonl", ".json"):
        candidates.extend(
            [
                root / task_name / f"test{suffix}",
                root / task_name.lower() / f"test{suffix}",
                root / f"{task_name.lower()}{suffix}",
                root / f"{task_name}{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    hint = " or use --task-path TASK=/absolute/path"
    raise FileNotFoundError(
        f"No data file found for {task_name} under {root}.{hint}"
    )


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got {value!r}") from exc
    if not values:
        raise ValueError("At least one integer is required")
    return values


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    # Keep ``1.0`` visible in filenames: it is part of the decoding manifest
    # and avoids confusing the final TEXT/VL profiles with legacy runs.
    return str(float(value))


def _last_boxed(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` expression, if present."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
    if start < 0:
        return None
    left = text.find("{", start)
    if left < 0:
        return None
    depth = 0
    for index in range(left, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _response_for_mode(raw_response: str, mode: str) -> str:
    if mode == "full":
        return raw_response
    boxed = _last_boxed(raw_response)
    return boxed or raw_response


def split_seeds(seeds: list[int], num_workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_workers)]
    for index, seed in enumerate(seeds):
        chunks[index % num_workers].append(seed)
    return [chunk for chunk in chunks if chunk]


def split_samples(samples: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    """Partition examples so every requested data-parallel GPU can work.

    Four seeds do not provide enough independent seed workers to occupy six
    cards.  Splitting examples instead lets each engine run all seeds for its
    disjoint shard, preserving exactly one response per (example, seed) while
    using every selected GPU.
    """
    chunks = [[] for _ in range(max(1, num_workers))]
    for index, sample in enumerate(samples):
        chunks[index % len(chunks)].append(sample)
    return [chunk for chunk in chunks if chunk]


def _worker_process_vllm(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one vLLM worker on one GPU or tensor-parallel GPU group."""
    gpu_group = payload["gpu_group"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_group)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    def progress(message: str) -> None:
        # The parent launcher redirects stdout to the dated queue log.  Keep
        # seed-level progress visible because vLLM's internal progress bar is
        # intentionally disabled for deterministic batch generation.
        print(
            f"[gen worker GPUs={','.join(map(str, gpu_group))}] {message}",
            flush=True,
        )

    # Import after CUDA_VISIBLE_DEVICES is set.  This also makes the parent
    # process safe to use for data inspection and --dry-run.
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": payload["model"],
        "tensor_parallel_size": int(payload["tensor_parallel_size"]),
        "dtype": payload["dtype"],
        "trust_remote_code": bool(payload["trust_remote_code"]),
        "enforce_eager": bool(payload["enforce_eager"]),
        "max_model_len": int(payload["max_model_len"]),
        "gpu_memory_utilization": float(payload["gpu_memory_utilization"]),
    }
    allowed_local_media_path = payload.get("allowed_local_media_path")
    if allowed_local_media_path:
        llm_kwargs["allowed_local_media_path"] = str(allowed_local_media_path)
    llm = LLM(**llm_kwargs)

    samples = payload["samples"]
    all_results: list[dict[str, Any]] = []
    conversations = [sample["messages"] for sample in samples]
    progress(
        f"loaded model; examples={len(samples)} seeds={payload['seeds']}"
    )
    for seed in payload["seeds"]:
        progress(f"start seed={seed}")
        sampling = SamplingParams(
            n=1,
            temperature=float(payload["temperature"]),
            top_p=float(payload["top_p"]),
            top_k=int(payload["top_k"]),
            repetition_penalty=float(payload["repetition_penalty"]),
            presence_penalty=float(payload["presence_penalty"]),
            max_tokens=int(payload["max_tokens"]),
            seed=int(seed),
        )
        outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
        for sample, output in zip(samples, outputs):
            completion = output.outputs[0]
            raw_response = str(completion.text)
            token_ids = getattr(completion, "token_ids", None) or []
            token_length = len(token_ids)
            finish_reason = getattr(completion, "finish_reason", None)
            truncated = finish_reason == "length" or token_length >= int(payload["max_tokens"])
            response = _response_for_mode(raw_response, payload["response_mode"])
            all_results.append(
                {
                    "example_id": sample["example_id"],
                    "source_id": sample.get("source_id"),
                    "question": sample["question"],
                    "prompt": sample["question"],
                    "answer": sample["answer"],
                    "seed": int(seed),
                    # Keep the complete trajectory when a caller asks for a
                    # reduced ``response`` representation.  In the default
                    # ``full`` mode ``response`` already is that trajectory,
                    # so avoid duplicating it on disk.
                    "raw_response": (
                        raw_response
                        if payload["response_mode"] != "full"
                        else None
                    ),
                    "response": response,
                    "token_length": token_length,
                    "finish_reason": finish_reason,
                    "truncated": bool(truncated),
                    "choices": sample.get("choices"),
                    "question_type": sample.get("question_type"),
                    "answer_type": sample.get("answer_type"),
                    "precision": sample.get("precision"),
                    "metadata": sample.get("metadata"),
                    "image_path": sample.get("image_path"),
                }
            )
        progress(f"finished seed={seed}; rows={len(all_results)}")
    return all_results


def _transformers_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style multimodal messages to Qwen3-VL messages.

    The prepared MMMU snapshot intentionally uses ``image_url`` blocks so it
    can also be consumed by vLLM/OpenAI-compatible clients.  Native
    Qwen3-VL's processor expects ``image`` blocks, and it expects text-only
    content to be represented as a list of typed blocks rather than a bare
    string.  Keep this conversion local to the Transformers fallback so the
    vLLM path receives the original schema unchanged.
    """
    converted: list[dict[str, Any]] = []
    for message in copy.deepcopy(messages):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, dict):
            content = [content]
            message["content"] = content
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, str):
                    blocks.append({"type": "text", "text": block})
                    continue
                if not isinstance(block, dict):
                    continue
                block = dict(block)
                block_type = block.get("type")
                if block_type in {"input_text", "text"}:
                    blocks.append({"type": "text", "text": str(block.get("text", ""))})
                elif block_type == "image_url":
                    image_url = block.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    blocks.append({"type": "image", "image": image_url})
                elif block_type == "image":
                    # Accept both the native Qwen form and a nested OpenAI
                    # image_url accidentally left in a hand-written sample.
                    if isinstance(block.get("image"), dict):
                        image = block["image"].get("url")
                        blocks.append({"type": "image", "image": image})
                    else:
                        blocks.append(block)
                else:
                    blocks.append(block)
            message["content"] = blocks
        converted.append(message)
    return converted


class _PresencePenalty:
    """A small Transformers LogitsProcessor matching vLLM's presence penalty."""

    def __init__(self, prompt_lengths: list[int], penalty: float):
        self.prompt_lengths = prompt_lengths
        self.penalty = float(penalty)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        for row_index, token_row in enumerate(input_ids):
            prompt_length = self.prompt_lengths[min(row_index, len(self.prompt_lengths) - 1)]
            generated = token_row[int(prompt_length) :]
            if generated.numel() == 0:
                continue
            token_ids = torch.unique(generated)
            scores[row_index, token_ids] -= self.penalty
        return scores


def _worker_process_transformers_text(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a generic text-only Transformers worker (currently Mellum2).

    Mellum2 support landed after the Transformers version bundled with the
    vLLM evaluation environment, so the queue gives it a small dedicated
    environment with a current Transformers release.  Keep this path separate
    from the Qwen3-VL processor path: Mellum is a normal causal LM and its
    tokenizer/chat template must be handled through ``AutoTokenizer``.
    """
    gpu_group = payload["gpu_group"]
    if len(gpu_group) != 1:
        raise ValueError(
            "Transformers text fallback requires one GPU per worker; "
            f"received group {gpu_group}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_group[0])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    def progress(message: str) -> None:
        print(
            f"[transformers text worker GPUs={','.join(map(str, gpu_group))}] {message}",
            flush=True,
        )

    model_path = payload["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(payload["trust_remote_code"])
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = str(payload.get("dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        trust_remote_code=bool(payload["trust_remote_code"]),
    )
    model.eval()
    device = next(model.parameters()).device
    progress(f"loaded model; examples={len(payload['samples'])} seeds={payload['seeds']}")

    all_results: list[dict[str, Any]] = []
    for seed in payload["seeds"]:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        progress(f"start seed={seed}")
        for sample in payload["samples"]:
            messages = copy.deepcopy(sample["messages"])
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            prompt_length = int(inputs["attention_mask"][0].sum().item())
            logits_processors = LogitsProcessorList()
            presence_penalty = float(payload["presence_penalty"])
            if presence_penalty:
                logits_processors.append(_PresencePenalty([prompt_length], presence_penalty))
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(payload["max_tokens"]),
                "do_sample": float(payload["temperature"]) > 0.0,
                "temperature": float(payload["temperature"]),
                "top_p": float(payload["top_p"]),
                "top_k": int(payload["top_k"]),
                "repetition_penalty": float(payload["repetition_penalty"]),
                "return_dict_in_generate": True,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if logits_processors:
                generation_kwargs["logits_processor"] = logits_processors
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            sequences = generated.sequences
            completion_ids = sequences[0, prompt_length:]
            raw_response = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            token_length = int(completion_ids.numel())
            finish_reason = "length" if token_length >= int(payload["max_tokens"]) else "stop"
            response = _response_for_mode(raw_response, payload["response_mode"])
            all_results.append(
                {
                    "example_id": sample["example_id"],
                    "source_id": sample.get("source_id"),
                    "question": sample["question"],
                    "prompt": sample["question"],
                    "answer": sample["answer"],
                    "seed": int(seed),
                    "raw_response": (
                        raw_response if payload["response_mode"] != "full" else None
                    ),
                    "response": response,
                    "token_length": token_length,
                    "finish_reason": finish_reason,
                    "truncated": bool(finish_reason == "length"),
                }
            )
        progress(f"finished seed={seed}; rows={len(all_results)}")
    return all_results


def _worker_process_transformers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a native Transformers worker for architectures unsupported by vLLM.

    The fallback deliberately processes one conversation at a time.  This is
    slower than vLLM batching, but it bounds multimodal memory and preserves
    the exact four-rollout-per-example protocol on a single reserved card.
    """
    gpu_group = payload["gpu_group"]
    if len(gpu_group) != 1:
        raise ValueError(
            "Transformers fallback requires one GPU per worker; "
            f"received group {gpu_group}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_group[0])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoProcessor, LogitsProcessorList

    config_path = Path(payload["model"]) / "config.json"
    try:
        model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")).lower()
    except (OSError, json.JSONDecodeError):
        model_type = ""
    if model_type in {"mellum", "iquestcoder"}:
        # This branch is reached only for callers that explicitly select the
        # generic Transformers backend.  ``auto`` resolves Mellum before the
        # worker pool is created, but retaining the guard makes direct calls
        # robust and keeps the architecture dispatch in one place.  IQuest
        # Coder is also a text-only custom causal LM; route it through the
        # same generic worker when the vLLM backend cannot load its remote
        # architecture.
        return _worker_process_transformers_text(payload)

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Qwen3-VL Transformers fallback requires a recent transformers "
            "installation with Qwen3VLForConditionalGeneration"
        ) from exc

    def progress(message: str) -> None:
        print(
            f"[transformers worker GPUs={','.join(map(str, gpu_group))}] {message}",
            flush=True,
        )

    model_path = payload["model"]
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=bool(payload["trust_remote_code"])
    )
    # Qwen3-VL recommends bfloat16 on the experiment's cards.  Use the
    # long-standing ``torch_dtype`` keyword so both the current model class and
    # older Transformers releases accept the fallback.
    dtype_name = str(payload.get("dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name, torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        trust_remote_code=bool(payload["trust_remote_code"]),
    )
    model.eval()
    device = next(model.parameters()).device
    progress(f"loaded model; examples={len(payload['samples'])} seeds={payload['seeds']}")

    all_results: list[dict[str, Any]] = []
    for seed in payload["seeds"]:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        progress(f"start seed={seed}")
        for sample in payload["samples"]:
            messages = _transformers_messages(sample["messages"])
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            prompt_lengths = [int(inputs["attention_mask"][index].sum().item()) for index in range(inputs["attention_mask"].shape[0])]
            logits_processors = LogitsProcessorList()
            presence_penalty = float(payload["presence_penalty"])
            if presence_penalty:
                logits_processors.append(_PresencePenalty(prompt_lengths, presence_penalty))
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(payload["max_tokens"]),
                "do_sample": float(payload["temperature"]) > 0.0,
                "temperature": float(payload["temperature"]),
                "top_p": float(payload["top_p"]),
                "top_k": int(payload["top_k"]),
                "repetition_penalty": float(payload["repetition_penalty"]),
                "return_dict_in_generate": True,
                "use_cache": True,
            }
            if logits_processors:
                generation_kwargs["logits_processor"] = logits_processors
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            sequences = generated.sequences
            # This worker uses one conversation at a time, so the first row is
            # the only generated completion.  Use the attention-mask length
            # rather than padded input_ids length for correctness.
            prompt_length = prompt_lengths[0]
            completion_ids = sequences[0, prompt_length:]
            raw_response = processor.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            token_length = int(completion_ids.numel())
            finish_reason = "length" if token_length >= int(payload["max_tokens"]) else "stop"
            response = _response_for_mode(raw_response, payload["response_mode"])
            all_results.append(
                {
                    "example_id": sample["example_id"],
                    "source_id": sample.get("source_id"),
                    "question": sample["question"],
                    "prompt": sample["question"],
                    "answer": sample["answer"],
                    "seed": int(seed),
                    "raw_response": (
                        raw_response
                        if payload["response_mode"] != "full"
                        else None
                    ),
                    "response": response,
                    "token_length": token_length,
                    "finish_reason": finish_reason,
                    "truncated": bool(finish_reason == "length"),
                }
            )
        progress(f"finished seed={seed}; rows={len(all_results)}")
    return all_results


def _worker_process(payload: dict[str, Any]) -> list[dict[str, Any]]:
    backend = str(payload.get("backend", "vllm")).lower()
    if backend == "transformers":
        return _worker_process_transformers(payload)
    return _worker_process_vllm(payload)


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SAR benchmark responses with vLLM")
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--tasks", nargs="+", required=True, help="Benchmark names")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=str(_default_data_dir()))
    parser.add_argument(
        "--task-path",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override a task data file; repeat for multiple tasks",
    )

    # Structured output.  --output-dir points directly at one benchmark dir;
    # when several tasks are requested, each task gets a child directory.
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--modality", choices=["TEXT", "VL"], default="TEXT")
    parser.add_argument("--model-name", help="Directory level, e.g. Qwen3-4B")
    parser.add_argument("--variant", help="Directory level, e.g. instruct/thinking/sar_ratio_1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Restrict generation to the first N examples (intended for smoke tests)",
    )

    parser.add_argument("--gpus", default="1,2,5,6,7,8", help="Comma-separated physical GPU ids")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument(
        "--allow-other-gpus",
        action="store_true",
        help="Allow ids outside the experiment's 1,2,5,6,7,8 allow-list",
    )

    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n", "--batch-size", dest="n", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float)
    parser.add_argument("--presence-penalty", "--presence_penalty", dest="presence_penalty", type=float)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int)
    parser.add_argument("--max-model-len", "--max_model_len", dest="max_model_len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument(
        "--allowed-local-media-path",
        help="Root directory allowed for file:// images passed to vLLM",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "transformers"],
        default="auto",
        help="Inference backend; auto uses native Transformers for Qwen3-VL",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--response-mode",
        choices=["full", "boxed_final_only"],
        default="full",
        help="Store full completion or only its last boxed expression",
    )
    parser.add_argument("--legacy-filename", action="store_true")
    parser.add_argument("--filename", help="Explicit JSONL filename (one task only)")
    parser.add_argument(
        "--no-math-prompt",
        action="store_true",
        help="Do not append the boxed-answer math instruction",
    )
    return parser


def _resolve_decoding(args: argparse.Namespace, task: str) -> dict[str, Any]:
    is_vl = args.modality == "VL"
    visual = task in VL_VISUAL_TASKS
    defaults = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0 if visual else 1.5,
        "max_tokens": 40960 if (is_vl and visual) else 32768,
        "max_model_len": 40960,
    }
    for key in tuple(defaults):
        value = getattr(args, key)
        if value is not None:
            defaults[key] = value
    if defaults["temperature"] < 0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < float(defaults["top_p"]) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if defaults["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be positive")
    if defaults["top_k"] < -1:
        raise ValueError("top_k must be -1 (disabled) or non-negative")
    if defaults["max_tokens"] <= 0 or defaults["max_model_len"] <= 0:
        raise ValueError("max_tokens and max_model_len must be positive")
    if not 0.0 < float(args.gpu_memory_utilization) <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    return defaults


def _output_dir(args: argparse.Namespace, task: str, tasks: list[str]) -> Path:
    if args.output_dir:
        root = Path(args.output_dir).expanduser()
        return root if len(tasks) == 1 else root / task
    if not args.model_name or not args.variant:
        raise ValueError(
            "Provide --output-dir, or provide both --model-name and --variant "
            "for results/<modality>/<model>/<variant>/<benchmark>"
        )
    return Path(args.results_root) / args.modality / args.model_name / args.variant / task


def _filename(args: argparse.Namespace, task: str, decoding: dict[str, Any], n: int) -> str:
    if args.filename:
        if len(args.tasks) != 1:
            raise ValueError("--filename can only be used with one task")
        return args.filename
    if args.legacy_filename:
        return (
            f"{task.lower()}_t{_format_number(decoding['temperature'])}"
            f"_p{_format_number(decoding['top_p'])}_n{n}"
            f"-MNT{decoding['max_tokens']}.jsonl"
        )
    return (
        f"{task.lower()}_t{_format_number(decoding['temperature'])}"
        f"_p{_format_number(decoding['top_p'])}"
        f"_k{decoding['top_k']}"
        f"_rp{_format_number(decoding['repetition_penalty'])}"
        f"_pp{_format_number(decoding['presence_penalty'])}"
        f"_n{n}-MNT{decoding['max_tokens']}.jsonl"
    )


def _natural_example_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    tasks = [_canonical_task(x) for x in args.tasks]
    unknown = [x for x in tasks if x not in TASK_REGISTRY]
    if unknown:
        parser.error(f"Unknown task(s): {', '.join(unknown)}")

    try:
        seeds = _parse_int_list(args.seeds)
        if args.n is not None:
            if args.n <= 0 or args.n > len(seeds):
                raise ValueError("--n must be between 1 and the number of --seeds")
            seeds = seeds[: args.n]
        gpu_ids = _parse_int_list(args.gpus)
        if not args.allow_other_gpus:
            bad = sorted(set(gpu_ids) - ALLOWED_GPU_IDS)
            if bad:
                raise ValueError(
                    f"GPU ids {bad} are disabled for this experiment; allowed ids are "
                    f"{sorted(ALLOWED_GPU_IDS)} (use --allow-other-gpus only when intentional)"
                )
        if args.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if len(gpu_ids) % args.tensor_parallel_size:
            raise ValueError("number of GPUs must be divisible by tensor_parallel_size")
        task_overrides = _parse_task_paths(args.task_path)
        backend = _resolve_backend(args.model, args.backend)
        if backend == "transformers" and args.tensor_parallel_size != 1:
            raise ValueError("Transformers backend currently requires --tp 1")
    except ValueError as exc:
        parser.error(str(exc))

    requested_gpu_groups = [
        gpu_ids[i : i + args.tensor_parallel_size]
        for i in range(0, len(gpu_ids), args.tensor_parallel_size)
    ]

    for task in tasks:
        try:
            data_path = resolve_task_path(task, args.data_dir, task_overrides)
            samples = load_samples(data_path, task)
            if args.limit is not None:
                samples = samples[: args.limit]
            decoding = _resolve_decoding(args, task)
            output_dir = _output_dir(args, task, tasks)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = _filename(args, task, decoding, len(seeds))
            output_file = output_dir / filename
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        # Use one data-parallel engine per selected GPU group.  Each engine
        # receives a disjoint example shard and all requested seeds, so six
        # available cards are all useful even though the protocol has only
        # four rollouts per example.
        gpu_groups = requested_gpu_groups[: min(len(requested_gpu_groups), len(samples))]
        sample_chunks = split_samples(samples, len(gpu_groups))
        effective_gpu_ids = [gpu for group in gpu_groups for gpu in group]

        for sample in samples:
            sample["messages"] = _apply_prompt(
                sample, task, append_boxed=not args.no_math_prompt
            )

        manifest = {
            "schema_version": 2,
            "model": str(Path(args.model).expanduser().resolve()),
            "model_name": args.model_name,
            "variant": args.variant,
            "modality": args.modality,
            "benchmark": task,
            "data_path": str(data_path.resolve()),
            "num_examples": len(samples),
            "seeds": seeds,
            "gpus": effective_gpu_ids,
            "requested_gpus": gpu_ids,
            "tensor_parallel_size": args.tensor_parallel_size,
            "backend": backend,
            "allowed_local_media_path": args.allowed_local_media_path,
            "decoding": {
                **decoding,
                "greedy": False,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "dtype": args.dtype,
                "enforce_eager": args.enforce_eager,
                "trust_remote_code": args.trust_remote_code,
                "response_mode": args.response_mode,
            },
            "output_file": str(output_file.resolve()),
            "runner": {
                "script": str(Path(__file__).resolve()),
                "python": sys.version.split()[0],
                "backend": backend,
                "vllm": _package_version("vllm"),
                "transformers": _package_version("transformers"),
            },
        }
        # A dry run is an inspection operation: it must still print the
        # manifest/derived filename when the target already exists.  Only a
        # real generation run treats an existing file as an error.
        if output_file.exists() and not args.overwrite and not args.dry_run:
            parser.error(f"Output exists; use --overwrite or change parameters: {output_file}")

        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if args.dry_run:
            continue

        payloads = []
        for gpu_group, sample_chunk in zip(gpu_groups, sample_chunks):
            payloads.append(
                {
                    "gpu_group": gpu_group,
                    "seeds": seeds,
                    "samples": sample_chunk,
                    "model": str(Path(args.model).expanduser().resolve()),
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "backend": backend,
                    **decoding,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "dtype": args.dtype,
                    "enforce_eager": args.enforce_eager,
                    "trust_remote_code": args.trust_remote_code,
                    "response_mode": args.response_mode,
                    "allowed_local_media_path": args.allowed_local_media_path,
                }
            )

        all_results: list[dict[str, Any]] = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(payloads), mp_context=context
        ) as executor:
            futures = [executor.submit(_worker_process, payload) for payload in payloads]
            for future in tqdm(as_completed(futures), total=len(futures), desc=task):
                all_results.extend(future.result())

        all_results.sort(
            key=lambda row: (
                _natural_example_key(row["example_id"]),
                seeds.index(row["seed"]) if row["seed"] in seeds else len(seeds),
            )
        )
        with output_file.open("w", encoding="utf-8") as handle:
            for row in all_results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["num_rows"] = len(all_results)
        manifest["rows_per_example"] = {
            str(example_id): sum(1 for row in all_results if row["example_id"] == example_id)
            for example_id in sorted({row["example_id"] for row in all_results}, key=_natural_example_key)
        }
        with (output_dir / "generation_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_results)} rows -> {output_file}")


if __name__ == "__main__":
    main()
