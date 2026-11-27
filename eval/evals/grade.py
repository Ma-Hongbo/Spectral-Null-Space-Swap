"""Grade SAR JSONL generations and report accuracy/token metrics.

The grader accepts both the archived filename
``hmmt25_t0.6_p0.95_n4-MNT32768.jsonl`` and the new explicit filename format
written by :mod:`gen_vllm`.  It groups rows by ``example_id`` (rather than
assuming that the first example has four rows), preserves questions for
CompassVerifier fallback grading, and reports AVG@n / Pass@n / per-seed
Pass@1, token statistics, and truncation rate.

LiveCodeBench and BFCL use their bundled reference evaluators when available.
MMMU is a multiple-choice benchmark, so ``--evaluator auto`` performs the
standard letter-exact answer check while retaining the official sample IDs.
If an external evaluator is unavailable, token/rollout diagnostics are still
written together with the evaluator error.

Verifier loading is lazy, so rule-perfect math files can be graded without
initializing a second vLLM model.  The default verifier path is the complete
local checkpoint found at ``/data/hongbo/sar/CompassVerifier-3B``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
import zlib
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .utils import (
        extract_answer,
        grade_answer_mathd,
        grade_answer_sympy,
        grade_answer_verl,
    )
except ImportError:  # Running as ``python eval/evals/grade.py``.
    from utils import (  # type: ignore
        extract_answer,
        grade_answer_mathd,
        grade_answer_sympy,
        grade_answer_verl,
    )


# Prefer the local NVMe mirror.  On this host the /data mergerfs mount can
# block even a metadata-only ``Path.is_dir`` while another disk is busy; a
# grading command must not hang before it has parsed its arguments.
_VERIFIER_CANDIDATES = (
    os.environ.get("COMPASS_VERIFIER_MODEL", ""),
    "/mnt/disk3/hongbo/sar/CompassVerifier-3B",
    "/data/hongbo/sar/CompassVerifier-3B",
)
DEFAULT_VERIFIER = next((p for p in _VERIFIER_CANDIDATES if p and Path(p).is_dir()), None)
if DEFAULT_VERIFIER is None:
    DEFAULT_VERIFIER = "opencompass/CompassVerifier-3B"

# Current operator authorization: GPU4 is ERROR; all other visible cards may
# be used by the verifier when an explicit GPU is supplied.
ALLOWED_GPU_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 9}
OFFICIAL_EXTERNAL_TASKS = {"livecodebench", "bfcl"}
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
HYPERPARAMETER_PATTERN = re.compile(
    rf"_t(?P<temperature>{FLOAT_PATTERN})"
    rf"_p(?P<top_p>{FLOAT_PATTERN})"
    rf"(?:_k(?P<top_k>-?\d+)"
    rf"_rp(?P<repetition_penalty>{FLOAT_PATTERN})"
    rf"_pp(?P<presence_penalty>{FLOAT_PATTERN}))?"
    rf"_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)",
    flags=re.IGNORECASE,
)

CV_PROMPT = """You are a strict benchmark answer grader.
Compare only the candidate's final answer with the standard answer; ignore the
candidate's reasoning. The standard answer and question are valid. Equivalent
mathematical expressions and equivalent multiple-choice formatting count as
correct. A truncated, empty, repetitive, or irrelevant response is INVALID.

Return exactly one label: A (CORRECT), B (INCORRECT), or C (INVALID).

<Original Question>
{question}
</Original Question>
<Standard Answer>
{gold_answer}
</Standard Answer>
<Candidate Answer>
{llm_response}
</Candidate Answer>
"""


def _gpu_state() -> dict[int, tuple[int, int]]:
    """Return ``gpu_id -> (MiB used, utilization %)`` when nvidia-smi works."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}
    state: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            state[int(fields[0])] = (int(fields[1]), int(fields[2]))
        except ValueError:
            continue
    return state


def _ready_gpu_ids() -> set[int] | None:
    """Return CUDA-ready cards, excluding cards with a stale driver state."""
    state = _gpu_state()
    if not state:
        return None
    ready: set[int] = set()
    for gpu in state:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "-q", "-d", "PERFORMANCE"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and "System is not in ready state" not in probe.stdout:
            ready.add(gpu)
    return ready


def _select_verifier_gpus(requested: str | None, tensor_parallel: int) -> str | None:
    """Keep verifier grading off a busy card when another authorized card is idle.

    Generation and grading can overlap on disjoint cards.  Dated wrappers pass
    the first generation card as ``--gpus`` by default, but that card may still
    be occupied by another benchmark's verifier.  We therefore retain the
    requested group when it is idle and otherwise choose an idle authorized
    group of the same TP width.  If utilization cannot be queried, or no
    suitable group exists, the requested value is preserved and the caller
    retains the original behavior.
    """
    if not requested:
        return requested
    try:
        requested_ids = [int(value.strip()) for value in requested.split(",") if value.strip()]
    except ValueError:
        return requested
    if not requested_ids or tensor_parallel <= 0:
        return requested
    state = _gpu_state()
    if not state:
        return requested
    ready = _ready_gpu_ids()

    def is_idle(group: list[int]) -> bool:
        return all(
            gpu in state
            and (ready is None or gpu in ready)
            and state[gpu][0] <= 2000
            and state[gpu][1] <= 5
            for gpu in group
        )

    if len(requested_ids) == tensor_parallel and is_idle(requested_ids):
        return requested

    idle_ids = sorted(
        gpu
        for gpu in ALLOWED_GPU_IDS
        if gpu in state
        and (ready is None or gpu in ready)
        and state[gpu][0] <= 2000
        and state[gpu][1] <= 5
    )
    if len(idle_ids) < tensor_parallel:
        return requested
    # TP groups are represented as a comma-separated list; vLLM does not
    # require physical adjacency, only the requested group size.
    selected = idle_ids[:tensor_parallel]
    return ",".join(str(gpu) for gpu in selected)


def parse_hyperparameters_from_filename(filename: str | Path) -> dict[str, str]:
    """Parse both legacy and explicit generation filenames.

    The values intentionally remain strings for compatibility with the old
    script; callers can cast them with :func:`_number` where needed.
    """
    name = Path(filename).name
    match = HYPERPARAMETER_PATTERN.search(name)
    if not match:
        return {}
    values = match.groupdict()
    return {key: value for key, value in values.items() if value is not None}


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _example_key(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _example_sort_key(value: Any) -> tuple[int, Any]:
    key = _example_key(value)
    return (0, key) if isinstance(key, int) else (1, str(key))


def _load_manifest(file_path: Path) -> dict[str, Any]:
    manifest_path = file_path.parent / "generation_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def process_jsonl_file(
    file_name: str | Path,
    seed_order: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Group JSONL rows by problem and retain all generation metadata.

    Old files may have no ``seed`` field.  In that case rows are kept in file
    order and receive synthetic ``rollout_1``, ``rollout_2`` labels so that
    AVG@n/Pass@n still work without pretending that the seeds are known.
    """
    file_name = Path(file_name)
    # Some archived generators did not write ``example_id``.  Their filename
    # still records ``n`` (usually 4), so recover the original problem groups
    # instead of treating every rollout as a separate problem.  New files
    # always carry an explicit ID and are unaffected by this fallback.
    legacy_n = _int_or_none(
        parse_hyperparameters_from_filename(file_name.name).get("n")
    ) or len(seed_order or []) or 1
    grouped: OrderedDict[int | str, dict[str, Any]] = OrderedDict()
    with file_name.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{file_name}:{line_number} is not a JSON object")
            raw_key = data.get("example_id")
            if raw_key is None:
                raw_key = data.get(
                    "problem_id", data.get("question_id", data.get("id"))
                )
            if raw_key is None:
                raw_key = (line_number - 1) // legacy_n
            key = _example_key(raw_key)
            item = grouped.setdefault(
                key,
                {
                    "example_id": key,
                    "question": "",
                    "gt": data.get("answer", data.get("ground_truth", "")),
                    "source_id": data.get("source_id"),
                    "responses": [],
                },
            )
            question = (
                data.get("question")
                or data.get("problem")
                or data.get("prompt")
                or data.get("input")
                or ""
            )
            if question and not item["question"]:
                item["question"] = str(question)
            if data.get("answer") is not None:
                item["gt"] = data.get("answer")
            if item.get("source_id") is None and data.get("source_id") is not None:
                item["source_id"] = data.get("source_id")
            response = data.get("response")
            if response in (None, ""):
                response = data.get("raw_response", "")
            token_length = _int_or_none(data.get("token_length"))
            seed = data.get("seed")
            seed = _int_or_none(seed) if seed is not None else None
            item["responses"].append(
                {
                    "response": "" if response is None else str(response),
                    "raw_response": data.get("raw_response"),
                    "seed": seed,
                    "token_length": token_length,
                    "finish_reason": data.get("finish_reason"),
                    "truncated": bool(data.get("truncated", False)),
                    "source_id": data.get("source_id"),
                }
            )

    items = [grouped[key] for key in sorted(grouped, key=_example_sort_key)]
    seed_rank = {seed: index for index, seed in enumerate(seed_order or [])}
    for item in items:
        responses = item["responses"]
        has_real_seeds = any(row["seed"] is not None for row in responses)
        if has_real_seeds:
            responses.sort(
                key=lambda row: (
                    seed_rank.get(row["seed"], len(seed_rank)),
                    row["seed"] if row["seed"] is not None else math.inf,
                )
            )
        for index, row in enumerate(responses, 1):
            if row["seed"] is None:
                row["rollout"] = f"rollout_{index}"
            else:
                row["rollout"] = str(row["seed"])
    return items


def get_diverse_score(sequences: Iterable[str], n: int = 4) -> float:
    distinct_ngrams = set()
    total_ngrams = 0
    for sequence in sequences:
        tokens = str(sequence).split()
        for index in range(len(tokens) - n + 1):
            distinct_ngrams.add(tuple(tokens[index : index + n]))
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams else 0.0


def truncate_prompt(tokenizer: Any, prompt: str, max_tokens: int = 32700) -> str:
    """Keep both the question prefix and candidate-answer suffix."""
    ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    if len(ids) <= max_tokens:
        return prompt
    head = max_tokens // 2
    tail = max_tokens - head
    ids = ids[:head] + ids[-tail:]
    return tokenizer.decode(ids, skip_special_tokens=False)


def _last_boxed_or_final(response: str) -> str:
    try:
        boxed = extract_answer(response)
        if boxed:
            return boxed
    except Exception:
        pass
    # GSM8K-style ``#### 42`` and simple final-answer lines.
    match = re.search(r"####\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""
    final = lines[-1]
    final = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?::|=|is)\s*",
        "",
        final,
        flags=re.I,
    )
    return final.strip("`* ")


def _simple_rule_grade(response: str, ground_truth: str) -> bool:
    """Math/short-answer rule grading, with a fallback for unboxed answers."""
    if not str(ground_truth).strip():
        return False
    # GSM8K stores a full chain-of-thought reference followed by the
    # canonical answer marker (``#### <answer>``).  The evaluator should
    # compare against that final answer rather than the whole rationale.
    gold_text = str(ground_truth).strip()
    if "####" in gold_text:
        gold_text = gold_text.rsplit("####", 1)[-1].strip()
    try:
        if grade_answer_verl(response, gold_text):
            return True
    except Exception:
        pass
    candidate = _last_boxed_or_final(response)
    if not candidate:
        return False
    # MMLU/MMLU-Pro and similar tasks use a single option letter.  Prefer the
    # last standalone option in the answer rather than a letter in the
    # reasoning.  MMLU-Pro can have up to ten choices (A-J), while ordinary
    # MMLU/ARC subsets use fewer; accepting the superset keeps the same rule
    # path.
    gold = gold_text.strip().strip(".$")
    if len(gold) == 1 and gold.upper() in "ABCDEFGHIJ":
        options = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", response.upper())
        return bool(options) and options[-1] == gold.upper()
    try:
        return grade_answer_mathd(candidate, gold_text) or grade_answer_sympy(
            candidate, gold_text
        )
    except Exception:
        return False


def _is_invalid(
    row: dict[str, Any],
    max_tokens_hint: int | None = None,
) -> tuple[bool, str]:
    response = str(row.get("response", ""))
    if not response.strip():
        return True, "empty"
    if row.get("truncated") or row.get("finish_reason") == "length":
        return True, "truncated"
    token_length = _int_or_none(row.get("token_length"))
    if max_tokens_hint and token_length is not None and token_length >= max_tokens_hint:
        # Legacy JSONL files did not store ``truncated``/``finish_reason``;
        # infer it from the MNT value encoded in their filename.
        return True, "truncated"
    return False, ""


def _strip_code_fence(response: str) -> str:
    """Extract a fenced code block when a model ignored the code-only prompt."""
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\s*\n?(.*?)```", response, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return response.strip()


def _lcb_test_cases(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "")
    try:
        payload = json.loads(text)
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload if isinstance(payload, list) else []
    except json.JSONDecodeError:
        payload = pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload if isinstance(payload, list) else []


def _official_livecodebench_grade(
    file_path: Path,
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    expected_n: int,
) -> tuple[dict[str, Any] | None, dict[int, list[bool]] | None, str | None]:
    """Run LiveCodeBench's sandboxed code-generation evaluator when available."""
    data_path = manifest.get("data_path")
    if data_path:
        data_path = Path(str(data_path)).expanduser()
        if not data_path.is_absolute():
            data_path = file_path.parent / data_path
    if not data_path or not data_path.is_file():
        return None, None, f"LiveCodeBench data_path is unavailable: {data_path!r}"
    try:
        source_rows: dict[str, dict[str, Any]] = {}
        source_rows_by_index: dict[int, dict[str, Any]] = {}
        source_rows_by_example_id: dict[str, dict[str, Any]] = {}
        with data_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                source_rows[str(row.get("source_id"))] = row
                source_rows_by_index[row_index] = row
                example_id = row.get("example_id", row_index)
                source_rows_by_example_id[str(example_id)] = row
        reference_root = Path(__file__).resolve().parents[2] / "references/LiveCodeBench"
        if not reference_root.is_dir():
            return None, None, f"LiveCodeBench reference is unavailable: {reference_root}"
        if str(reference_root) not in sys.path:
            sys.path.insert(0, str(reference_root))
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

        samples: list[dict[str, Any]] = []
        generations: list[list[str]] = []
        source_id_fallback_count = 0
        for item in items:
            raw_source_id = item["responses"][0].get("source_id") if item["responses"] else None
            source_id = str(raw_source_id) if raw_source_id not in (None, "") else ""
            source = source_rows.get(source_id)
            if source is None:
                # Older generation files predate source_id propagation but
                # retain the stable example_id and original row order.  Use
                # that only as an explicit, auditable compatibility fallback;
                # new files always take the source_id path above.
                example_key = str(item.get("example_id", ""))
                source = source_rows_by_example_id.get(example_key)
                if source is None:
                    try:
                        source = source_rows_by_index.get(int(item.get("example_id")))
                    except (TypeError, ValueError):
                        source = None
                if source is not None:
                    source_id_fallback_count += 1
            if source is None:
                return None, None, f"No LiveCodeBench source row for {source_id!r}"
            public = _lcb_test_cases(source.get("public_test_cases"))
            private = _lcb_test_cases(source.get("private_test_cases"))
            metadata = source.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            samples.append(
                {
                    "input_output": json.dumps(
                        {
                            "inputs": [case.get("input", "") for case in public + private],
                            "outputs": [case.get("output", "") for case in public + private],
                            "fn_name": metadata.get("func_name") if isinstance(metadata, dict) else None,
                        }
                    )
                }
            )
            generations.append([_strip_code_fence(str(row["response"])) for row in item["responses"]])

        workers = max(1, int(os.environ.get("SAR_LCB_EVAL_WORKERS", "8")))
        timeout = max(1, int(os.environ.get("SAR_LCB_TIMEOUT", "6")))
        k_values = sorted({1, max(1, expected_n)})
        metrics, results, _ = codegen_metrics(
            samples,
            generations,
            k_list=k_values,
            num_process_evaluate=workers,
            timeout=timeout,
            debug=False,
        )
        clean_metrics = {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        clean_metrics["num_problems"] = len(items)
        clean_metrics["rollouts_per_problem"] = expected_n
        clean_metrics["timeout_seconds"] = timeout
        clean_metrics["source_id_fallback_count"] = source_id_fallback_count
        collapsed = {
            int(key): [all(float(test_ok) > 0 for test_ok in generation) for generation in value]
            for key, value in results.items()
        }
        return clean_metrics, collapsed, None
    except Exception as exc:  # Keep grading/token metrics available on evaluator failure.
        return None, None, f"LiveCodeBench evaluator failed: {type(exc).__name__}: {exc}"


def _bfcl_call_to_python(value: Any) -> str | None:
    """Convert the JSON function-call schema used by our BFCL prompt to BFCL syntax."""
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if not isinstance(value, dict):
        return None
    function_name = value.get("function") or value.get("name")
    parameters = value.get("parameters", value.get("arguments", value.get("args")))
    if function_name is None:
        # Also accept the compact {"Tool.name": {...}} form emitted by some
        # function-calling checkpoints.
        reserved = {"turn", "calls", "function_calls", "tool_calls"}
        candidates = [(key, val) for key, val in value.items() if key not in reserved]
        if len(candidates) != 1:
            return None
        function_name, parameters = candidates[0]
    # BFCL's multi-turn checker executes methods on the class named by the
    # official prompt, so its ground-truth strings use ``mv(...)`` rather than
    # ``GorillaFileSystem.mv(...)``.  Accept either spelling from the model.
    function_name = str(function_name)
    if "." in function_name:
        function_name = function_name.rsplit(".", 1)[-1]
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return None
    arguments = ",".join(f"{key}={repr(val)}" for key, val in parameters.items())
    return f"{function_name}({arguments})"


def _bfcl_decode_response(response: str, num_turns: int) -> list[list[list[str]]] | None:
    """Decode one model response into the nested shape expected by BFCL's checker.

    The generation prompt asks for ``[[{function, parameters}, ...], ...]``
    (one list per turn).  A few models emit a wrapper object or a Python-call
    string, so those harmless variants are accepted as well.
    """
    text = _strip_code_fence(str(response)).strip()
    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Recover a JSON array when a model prepends a short explanation.
        left, right = text.find("["), text.rfind("]")
        if left < 0 or right <= left:
            return None
        try:
            payload = json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        payload = payload.get("turns", payload.get("responses", payload))
        if isinstance(payload, dict):
            ordered = []
            for key in sorted(payload, key=lambda item: str(item)):
                if str(key).lower().startswith(("turn", "step")):
                    ordered.append(payload[key])
            payload = ordered
    if not isinstance(payload, list):
        return None
    # With one turn, accepting a flat list of calls is convenient and remains
    # unambiguous.  For multiple turns the outer list must encode turn order.
    if num_turns == 1 and (not payload or isinstance(payload[0], dict)):
        payload = [payload]
    if len(payload) != num_turns:
        return None

    decoded: list[list[list[str]]] = []
    for turn in payload:
        if turn is None:
            turn = []
        if isinstance(turn, dict):
            turn = turn.get("calls", turn.get("function_calls", turn))
        if isinstance(turn, dict):
            turn = [turn]
        if isinstance(turn, str):
            turn = [turn]
        if not isinstance(turn, list):
            return None
        calls: list[str] = []
        for call in turn:
            # A nested list is treated as a second assistant step in the same
            # user turn; flattening it is equivalent for the BFCL state checker.
            if isinstance(call, list):
                nested = [_bfcl_call_to_python(item) for item in call]
                if any(item is None for item in nested):
                    return None
                calls.extend(item for item in nested if item is not None)
                continue
            converted = _bfcl_call_to_python(call)
            if converted is None:
                return None
            calls.append(converted)
        decoded.append([calls] if calls else [])
    return decoded


def _official_bfcl_grade(
    file_path: Path,
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    expected_n: int,
) -> tuple[dict[str, Any] | None, dict[int, list[bool]] | None, str | None]:
    """Run the repository's official BFCL multi-turn state checker."""
    reference_root = Path(__file__).resolve().parents[2] / "references/gorilla/berkeley-function-call-leaderboard"
    data_root = reference_root / "bfcl_eval/data"
    prompt_file = data_root / "BFCL_v4_multi_turn_base.json"
    answer_file = data_root / "possible_answer/BFCL_v4_multi_turn_base.json"
    if not prompt_file.is_file() or not answer_file.is_file():
        return None, None, f"BFCL reference data is unavailable under {data_root}"
    try:
        if str(reference_root) not in sys.path:
            sys.path.insert(0, str(reference_root))
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

        prompt_rows: dict[str, dict[str, Any]] = {}
        prompt_rows_by_index: dict[int, dict[str, Any]] = {}
        for row_index, line in enumerate(
            line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ):
            row = json.loads(line)
            prompt_rows[str(row["id"])] = row
            prompt_rows_by_index[row_index] = row
        answer_rows: dict[str, dict[str, Any]] = {}
        answer_rows_by_index: dict[int, dict[str, Any]] = {}
        for row_index, line in enumerate(
            line for line in answer_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ):
            row = json.loads(line)
            answer_rows[str(row["id"])] = row
            answer_rows_by_index[row_index] = row
        results: dict[int, list[bool]] = {}
        source_id_fallback_count = 0
        for item_index, item in enumerate(items):
            raw_source_id = item["responses"][0].get("source_id") if item["responses"] else None
            source_id = str(raw_source_id) if raw_source_id not in (None, "") else ""
            prompt_row = prompt_rows.get(source_id)
            answer_row = answer_rows.get(source_id)
            if prompt_row is None or answer_row is None:
                # Compatibility with pre-source_id generation files.  BFCL's
                # prepared prompt and answer files are aligned by row order;
                # retain an explicit count in the returned metrics.
                prompt_row = prompt_rows_by_index.get(item_index)
                answer_row = answer_rows_by_index.get(item_index)
                if prompt_row is not None and answer_row is not None:
                    source_id_fallback_count += 1
            if prompt_row is None or answer_row is None:
                return None, None, f"No BFCL reference row for {source_id!r}"
            ground_truth = answer_row.get("ground_truth")
            if not isinstance(ground_truth, list):
                return None, None, f"Malformed BFCL ground truth for {source_id!r}"
            num_turns = len(ground_truth)
            item_scores: list[bool] = []
            for row in item["responses"]:
                decoded = _bfcl_decode_response(str(row.get("response", "")), num_turns)
                valid = False
                if decoded is not None:
                    try:
                        checker_result = multi_turn_checker(
                            decoded,
                            ground_truth,
                            prompt_row,
                            "multi_turn_base",
                            "sar_local",
                        )
                        valid = bool(checker_result.get("valid"))
                    except Exception:
                        valid = False
                item_scores.append(valid)
            results[item_index] = item_scores
        correct = sum(score for scores in results.values() for score in scores)
        total = sum(len(scores) for scores in results.values())
        metrics = {
            "accuracy": correct / total if total else 0.0,
            "correct_count": correct,
            "total_count": total,
            "num_problems": len(items),
            "rollouts_per_problem": expected_n,
            "checker": "bfcl_eval.multi_turn_checker",
            "category": "multi_turn_base",
            "source_id_fallback_count": source_id_fallback_count,
        }
        return metrics, results, None
    except Exception as exc:
        return None, None, f"BFCL evaluator failed: {type(exc).__name__}: {exc}"


def _verdict(text: str) -> str:
    """Normalize CompassVerifier output to A/B/C."""
    normalized = re.sub(r"[^A-Z\s]", " ", str(text).upper()).strip()
    if re.match(r"^(?:A|CORRECT)(?:\s|$)", normalized):
        return "A"
    if re.match(r"^(?:B|INCORRECT)(?:\s|$)", normalized):
        return "B"
    if re.match(r"^(?:C|INVALID)(?:\s|$)", normalized):
        return "C"
    first = normalized.splitlines()[0] if normalized else ""
    if re.search(r"\bCORRECT\b", first) and not re.search(r"\bINCORRECT\b", first):
        return "A"
    if re.search(r"\bINCORRECT\b", first):
        return "B"
    if re.search(r"\bINVALID\b", first):
        return "C"
    return "C"


def _build_verifier_prompt(question: str, gold_answer: str, response: str) -> str:
    return CV_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        llm_response=response,
    )


class VerifierRunner:
    """Lazy vLLM CompassVerifier wrapper."""

    def __init__(self, args: argparse.Namespace):
        self.model = args.model
        self.tp = args.tp
        self.max_tokens = args.verifier_max_tokens
        self.temperature = args.verifier_temperature
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.dtype = args.dtype
        self.trust_remote_code = args.trust_remote_code
        self.max_input_tokens = args.max_input_tokens
        self.gpus = args.gpus
        self.tokenizer = None
        self.llm = None
        self.sampling_params = None

    def _ensure(self) -> None:
        if self.llm is not None:
            return
        if self.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tp,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=self.max_tokens,
            seed=0,
        )

    def _chat_prompt(self, prompt: str) -> str:
        assert self.tokenizer is not None
        try:
            prompt = truncate_prompt(self.tokenizer, prompt, self.max_input_tokens)
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return prompt

    def score(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        self._ensure()
        assert self.llm is not None and self.sampling_params is not None
        inputs = [self._chat_prompt(prompt) for prompt in prompts]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        return [_verdict(output.outputs[0].text) for output in outputs]


def _legacy_score(
    prompts: list[str],
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    max_input_tokens: int,
) -> list[str]:
    if not prompts:
        return []
    inputs = []
    for prompt in prompts:
        try:
            prompt = truncate_prompt(tokenizer, prompt, max_input_tokens)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
        inputs.append(prompt)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    return [_verdict(output.outputs[0].text) for output in outputs]


def _task_from_file(file_path: Path) -> str:
    stem = file_path.stem
    return stem.split("_t", 1)[0] if "_t" in stem else stem


def _seed_order_from_manifest(file_path: Path) -> list[int] | None:
    manifest = _load_manifest(file_path)
    values = manifest.get("seeds")
    if not isinstance(values, list):
        return None
    return [seed for seed in (_int_or_none(x) for x in values) if seed is not None]


def grade_file(
    file_path: str | Path,
    tokenizer: Any = None,
    llm: Any = None,
    sampling_params: Any = None,
    verbose: bool = False,
    *,
    verifier_runner: VerifierRunner | None = None,
    disable_verifier: bool = False,
    max_input_tokens: int = 32700,
    evaluator: str = "auto",
    write_details: bool = False,
) -> dict[str, Any] | None:
    """Grade one JSONL file and return a JSON-serializable summary."""
    file_path = Path(file_path)
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    manifest = _load_manifest(file_path)
    # ``--filename`` permits an arbitrary output name; the manifest is the
    # authoritative benchmark label in that case.  Fall back to the filename
    # for legacy JSONL files without a manifest.
    task_label = str(manifest.get("benchmark") or _task_from_file(file_path))
    task_name = task_label.lower()
    seed_order = _seed_order_from_manifest(file_path)
    items = process_jsonl_file(file_path, seed_order=seed_order)
    if not items:
        return None

    expected_n = _int_or_none(hyperparams.get("n"))
    if expected_n is None and seed_order:
        expected_n = len(seed_order)
    if expected_n is None:
        expected_n = max(len(item["responses"]) for item in items)
    max_tokens_hint = _int_or_none(hyperparams.get("max_tokens"))

    token_lengths: list[int] = []
    score_by_problem: dict[Any, list[bool]] = defaultdict(list)
    score_by_seed: dict[str, list[bool]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    fallback_jobs: list[tuple[int, str]] = []
    pending: list[dict[str, Any]] = []
    rule_correct = 0
    verifier_correct = 0
    verifier_count = 0
    invalid_count = 0
    truncated_count = 0
    missing_question_count = 0
    format_error_count = 0

    external_only = evaluator == "auto" and task_name in OFFICIAL_EXTERNAL_TASKS
    evaluation_mode = "official_external" if external_only else evaluator
    if evaluator == "auto" and task_name == "mmmu":
        evaluation_mode = "standard_option"
    official_metrics: dict[str, Any] | None = None
    official_results: dict[int, list[bool]] | None = None
    official_error: str | None = None

    for item in items:
        problem_detail = {
            "example_id": item["example_id"],
            "source_id": item.get("source_id"),
            "question": item.get("question", ""),
            "answer": item.get("gt", ""),
            "rollouts": [],
        }
        for response_index, row in enumerate(item["responses"]):
            if row.get("token_length") is not None:
                token_lengths.append(row["token_length"])
            invalid, invalid_reason = _is_invalid(row, max_tokens_hint)
            if invalid:
                invalid_count += 1
                if invalid_reason == "truncated":
                    truncated_count += 1
                format_error_count += 1
            if (
                row.get("truncated")
                or row.get("finish_reason") == "length"
                or (
                    max_tokens_hint
                    and _int_or_none(row.get("token_length")) is not None
                    and _int_or_none(row.get("token_length")) >= max_tokens_hint
                )
            ):
                truncated_count += 0 if invalid_reason == "truncated" else 1

            score: bool = False
            verdict = "C" if invalid else "B"
            method = "invalid" if invalid else "unverified"
            if not external_only and not invalid:
                if _simple_rule_grade(str(row["response"]), str(item["gt"])):
                    score = True
                    verdict = "A"
                    method = "rule"
                    rule_correct += 1
                elif item.get("question", ""):
                    prompt = _build_verifier_prompt(
                        str(item["question"]), str(item["gt"]), str(row["response"])
                    )
                    fallback_jobs.append((len(pending), prompt))
                    pending.append(
                        {
                            "item": item,
                            "row": row,
                            "response_index": response_index,
                            "verdict": verdict,
                            "method": method,
                        }
                    )
                    verifier_count += 1
                else:
                    missing_question_count += 1
                    verdict = "C"
                    method = "missing_question"

            problem_detail["rollouts"].append(
                {
                    "seed": row.get("seed"),
                    "rollout": row.get("rollout"),
                    "token_length": row.get("token_length"),
                    "finish_reason": row.get("finish_reason"),
                    "truncated": bool(row.get("truncated")),
                    "score": score,
                    "verdict": verdict,
                    "method": method,
                }
            )
            row["_score"] = score
            row["_verdict"] = verdict
            row["_method"] = method
            row["_detail"] = problem_detail["rollouts"][-1]
        details.append(problem_detail)

    if fallback_jobs and not disable_verifier and not external_only:
        prompts = [prompt for _, prompt in fallback_jobs]
        if verifier_runner is not None:
            verdicts = verifier_runner.score(prompts)
        elif tokenizer is not None and llm is not None and sampling_params is not None:
            verdicts = _legacy_score(
                prompts, tokenizer, llm, sampling_params, max_input_tokens
            )
        else:
            verdicts = ["C"] * len(prompts)
        for pending_item, verdict in zip(pending, verdicts):
            row = pending_item["row"]
            score = verdict == "A"
            row["_score"] = score
            row["_verdict"] = verdict
            row["_method"] = "verifier"
            row["_detail"]["score"] = score
            row["_detail"]["verdict"] = verdict
            row["_detail"]["method"] = "verifier"
            if score:
                verifier_correct += 1
    elif pending:
        for pending_item in pending:
            pending_item["row"]["_method"] = "verifier_disabled"
            pending_item["row"]["_detail"]["method"] = "verifier_disabled"

    if external_only and task_name == "livecodebench":
        official_metrics, official_results, official_error = _official_livecodebench_grade(
            file_path, items, manifest, expected_n
        )
        if official_metrics is not None:
            evaluation_mode = "official_livecodebench"
    elif external_only and task_name == "bfcl":
        official_metrics, official_results, official_error = _official_bfcl_grade(
            file_path, items, manifest, expected_n
        )
        if official_metrics is not None:
            evaluation_mode = "official_bfcl"

    # Apply official per-rollout results (when the external checker loaded).
    if external_only and official_results is not None:
        method = evaluation_mode
        for item_index, item in enumerate(items):
            scores = official_results.get(item_index, [])
            item_detail = next(
                (detail for detail in details if detail["example_id"] == item["example_id"]),
                None,
            )
            for response_index, row in enumerate(item["responses"]):
                invalid, _ = _is_invalid(row, max_tokens_hint)
                score = (
                    not invalid
                    and bool(scores[response_index])
                    if response_index < len(scores)
                    else False
                )
                row["_score"] = score
                row["_verdict"] = "A" if score else "B"
                row["_method"] = method
                if item_detail and response_index < len(item_detail["rollouts"]):
                    item_detail["rollouts"][response_index].update(
                        {"score": score, "verdict": row["_verdict"], "method": method}
                    )

    # Build per-problem and per-seed aggregates after fallback verdicts are set.
    for item in items:
        for row in item["responses"]:
            score = bool(row.get("_score", False))
            score_by_problem[item["example_id"]].append(score)
            seed_key = row.get("rollout") or str(row.get("seed"))
            score_by_seed[str(seed_key)].append(score)

    problem_means = [
        sum(scores) / len(scores) for scores in score_by_problem.values() if scores
    ]
    problem_pass = [any(scores) for scores in score_by_problem.values()]
    avg_at_n = statistics.mean(problem_means) if problem_means else 0.0
    pass_at_n = statistics.mean(problem_pass) if problem_pass else 0.0
    pass1_by_seed = {
        seed: statistics.mean(scores) if scores else 0.0
        for seed, scores in sorted(score_by_seed.items())
    }
    pass1_mean = statistics.mean(pass1_by_seed.values()) if pass1_by_seed else 0.0
    diverse_scores = [
        get_diverse_score([str(row["response"]) for row in item["responses"]])
        for item in items
    ]
    distinct_4gram = statistics.mean(diverse_scores) if diverse_scores else 0.0
    solve_none = sum(not any(scores) for scores in score_by_problem.values())
    solve_all = sum(all(scores) for scores in score_by_problem.values())
    missing_rollouts = sum(
        max(0, expected_n - len(item["responses"])) for item in items
    )
    extra_rollouts = sum(
        max(0, len(item["responses"]) - expected_n) for item in items
    )
    mean_tokens = statistics.mean(token_lengths) if token_lengths else 0.0
    std_tokens = statistics.pstdev(token_lengths) if len(token_lengths) > 1 else 0.0
    cv_tokens = std_tokens / mean_tokens if mean_tokens else 0.0

    officially_scored = official_metrics is not None
    scored = not external_only or officially_scored
    summary: dict[str, Any] = {
        "file": str(file_path),
        "task_name": task_label,
        "evaluation_mode": evaluation_mode,
        "hyperparameters": hyperparams,
        "manifest": manifest,
        "num_problems": len(items),
        "expected_rollouts_per_problem": expected_n,
        "total_rollouts": sum(len(item["responses"]) for item in items),
        "missing_rollouts": missing_rollouts,
        "extra_rollouts": extra_rollouts,
        "mean_score": avg_at_n if scored else None,
        "avg_at_n": avg_at_n if scored else None,
        "best_score": pass_at_n if scored else None,
        "pass_at_n": pass_at_n if scored else None,
        "pass1_by_seed": pass1_by_seed if scored else {},
        "pass1_mean": pass1_mean if scored else None,
        "solve_none": solve_none if scored else None,
        "solve_all": solve_all if scored else None,
        "distinct_4gram": distinct_4gram,
        "avg_token_length": mean_tokens,
        "std_token_length": std_tokens,
        "cv_token_length": cv_tokens,
        "truncated_rollouts": truncated_count,
        "truncation_rate": (
            truncated_count / sum(len(item["responses"]) for item in items)
            if sum(len(item["responses"]) for item in items)
            else 0.0
        ),
        "format_error_rollouts": format_error_count,
        "invalid_rollouts": invalid_count,
        "missing_question_rollouts": missing_question_count,
        "rule_correct_rollouts": rule_correct,
        "verifier_rollouts": verifier_count,
        "verifier_correct_rollouts": verifier_correct,
        "verifier_enabled": not disable_verifier and not external_only,
    }
    if official_metrics is not None:
        summary["official_metrics"] = official_metrics
    if official_error is not None:
        summary["official_error"] = official_error
    if write_details:
        detail_path = file_path.with_suffix(".grading_details.jsonl")
        with detail_path.open("w", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
        summary["details_file"] = str(detail_path)
    if verbose:
        print(
            f"[graded] {file_path}: problems={len(items)} "
            f"AVG@{expected_n}={summary['avg_at_n']} "
            f"Pass@{expected_n}={summary['pass_at_n']} "
            f"trunc={summary['truncation_rate']}"
        )
    return summary


def _discover_files(args: argparse.Namespace) -> list[Path]:
    directories = [Path(x).expanduser() for x in args.eval_dirs]
    if args.results_root:
        directories.append(Path(args.results_root).expanduser())
    if not directories:
        raise ValueError("Provide --eval-dir or --results-root")
    files: set[Path] = set()
    for directory in directories:
        if directory.is_file() and directory.suffix == ".jsonl":
            files.add(directory)
        elif directory.is_dir():
            # A results root contains TEXT/VL/model/variant/benchmark levels;
            # scanning --results-root therefore implies recursive discovery.
            recursive = args.recursive or bool(args.results_root)
            # A benchmark directory can retain an archived generation next to
            # the current run (for example HMMT25's historical t0.6 file).
            # When grading one directory, its manifest is the unambiguous
            # pointer to the current output.  Explicit recursive/root scans
            # still include every JSONL so archive audits remain possible.
            if not recursive:
                manifest_path = directory / "generation_manifest.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_output = manifest.get("output_file")
                except (OSError, json.JSONDecodeError):
                    manifest_output = None
                if manifest_output:
                    manifest_file = Path(manifest_output).expanduser()
                    if not manifest_file.is_absolute():
                        manifest_file = manifest_path.parent / manifest_file
                    if manifest_file.is_file() and manifest_file.suffix == ".jsonl":
                        files.add(manifest_file)
                        continue
            iterator = directory.rglob("*.jsonl") if recursive else directory.glob("*.jsonl")
            files.update(path for path in iterator if ".grading_details." not in path.name)
    return sorted(files)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grade SAR JSONL benchmark outputs")
    parser.add_argument(
        "--eval-dir",
        "--eval_dir",
        dest="eval_dirs",
        action="append",
        default=[],
        help="Benchmark directory or JSONL file; repeatable",
    )
    parser.add_argument("--results-root", help="Recursively scan a results root")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--name", help="Optional experiment label in output metadata")
    parser.add_argument("--output", help="Summary JSON path; defaults beside the input")

    parser.add_argument("--model", default=DEFAULT_VERIFIER, help="CompassVerifier model path")
    parser.add_argument("--tp", type=int, default=1, help="Verifier tensor parallel size")
    parser.add_argument("--gpus", help="Optional CUDA_VISIBLE_DEVICES for the verifier")
    parser.add_argument(
        "--allow-other-gpus",
        action="store_true",
        help="Allow ids outside the experiment's 1,2,5,6,7,8 allow-list",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument(
        "--verifier-max-tokens",
        "--max-tokens",
        "--max_tokens",
        dest="verifier_max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--verifier-temperature",
        "--temperature",
        dest="verifier_temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-input-tokens", type=int, default=32700)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-verifier", "--no-verifier", action="store_true")
    parser.add_argument(
        "--evaluator",
        choices=["auto", "math", "generic"],
        default="auto",
        help="auto uses standard options for MMMU and bundled official evaluators for LiveCodeBench/BFCL",
    )
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.tp <= 0 or args.verifier_max_tokens <= 0 or args.max_input_tokens <= 0:
        parser.error("tp, verifier-max-tokens and max-input-tokens must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.gpus and not args.allow_other_gpus:
        try:
            gpu_ids = {int(value.strip()) for value in args.gpus.split(",") if value.strip()}
        except ValueError:
            parser.error("--gpus must be a comma-separated list of integers")
        bad = sorted(gpu_ids - ALLOWED_GPU_IDS)
        if bad:
            parser.error(
                f"GPU ids {bad} are disabled for this experiment; allowed ids are "
                f"{sorted(ALLOWED_GPU_IDS)} (use --allow-other-gpus only when intentional)"
            )
    if not args.disable_verifier and args.gpus:
        selected_gpus = _select_verifier_gpus(args.gpus, args.tp)
        if selected_gpus != args.gpus:
            print(
                f"[grade] requested verifier GPU(s) {args.gpus} are busy; "
                f"using idle authorized GPU(s) {selected_gpus}"
            )
            args.gpus = selected_gpus
    try:
        files = _discover_files(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not files:
        parser.error("No JSONL files found")

    runner = None if args.disable_verifier else VerifierRunner(args)
    summaries = []
    for file_path in files:
        summary = grade_file(
            file_path,
            verbose=args.verbose,
            verifier_runner=runner,
            disable_verifier=args.disable_verifier,
            max_input_tokens=args.max_input_tokens,
            evaluator=args.evaluator,
            write_details=args.write_details,
        )
        if summary is not None:
            summaries.append(summary)

    if args.output:
        output_path = Path(args.output).expanduser()
    elif len(args.eval_dirs) == 1 and Path(args.eval_dirs[0]).expanduser().is_dir():
        output_path = Path(args.eval_dirs[0]).expanduser() / "grading_results.json"
    elif len(files) == 1:
        output_path = files[0].parent / "grading_results.json"
    else:
        common = Path(args.results_root).expanduser() if args.results_root else Path.cwd()
        output_path = common / "grading_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "verifier": {
            "enabled": not args.disable_verifier,
            "model": args.model,
            "tp": args.tp,
            "temperature": args.verifier_temperature,
            "max_tokens": args.verifier_max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpus": args.gpus,
        },
        "files": summaries,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Done] Grading summary saved to {output_path}")


if __name__ == "__main__":
    main()
