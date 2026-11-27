"""Dynamic queue for Qwen3-Omni MMAU, MMAR and MMSU audio evaluations.

The controller does not stop existing processes.  It watches physical free
VRAM, starts TP=2 vLLM jobs when a genuinely free pair is available, and
falls back to TP=3 when three cards each have enough shared free VRAM for a
30B checkpoint.  It lowers only the runtime concurrency knobs when a job is
sharing cards with an external process.  Decoding parameters and max token
limits remain fixed.  A one-item smoke run is completed before each formal
job, and all formal trajectories are written by the append-only audio runner.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GEN_PYTHON = Path(
    os.environ.get(
        "OMNI_AUDIO_GEN_PYTHON",
        "/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python",
    )
)
GRADE_PYTHON = Path(
    os.environ.get(
        "OMNI_AUDIO_GRADE_PYTHON", "/home/hongbo/.conda/envs/ar/bin/python"
    )
)
GEN = ROOT / "eval/evals/gen_omni_audio_vllm.py"
GRADE = ROOT / "eval/evals/grade_omni_audio.py"
DATA_ROOT = Path("/data/hongbo/audio_benchmarks/prepared")
OUT_ROOT = ROOT / "results/OMNI/Qwen3-Omni-30B-A3B"
SMOKE_ROOT = ROOT / "results/OMNI/_audio_smoke"
LOG_ROOT = ROOT / "results/_queue_logs"

MODELS = {
    "instruct": Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Instruct"),
    "thinking": Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Thinking"),
    "sar_ratio_1": Path(
        "/data/hongbo/sar/Qwen3-Omni-30B-A3B-Instruct-subspace-swap-ratio1-thinking"
    ),
}
BENCHMARKS = ("MMAU", "MMAR", "MMSU")
SEEDS = (101, 202, 303, 404)
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7, 8, 9)

POLL_SECONDS = max(5, int(os.environ.get("OMNI_AUDIO_QUEUE_POLL_SECONDS", "20")))
MAX_CONCURRENT = max(1, int(os.environ.get("OMNI_AUDIO_MAX_CONCURRENT", "4")))
# The Omni checkpoint needs roughly 28 GiB per card before vLLM's KV/audio
# allocations.  A card with only 25--30 GiB free is therefore not a viable
# pair even though it may look available in nvidia-smi; wait for a genuinely
# free TP pair instead of repeatedly triggering an OOM beside another job.
MIN_PAIR_FREE_MIB = int(os.environ.get("OMNI_AUDIO_MIN_PAIR_FREE_MIB", "40000"))
MIN_TRIO_FREE_MIB = int(os.environ.get("OMNI_AUDIO_MIN_TRIO_FREE_MIB", "25000"))
RESERVE_MIB = int(os.environ.get("OMNI_AUDIO_VLLM_RESERVE_MIB", "512"))
MAX_GPU_UTILIZATION = 0.95
MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Task:
    variant: str
    benchmark: str

    @property
    def model(self) -> Path:
        return MODELS[self.variant]

    @property
    def data(self) -> Path:
        return DATA_ROOT / self.benchmark / "samples.jsonl"

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / self.variant / self.benchmark

    @property
    def smoke_dir(self) -> Path:
        return SMOKE_ROOT / self.variant / self.benchmark

    @property
    def label(self) -> str:
        return f"{self.variant}_{self.benchmark}"


@dataclass
class Running:
    process: subprocess.Popen[Any]
    task: Task
    cards: tuple[int, ...]
    stage: str
    attempt: int
    log_path: Path


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (LOG_ROOT / "omni_audio_queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_state() -> dict[int, tuple[int, int, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    state: dict[int, tuple[int, int, int]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            gpu, total, free, utilization = map(int, fields)
        except ValueError:
            continue
        state[gpu] = (total, free, utilization)
    return state


def snapshot() -> str:
    return "; ".join(
        f"gpu{gpu}:free={free}MiB,util={util}%"
        for gpu, (_, free, util) in sorted(gpu_state().items())
        if gpu in ALLOWED_GPUS
    )


def external_generation_labels() -> set[str]:
    """Find audio generation jobs launched by an earlier queue controller.

    The queue controller may be restarted to change its concurrency without
    interrupting the vLLM children.  Keep their task labels reserved until
    the child exits, so a replacement controller never writes the same JSONL
    concurrently.
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "args="],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    labels: set[str] = set()
    for command in result.stdout.splitlines():
        if "gen_omni_audio_vllm.py" not in command:
            continue
        variant = re.search(r"--variant\s+(\S+)", command)
        benchmark = re.search(r"--benchmark\s+(\S+)", command)
        if variant and benchmark:
            labels.add(f"{variant.group(1)}_{benchmark.group(1)}")
    return labels


def choose_pair(
    reserved: set[int],
) -> tuple[tuple[int, int], float, tuple[int, int, int, int, int]] | None:
    state = gpu_state()
    candidates: list[tuple[int, int, tuple[int, int], float, tuple[int, int, int, int, int]]] = []
    for index, first in enumerate(ALLOWED_GPUS):
        if first in reserved or first not in state:
            continue
        total_a, free_a, util_a = state[first]
        for second in ALLOWED_GPUS[index + 1 :]:
            if second in reserved or second not in state:
                continue
            total_b, free_b, util_b = state[second]
            minimum_free = min(free_a, free_b)
            if minimum_free < MIN_PAIR_FREE_MIB:
                continue
            dynamic = min(
                (free_a - RESERVE_MIB) / total_a,
                (free_b - RESERVE_MIB) / total_b,
                MAX_GPU_UTILIZATION,
            )
            if dynamic <= 0:
                continue
            candidates.append(
                (
                    minimum_free,
                    free_a + free_b,
                    (first, second),
                    dynamic,
                    (free_a, free_b, util_a, util_b, minimum_free),
                )
            )
    if not candidates:
        return None
    _, _, cards, dynamic, details = max(candidates)
    return cards, dynamic, details


def choose_trio(
    reserved: set[int],
) -> tuple[tuple[int, ...], float, tuple[int, ...]] | None:
    """Choose three shared-memory cards for a TP=3 30B fallback."""
    state = gpu_state()
    available = [
        gpu
        for gpu in ALLOWED_GPUS
        if gpu not in reserved
        and gpu in state
        and state[gpu][1] >= MIN_TRIO_FREE_MIB
    ]
    candidates: list[tuple[int, int, tuple[int, ...], float, tuple[int, ...]]] = []
    for first_index, first in enumerate(available):
        for second_index in range(first_index + 1, len(available)):
            second = available[second_index]
            for third in available[second_index + 1 :]:
                cards = (first, second, third)
                totals = tuple(state[gpu][0] for gpu in cards)
                frees = tuple(state[gpu][1] for gpu in cards)
                utils = tuple(state[gpu][2] for gpu in cards)
                dynamic = min(
                    [
                        (free - RESERVE_MIB) / total
                        for free, total in zip(frees, totals)
                    ]
                    + [MAX_GPU_UTILIZATION]
                )
                if dynamic <= 0:
                    continue
                candidates.append(
                    (min(frees), sum(frees), cards, dynamic, (*frees, *utils))
                )
    if not candidates:
        return None
    _, _, cards, dynamic, details = max(candidates)
    return cards, dynamic, details


def _output_file(task: Task, directory: Path | None = None) -> Path:
    directory = directory or task.output_dir
    return directory / (
        f"{task.benchmark.lower()}_t0.7_p0.95_k20_rp1.0_pp1.5_n4-MNT32768.jsonl"
    )


def _count_samples(task: Task) -> int:
    with task.data.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _first_sample_id(task: Task) -> str:
    with task.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return str(json.loads(line)["example_id"])
    raise ValueError(f"empty audio dataset: {task.data}")


def _generation_complete(task: Task) -> bool:
    path = _output_file(task)
    if not path.is_file():
        return False
    expected = _count_samples(task) * len(SEEDS)
    actual = 0
    keys: set[tuple[str, int]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["example_id"]), int(row["seed"]))
                keys.add(key)
                actual += 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return actual == expected and len(keys) == expected


def _smoke_complete(task: Task) -> bool:
    path = _output_file(task, task.smoke_dir)
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    expected = {(_first_sample_id(task), seed) for seed in SEEDS}
    actual = {(str(row.get("example_id")), int(row.get("seed"))) for row in rows}
    return len(rows) == 4 and actual == expected


def _grade_complete(task: Task) -> bool:
    path = task.output_dir / "grading_results.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        expected = _count_samples(task) * len(SEEDS)
        return bool(files) and files[0].get("total_rollouts") == expected
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _runtime_knobs(dynamic: float) -> tuple[int, int]:
    # Audio encoder memory is in addition to the language KV cache.  Keep the
    # same decoding/max-token configuration, but reduce concurrent requests
    # when a pair is co-resident with another user's process.
    if dynamic < 0.75:
        return 1, 1
    if dynamic < 0.88:
        return 2, 2
    return 8, 8


def start_generation(task: Task, cards: tuple[int, ...], dynamic: float, stage: str, attempt: int) -> Running:
    directory = task.smoke_dir if stage == "smoke" else task.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / (
        f"omni_audio_{stage}_{task.variant}_{task.benchmark}_"
        f"{'_'.join(str(card) for card in cards)}.log"
    )
    max_num_seqs, chunk_size = _runtime_knobs(dynamic)
    pipeline_parallel_size = 3 if len(cards) == 3 else 1
    command = [
        str(GEN_PYTHON), str(GEN),
        "--model", str(task.model),
        "--output-dir", str(directory),
        "--variant", task.variant,
        "--benchmark", task.benchmark,
        "--data", str(task.data),
        "--gpus", ",".join(str(card) for card in cards),
        "--pipeline-parallel-size", str(pipeline_parallel_size),
        "--seeds", ",".join(str(seed) for seed in SEEDS),
        "--n", str(len(SEEDS)),
        "--temperature", "0.7",
        "--top-p", "0.95",
        "--top-k", "20",
        "--repetition-penalty", "1.0",
        "--presence-penalty", "1.5",
        "--max-tokens", "32768",
        "--max-model-len", "40960",
        "--gpu-memory-utilization", f"{dynamic:.3f}",
        "--max-num-seqs", str(max_num_seqs),
        "--request-chunk-size", str(chunk_size),
    ]
    if stage == "smoke":
        command.extend(["--limit", "1"])
    handle = path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(
        f"VLLM_START stage={stage} task={task.label} pid={process.pid} "
        f"GPUs={cards} TP={len(cards) // pipeline_parallel_size} "
        f"PP={pipeline_parallel_size} gpu_memory_utilization={dynamic:.3f} "
        f"max_num_seqs={max_num_seqs} chunk={chunk_size} attempt={attempt} "
        "full_trajectory=true"
    )
    return Running(process, task, cards, stage, attempt, path)


def start_grade(task: Task) -> Running:
    path = LOG_ROOT / f"omni_audio_grade_{task.variant}_{task.benchmark}.log"
    command = [
        str(GRADE_PYTHON), str(GRADE),
        "--input", str(_output_file(task)),
        "--output", str(task.output_dir / "grading_results.json"),
        "--write-details",
    ]
    handle = path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(f"GRADE_START task={task.label} pid={process.pid} cpu_only=true")
    return Running(process, task, (), "grade", 1, path)


def tasks() -> list[Task]:
    # Grouping by benchmark gives each audio benchmark a prompt/data path in
    # the queue, while the dynamic scheduler can interleave them as slots
    # appear.  There is deliberately no phase barrier between benchmarks.
    return [Task(variant, benchmark) for benchmark in BENCHMARKS for variant in MODELS]


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    all_tasks = tasks()
    missing_data = [task for task in all_tasks if not task.data.is_file()]
    if missing_data:
        for task in missing_data:
            log(f"MISSING_DATA task={task.label} path={task.data}")
        return 1
    log(
        "QUEUE_START audio=MMAU+MMAR+MMSU models=instruct,thinking,sar_ratio_1 "
        "seeds=101,202,303,404 backend=vllm full_trajectory=true"
    )
    pending: list[Task] = []
    smoked: set[str] = set()
    external_active = external_generation_labels()
    for task in all_tasks:
        if _smoke_complete(task):
            smoked.add(task.label)
        if _generation_complete(task) and _grade_complete(task):
            log(f"SKIP_COMPLETE task={task.label}")
            continue
        pending.append(task)
    log(
        "PENDING "
        + (",".join(task.label for task in pending) if pending else "none")
    )
    if external_active:
        log("EXTERNAL_ACTIVE " + ",".join(sorted(external_active)))

    active: list[Running] = []
    attempts: dict[str, int] = {task.label: 0 for task in pending}
    while pending or active:
        changed = False
        for run in list(active):
            return_code = run.process.poll()
            if return_code is None:
                continue
            active.remove(run)
            changed = True
            if run.stage == "smoke":
                if return_code == 0 and _smoke_complete(run.task):
                    smoked.add(run.task.label)
                    log(f"SMOKE_DONE task={run.task.label} pid={run.process.pid}")
                    pending.insert(0, run.task)
                else:
                    attempts[run.task.label] += 1
                    log(
                        f"SMOKE_FAILED task={run.task.label} pid={run.process.pid} "
                        f"rc={return_code} attempt={attempts[run.task.label]}"
                    )
                    if attempts[run.task.label] < MAX_ATTEMPTS:
                        pending.insert(0, run.task)
                    else:
                        log(f"SMOKE_GAVE_UP task={run.task.label}")
            elif run.stage == "generation":
                if return_code == 0 and _generation_complete(run.task):
                    log(f"VLLM_DONE task={run.task.label} pid={run.process.pid}")
                    active.append(start_grade(run.task))
                else:
                    attempts[run.task.label] += 1
                    log(
                        f"VLLM_FAILED task={run.task.label} pid={run.process.pid} "
                        f"rc={return_code} attempt={attempts[run.task.label]}"
                    )
                    if attempts[run.task.label] < MAX_ATTEMPTS:
                        pending.insert(0, run.task)
                    else:
                        log(f"VLLM_GAVE_UP task={run.task.label}")
            else:
                if return_code == 0 and _grade_complete(run.task):
                    log(f"GRADE_DONE task={run.task.label} pid={run.process.pid}")
                else:
                    log(f"GRADE_FAILED task={run.task.label} pid={run.process.pid} rc={return_code}")
                    if _generation_complete(run.task) and not _grade_complete(run.task):
                        pending.insert(0, run.task)

        reserved = {card for run in active for card in run.cards}
        while pending and len([run for run in active if run.stage != "grade"]) < MAX_CONCURRENT:
            # Smoke jobs can outlive a controller restart.  Reconcile their
            # append-only output before choosing the next stage, otherwise a
            # replacement controller would unnecessarily launch the same
            # one-item smoke a second time.
            for candidate in pending:
                if candidate.label not in smoked and _smoke_complete(candidate):
                    smoked.add(candidate.label)
                    log(f"SMOKE_RECOGNIZED task={candidate.label}")
            # A controller restart must not duplicate a generation still
            # owned by the previous controller.  Scan past those tasks; once
            # their process exits, they become eligible for grading/retry.
            external_active = external_generation_labels()
            candidate_index = next(
                (index for index, candidate in enumerate(pending)
                 if candidate.label not in external_active),
                None,
            )
            if candidate_index is None:
                break
            task = pending[candidate_index]
            if _generation_complete(task):
                pending.pop(candidate_index)
                if _grade_complete(task):
                    log(f"SKIP_COMPLETE task={task.label}")
                    continue
                active.append(start_grade(task))
                continue
            selection = choose_pair(reserved)
            selection_mode = "TP2"
            if selection is None:
                selection = choose_trio(reserved)
                selection_mode = "TP3"
            if selection is None:
                break
            cards, dynamic, details = selection
            pending.pop(candidate_index)
            attempts[task.label] += 1
            stage = "generation" if task.label in smoked else "smoke"
            active.append(start_generation(task, cards, dynamic, stage, attempts[task.label]))
            reserved.update(cards)
            log(
                f"SLOT_SELECTED mode={selection_mode} stage={stage} task={task.label} "
                f"cards={cards} details={details}"
            )
        if pending and not changed:
            log(
                f"WAIT_AUDIO_SLOT pending={','.join(task.label for task in pending)} "
                f"snapshot={snapshot()}"
            )
        if pending or active:
            time.sleep(POLL_SECONDS)
    log("QUEUE_FINISHED audio=MMAU+MMAR+MMSU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
