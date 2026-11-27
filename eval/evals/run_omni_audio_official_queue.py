"""Run the reset Qwen3-Omni audio evaluation with the official settings.

The queue is benchmark-major: all three model variants finish MMAR before
MMSU starts, and all MMSU variants finish before MMAU_testmini starts.  Each
task is one vLLM process covering all four seeds and writes append-only,
complete trajectories.  To use eight cards without an invalid TP degree, the
thinking variant uses TP=4 and the other two variants use TP=2.  Every
process still matches tensor_parallel_size=torch.cuda.device_count() in the
official example.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/home/hongbo/sar-reproduction")
GEN_PYTHON = Path(
    os.environ.get(
        "OMNI_AUDIO_GEN_PYTHON",
        "/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python",
    )
)
GRADE_PYTHON = Path(
    os.environ.get(
        "OMNI_AUDIO_GRADE_PYTHON",
        "/home/hongbo/.conda/envs/ar/bin/python",
    )
)
GEN = ROOT / "eval/evals/gen_omni_audio_vllm.py"
GRADE = ROOT / "eval/evals/grade_omni_audio.py"
DATA_ROOT = Path(
    os.environ.get(
        "OMNI_AUDIO_DATA_ROOT",
        "/mnt/disk3/hongbo/audio_benchmarks/prepared",
    )
)
OUT_ROOT = ROOT / "results/OMNI/Qwen3-Omni-30B-A3B"
LOG_ROOT = ROOT / "results/_queue_logs"
MODEL_ROOT = Path(
    os.environ.get("OMNI_AUDIO_MODEL_ROOT", "/mnt/disk3/hongbo/sar")
)

MODELS = {
    "instruct": MODEL_ROOT / "Qwen3-Omni-30B-A3B-Instruct",
    "thinking": MODEL_ROOT / "Qwen3-Omni-30B-A3B-Thinking",
    "sar_ratio_1": MODEL_ROOT
    / "Qwen3-Omni-30B-A3B-Instruct-subspace-swap-ratio1-thinking",
}

# MMAU_testmini is the experiment label.  Its prepared 1,000-example input
# is the official gamma-lab-umd/MMAU-test-mini test_mini split.
DATASETS = {
    "MMAR": DATA_ROOT / "MMAR/samples.jsonl",
    "MMSU": DATA_ROOT / "MMSU/samples.jsonl",
    "MMAU_testmini": DATA_ROOT / "MMAU/samples.jsonl",
}
BENCHMARKS = tuple(
    item.strip()
    for item in os.environ.get(
        "OMNI_AUDIO_BENCHMARKS", "MMAR,MMSU,MMAU_testmini"
    ).split(",")
    if item.strip()
)
SEEDS = (101, 202, 303, 404)

# GPU 4 was the card previously reported as ERROR.  The other cards are
# eligible.  The thinking task uses four visible GPUs (32 attention heads are
# divisible by four); instruct and SAR use two each.  Thus eight cards are
# occupied while every process follows the official visible-GPU rule.
GPU_POOL = tuple(
    int(item.strip())
    for item in os.environ.get("OMNI_AUDIO_GPU_POOL", "0,1,2,3,5,6,7,8,9").split(",")
    if item.strip()
)
TP4_VARIANT = os.environ.get("OMNI_AUDIO_TP4_VARIANT", "thinking")
GPU_COUNT_BY_VARIANT = {
    variant: (4 if variant == TP4_VARIANT else 2) for variant in MODELS
}
GPU_MEMORY_UTILIZATION_BY_VARIANT = {
    "instruct": float(os.environ.get("OMNI_AUDIO_INSTRUCT_GPU_UTILIZATION", "0.95")),
    "thinking": float(os.environ.get("OMNI_AUDIO_THINKING_GPU_UTILIZATION", "0.55")),
    "sar_ratio_1": float(os.environ.get("OMNI_AUDIO_SAR_GPU_UTILIZATION", "0.95")),
}
if any(
    not 0.0 < value <= 1.0
    for value in GPU_MEMORY_UTILIZATION_BY_VARIANT.values()
):
    raise ValueError(
        "OMNI_AUDIO_*_GPU_UTILIZATION values must be in the interval (0, 1]"
    )
MAX_CONCURRENT = len(MODELS)
POLL_SECONDS = max(5, int(os.environ.get("OMNI_AUDIO_QUEUE_POLL_SECONDS", "20")))
MIN_FREE_MIB = int(os.environ.get("OMNI_AUDIO_OFFICIAL_MIN_FREE_MIB", "30000"))
MAX_GPU_UTILIZATION = int(
    os.environ.get("OMNI_AUDIO_MAX_GPU_UTILIZATION", "20")
)
if not 0 <= MAX_GPU_UTILIZATION <= 100:
    raise ValueError("OMNI_AUDIO_MAX_GPU_UTILIZATION must be in [0, 100]")
MAX_ATTEMPTS = 3
IGNORED_GENERATION_PIDS = {
    int(item.strip())
    for item in os.environ.get("OMNI_AUDIO_IGNORE_GENERATION_PIDS", "").split(",")
    if item.strip()
}
REQUIRE_EXCLUSIVE_GPU = (
    os.environ.get("OMNI_AUDIO_REQUIRE_EXCLUSIVE_GPU", "0") == "1"
)


@dataclass(frozen=True)
class Task:
    variant: str
    benchmark: str

    @property
    def model(self) -> Path:
        return MODELS[self.variant]

    @property
    def data(self) -> Path:
        return DATASETS[self.benchmark]

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / self.variant / self.benchmark

    @property
    def output_file(self) -> Path:
        return self.output_dir / (
            f"{self.benchmark.lower()}_t0.6_p0.95_k20_rp1.0_pp0.0_n4-MNT16384.jsonl"
        )

    @property
    def label(self) -> str:
        return f"{self.variant}_{self.benchmark}"


@dataclass
class Running:
    process: Any
    task: Task
    cards: tuple[int, ...]
    attempt: int
    log_path: Path
    started_at: float


class AdoptedProcess:
    """Small Popen-compatible handle for a generator started by an old queue."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        except PermissionError:
            return None
        return None


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (LOG_ROOT / "omni_audio_official_queue.log").open(
        "a", encoding="utf-8"
    ) as handle:
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


def state_summary() -> str:
    state = gpu_state()
    return "; ".join(
        f"gpu{gpu}:free={state[gpu][1]}MiB,util={state[gpu][2]}%"
        for gpu in GPU_POOL
        if gpu in state
    )


def compute_busy_gpus() -> dict[int, set[int]] | None:
    """Return GPU -> CUDA compute PIDs, or None if the query failed."""
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        app_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    uuid_to_gpu: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            uuid_to_gpu[fields[1]] = int(fields[0])
        except ValueError:
            continue

    busy: dict[int, set[int]] = {}
    for line in app_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        gpu = uuid_to_gpu.get(fields[0])
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if gpu is not None:
            busy.setdefault(gpu, set()).add(pid)
    return busy


def active_generation_tasks() -> dict[str, tuple[int, tuple[int, ...]]]:
    """Return live generator main processes so a replacement queue can adopt them."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    active: dict[str, tuple[int, tuple[int, ...]]] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)$", line)
        if not match:
            continue
        pid, command = int(match.group(1)), match.group(2)
        if (
            pid == os.getpid()
            or pid in IGNORED_GENERATION_PIDS
            or "gen_omni_audio_vllm.py" not in command
        ):
            continue
        variant = re.search(r"--variant\s+(\S+)", command)
        benchmark = re.search(r"--benchmark\s+(\S+)", command)
        gpu_match = re.search(r"--gpus\s+(\S+)", command)
        if not variant or not benchmark or not gpu_match:
            continue
        try:
            cards = tuple(
                int(item) for item in gpu_match.group(1).split(",") if item
            )
        except ValueError:
            continue
        label = f"{variant.group(1)}_{benchmark.group(1)}"
        previous = active.get(label)
        # If an old launcher left a child behind, prefer the newest generator
        # process for the same task.  This prevents adopting a stale blocked
        # process after a successful resume process exits.
        if previous is None or pid > previous[0]:
            active[label] = (pid, cards)
    return active


def choose_group(reserved: set[int], count: int) -> tuple[int, ...] | None:
    state = gpu_state()
    busy = compute_busy_gpus() if REQUIRE_EXCLUSIVE_GPU else {}
    if REQUIRE_EXCLUSIVE_GPU and busy is None:
        return None
    available = [
        gpu
        for gpu in GPU_POOL
        if gpu not in reserved
        and gpu in state
        and state[gpu][1] >= MIN_FREE_MIB
        and state[gpu][2] <= MAX_GPU_UTILIZATION
        and (not REQUIRE_EXCLUSIVE_GPU or not busy.get(gpu))
    ]
    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for group in itertools.combinations(available, count):
        free = [state[gpu][1] for gpu in group]
        candidates.append((min(free), sum(free), tuple(group)))
    if not candidates:
        return None
    _, _, group = max(candidates)
    return group


def _read_rows(path: Path) -> tuple[int, set[tuple[str, int]]]:
    if not path.is_file():
        return 0, set()
    count = 0
    keys: set[tuple[str, int]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["example_id"]), int(row["seed"]))
                keys.add(key)
                count += 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0, set()
    return count, keys


def expected_keys(task: Task) -> set[tuple[str, int]]:
    sample_ids: list[str] = []
    with task.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample_ids.append(str(json.loads(line)["example_id"]))
    return {(example_id, seed) for example_id in sample_ids for seed in SEEDS}


def generation_complete(task: Task) -> bool:
    expected = expected_keys(task)
    _, keys = _read_rows(task.output_file)
    return len(keys) == len(expected) and keys == expected


def grade_complete(task: Task) -> bool:
    path = task.output_dir / "grading_results.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = len(expected_keys(task))
        files = payload.get("files", [])
        return bool(files) and files[0].get("total_rollouts") == expected
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def progress(task: Task) -> str:
    count, keys = _read_rows(task.output_file)
    expected = len(expected_keys(task))
    return f"{task.label}={count}/{expected} rows ({len(keys)} unique)"


def run_grade(task: Task, log_path: Path) -> int:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(GRADE_PYTHON),
        str(GRADE),
        "--input",
        str(task.output_file),
        "--output",
        str(task.output_dir / "grading_results.json"),
        "--write-details",
    ]
    log(f"grading {task.label}: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(result.returncode)


def start_generation(task: Task, cards: tuple[int, ...], attempt: int) -> Running:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / (
        f"omni_audio_official_{task.variant}_{task.benchmark}_"
        f"{'_'.join(str(card) for card in cards)}_attempt{attempt}.log"
    )
    command = [
        str(GEN_PYTHON),
        str(GEN),
        "--model",
        str(task.model),
        "--output-dir",
        str(task.output_dir),
        "--benchmark",
        task.benchmark,
        "--variant",
        task.variant,
        "--data",
        str(task.data),
        "--gpus",
        ",".join(str(card) for card in cards),
        "--pipeline-parallel-size",
        "1",
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION_BY_VARIANT[task.variant]),
        "--official-settings",
    ]
    log(
        f"launch {task.label} attempt={attempt} cards={cards} "
        "official_gpu_memory_utilization="
        f"{GPU_MEMORY_UTILIZATION_BY_VARIANT[task.variant]} "
        f"command={' '.join(command)}"
    )
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    handle.close()
    return Running(
        process=process,
        task=task,
        cards=cards,
        attempt=attempt,
        log_path=log_path,
        started_at=time.time(),
    )


def run_phase(benchmark: str) -> None:
    tasks = [Task(variant=variant, benchmark=benchmark) for variant in MODELS]
    pending = [task for task in tasks if not grade_complete(task)]
    done = [task for task in tasks if task not in pending]
    if done:
        log(
            f"phase={benchmark} already complete: "
            + ", ".join(task.label for task in done)
        )
    running: dict[str, Running] = {}
    attempts: dict[str, int] = {task.label: 0 for task in tasks}

    # A controller can disappear while its vLLM children survive (for example
    # when a launcher session is recycled).  Adopt those children instead of
    # starting duplicate engines against the same output file.
    active = active_generation_tasks()
    for task in list(pending):
        adopted = active.get(task.label)
        if adopted is None:
            continue
        pending.remove(task)
        pid, cards = adopted
        attempts[task.label] = 1
        running[task.label] = Running(
            process=AdoptedProcess(pid),
            task=task,
            cards=cards,
            attempt=1,
            log_path=LOG_ROOT / f"omni_audio_adopted_{task.variant}_{benchmark}.log",
            started_at=time.time(),
        )
        log(
            f"adopt {task.label} pid={pid} cards={cards}; "
            f"{progress(task)}"
        )

    while pending or running:
        reserved = {card for job in running.values() for card in job.cards}
        while pending and len(running) < MAX_CONCURRENT:
            selected: tuple[int, Task, tuple[int, ...]] | None = None
            # A larger TP task may be temporarily blocked by an external
            # process while a later same-phase TP=2 task still fits.  Launch
            # the fitting task instead of leaving otherwise safe GPUs idle;
            # the benchmark-level phase barrier remains unchanged.
            for index, candidate in enumerate(pending):
                group = choose_group(
                    reserved, GPU_COUNT_BY_VARIANT[candidate.variant]
                )
                if group is not None:
                    selected = (index, candidate, group)
                    break
            if selected is None:
                break
            index, task, group = selected
            pending.pop(index)
            attempts[task.label] += 1
            job = start_generation(task, group, attempts[task.label])
            running[task.label] = job
            reserved.update(group)

        finished: list[str] = []
        for label, job in list(running.items()):
            returncode = job.process.poll()
            if returncode is None:
                continue
            elapsed = time.time() - job.started_at
            log(
                f"exit {label} rc={returncode} elapsed={elapsed / 3600:.2f}h "
                f"{progress(job.task)} cards={job.cards}"
            )
            if returncode == 0 and generation_complete(job.task):
                grade_rc = run_grade(job.task, job.log_path)
                if grade_rc == 0 and grade_complete(job.task):
                    log(f"completed {label}: {progress(job.task)}")
                    finished.append(label)
                    continue
                log(
                    f"grading incomplete for {label} rc={grade_rc}; "
                    f"will retry (attempt {attempts[label]}/{MAX_ATTEMPTS})"
                )
            else:
                log(
                    f"generation incomplete for {label}; "
                    f"will retry (attempt {attempts[label]}/{MAX_ATTEMPTS})"
                )

            if attempts[label] >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{label} failed after {MAX_ATTEMPTS} attempts; "
                    f"inspect {job.log_path} ({progress(job.task)})"
                )
            pending.append(job.task)
            finished.append(label)

        for label in finished:
            running.pop(label, None)

        if pending or running:
            log(
                f"phase={benchmark} waiting; "
                + ", ".join(progress(task) for task in tasks)
                + f"; {state_summary()}"
            )
            time.sleep(POLL_SECONDS)

    log(f"phase complete={benchmark}; all three variants graded")


def main() -> int:
    if not GEN_PYTHON.is_file():
        raise FileNotFoundError(f"generation Python not found: {GEN_PYTHON}")
    if not GEN.is_file() or not GRADE.is_file():
        raise FileNotFoundError(f"evaluation scripts missing: {GEN} / {GRADE}")
    required_gpus = sum(GPU_COUNT_BY_VARIANT.values())
    if len(GPU_POOL) < min(GPU_COUNT_BY_VARIANT.values()):
        raise ValueError("GPU pool is smaller than the minimum task GPU count")
    if len(GPU_POOL) < required_gpus:
        log(
            f"warning: {len(GPU_POOL)} GPUs available for "
            f"{len(MODELS)} tasks requiring {required_gpus}; tasks will be serialized"
        )
    log(
        f"official queue started; benchmarks={BENCHMARKS}; seeds={SEEDS}; "
        f"gpu_pool={GPU_POOL}; gpu_count_by_variant={GPU_COUNT_BY_VARIANT}; "
        f"max_concurrent={MAX_CONCURRENT}; poll={POLL_SECONDS}s; "
        f"min_free={MIN_FREE_MIB}MiB; max_gpu_utilization={MAX_GPU_UTILIZATION}%"
    )
    log(
        "official settings: VLLM_USE_V1=0, max_model_len=32768, "
        "max_tokens=16384, max_num_seqs=8, temperature=0.6, top_p=0.95, "
        "top_k=20, gpu_memory_utilization_by_variant="
        f"{GPU_MEMORY_UTILIZATION_BY_VARIANT}, "
        f"require_exclusive_gpu={REQUIRE_EXCLUSIVE_GPU}, "
        "limit_mm_per_prompt={image:0,video:0,audio:1}, "
        "process_mm_info(use_audio_in_video=True), full trajectories"
    )
    log(f"initial GPU state: {state_summary()}")
    for benchmark in BENCHMARKS:
        run_phase(benchmark)
    log("official queue finished: MMAR -> MMSU -> MMAU_testmini")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("official queue interrupted")
        raise
    except Exception as exc:
        log(f"official queue failed: {type(exc).__name__}: {exc}")
        raise
