#!/usr/bin/env python3
"""Continue the VL queue, then build and evaluate the Omni SAR ratio sweep.

The controller deliberately waits for the existing Qwen3-VL follow-up
controller instead of competing with its two live MathVista jobs.  Once that
queue has written its success marker, this controller builds five Omni
checkpoints on CPU and evaluates each checkpoint on GSM8K, MMLU, MMAR, MMSU,
and MMAU_testmini with the existing four-seed vLLM settings.

At most six physical GPUs are ever reserved by this controller.  Generation
and grading are resumable, and status lines are emitted every fifteen
minutes while resource polling continues once per minute.
"""

from __future__ import annotations

import collections
import fcntl
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOG_ROOT = RESULTS / "_queue_logs" / "qwen3_omni_sar_ratio_followup"
CONTROLLER_LOG = LOG_ROOT / "controller.log"
LOCK_PATH = RESULTS / ".qwen3_omni_sar_ratio_followup.lock"

BUILD_PYTHON = Path(
    os.environ.get("OMNI_SAR_BUILD_PYTHON", "/home/hongbo/.conda/envs/ar/bin/python")
)
GEN_TEXT_PYTHON = Path(
    os.environ.get(
        "OMNI_SAR_TEXT_PYTHON",
        "/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python",
    )
)
GEN_AUDIO_PYTHON = Path(
    os.environ.get(
        "OMNI_SAR_AUDIO_PYTHON",
        "/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python",
    )
)
GRADE_PYTHON = Path(
    os.environ.get("OMNI_SAR_GRADE_PYTHON", "/home/hongbo/.conda/envs/ar/bin/python")
)

BUILDER = ROOT / "scripts/build_qwen3_30b_a3b_sar_ratio_sweep.py"
TEXT_GEN = ROOT / "eval/evals/gen_omni_vllm.py"
AUDIO_GEN = ROOT / "eval/evals/gen_omni_audio_vllm.py"
TEXT_GRADE = ROOT / "eval/evals/grade.py"
AUDIO_GRADE = ROOT / "eval/evals/grade_omni_audio.py"
VL_CONTROLLER_MARKER = "20260916_qwen3_vl_sar_ratio_followup_controller.py"
VL_CONTROLLER_LOG = RESULTS / "_queue_logs/qwen3_vl_sar_ratio_followup/controller.log"

BASE_MODEL = Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Instruct")
THINKING_MODEL = Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Thinking")
OUTPUT_ROOT = Path("/mnt/disk2/hongbo/sar/Qwen3-Omni-30B-A3B-sar-ratio-sweep")
RESULT_ROOT = RESULTS / "OMNI/Qwen3-Omni-30B-A3B"
VERIFIER = Path(os.environ.get("OMNI_SAR_VERIFIER", "/data/hongbo/sar/CompassVerifier-3B"))
AUDIO_DATA_ROOT = Path(
    os.environ.get("OMNI_AUDIO_DATA_ROOT", "/mnt/disk3/hongbo/audio_benchmarks/prepared")
)

RATIOS = ("0.9", "0.8", "0.7", "0.6", "0.5")
SEEDS = (101, 202, 303, 404)
TEXT_BENCHMARKS = ("GSM8K", "MMLU")
AUDIO_BENCHMARKS = ("MMAR", "MMSU", "MMAU_testmini")

# This is the same six-card ownership boundary as the running VL controller.
# GPU 4 is excluded from audio groups because the established Omni audio
# queue recorded it as a known-bad card; it remains available for text groups.
GPU_POOL = tuple(
    int(item.strip())
    for item in os.environ.get("OMNI_SAR_GPU_POOL", "0,1,3,4,5,7").split(",")
    if item.strip()
)
MAX_CARDS = 6
assert len(GPU_POOL) <= MAX_CARDS
AUDIO_BAD_GPUS = frozenset(
    int(item.strip())
    for item in os.environ.get("OMNI_SAR_AUDIO_BAD_GPUS", "4").split(",")
    if item.strip()
)
# grade.py rejects GPU4 (and any other id omitted from its allow-list).  Text
# generation may legitimately use GPU4, so choose a verifier card from the
# generation group instead of blindly reusing cards[0].
GRADE_ALLOWED_GPUS = frozenset(
    int(item.strip())
    for item in os.environ.get("OMNI_SAR_GRADE_ALLOWED_GPUS", "0,1,2,3,5,6,7,8,9").split(",")
    if item.strip()
)

POLL_SECONDS = max(10, int(os.environ.get("OMNI_SAR_POLL_SECONDS", "60")))
MONITOR_SECONDS = max(60, int(os.environ.get("OMNI_SAR_MONITOR_SECONDS", "900")))
MIN_FREE_MIB = int(os.environ.get("OMNI_SAR_MIN_FREE_MIB", "30000"))
MAX_UTILIZATION = int(os.environ.get("OMNI_SAR_MAX_UTILIZATION", "20"))
MAX_ATTEMPTS = max(1, int(os.environ.get("OMNI_SAR_MAX_ATTEMPTS", "3")))
BUILD_THREADS = max(1, int(os.environ.get("OMNI_SAR_BUILD_THREADS", "32")))


@dataclass(frozen=True)
class Task:
    ratio: str
    benchmark: str

    @property
    def kind(self) -> str:
        return "text" if self.benchmark in TEXT_BENCHMARKS else "audio"

    @property
    def variant(self) -> str:
        return f"sar_ratio_{self.ratio}"

    @property
    def model(self) -> Path:
        return OUTPUT_ROOT / f"qwen3-30b-a3b-sar-ratio-{self.ratio}"

    @property
    def data(self) -> Path:
        if self.kind == "text":
            return ROOT / "eval/data/VL_GSM8K/test.jsonl" if self.benchmark == "GSM8K" else ROOT / "eval/data/VL_MMLU/test.jsonl"
        name = "MMAU" if self.benchmark == "MMAU_testmini" else self.benchmark
        return AUDIO_DATA_ROOT / name / "samples.jsonl"

    @property
    def output_dir(self) -> Path:
        return RESULT_ROOT / self.variant / self.benchmark

    @property
    def output_file(self) -> Path:
        if self.kind == "text":
            return self.output_dir / (
                f"{self.benchmark.lower()}_t0.7_p0.95_k20_rp1.0_pp1.5_"
                "n4-MNT32768.jsonl"
            )
        return self.output_dir / (
            f"{self.benchmark.lower()}_t0.6_p0.95_k20_rp1.0_pp0.0_"
            "n4-MNT16384.jsonl"
        )

    @property
    def expected_count(self) -> int:
        count = 0
        with self.data.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    @property
    def key(self) -> str:
        return f"{self.variant}/{self.benchmark}"


@dataclass
class Running:
    task: Task
    process: subprocess.Popen[Any]
    cards: tuple[int, ...]
    stage: str
    attempt: int
    log_path: Path
    started: float


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, OSError):
        return ""


def live_pids(fragment: str) -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if fragment in proc_cmdline(pid):
            result.append(pid)
    return sorted(result)


def vl_queue_succeeded() -> bool:
    try:
        text = VL_CONTROLLER_LOG.read_text(encoding="utf-8")
    except OSError:
        return False
    return "QUEUE_DONE all_10_weights_all_40_benchmarks_complete" in text


def vl_queue_failed() -> bool:
    try:
        text = VL_CONTROLLER_LOG.read_text(encoding="utf-8")
    except OSError:
        return False
    return "VL SAR follow-up ended with failures=" in text or "_FAILED_FINAL" in text


def wait_for_vl_queue() -> None:
    log("VL_GATE_WAIT keep_existing_controller_and_two_mathvista_tasks=true")
    last_report = 0.0
    while True:
        pids = live_pids(VL_CONTROLLER_MARKER)
        if not pids:
            if vl_queue_succeeded():
                log("VL_GATE_PASSED existing_40_task_queue_complete=true")
                return
            if vl_queue_failed():
                raise RuntimeError(
                    "existing VL controller exited without a success marker; inspect "
                    f"{VL_CONTROLLER_LOG}"
                )
        now = time.monotonic()
        if now - last_report >= MONITOR_SECONDS:
            last_report = now
            log(f"VL_GATE_WAIT controller_pids={pids or 'none'}")
        time.sleep(POLL_SECONDS)


def manifest_complete(directory: Path) -> bool:
    manifest_path = directory / "sar_build_manifest.json"
    index_path = directory / "model.safetensors.index.json"
    if not manifest_path.is_file() or not index_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        return (
            manifest.get("status") == "complete"
            and manifest.get("method") == "P_base_subspace + P_perp_thinking"
            and all((directory / shard).is_file() for shard in shards)
        )
    except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
        return False


def build_ratio_checkpoints() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if all(
        manifest_complete(OUTPUT_ROOT / f"qwen3-30b-a3b-sar-ratio-{ratio}")
        for ratio in RATIOS
    ):
        log("SAR_BUILD_SKIP all_five_omni_ratio_checkpoints_complete=true")
        return

    command = [
        str(BUILD_PYTHON),
        str(BUILDER),
        "--base",
        str(BASE_MODEL),
        "--thinking",
        str(THINKING_MODEL),
        "--output-root",
        str(OUTPUT_ROOT),
        "--ratios",
        ",".join(RATIOS),
        "--device",
        "cpu",
        "--threads",
        str(BUILD_THREADS),
        "--resume",
    ]
    log(
        "SAR_BUILD_START ratios="
        + ",".join(RATIOS)
        + f" output_root={OUTPUT_ROOT} device=cpu threads={BUILD_THREADS}"
    )
    log_path = LOG_ROOT / "omni_sar_ratio_build.log"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        started = time.monotonic()
        last_report = started
        while process.poll() is None:
            now = time.monotonic()
            if now - last_report >= MONITOR_SECONDS:
                last_report = now
                log(f"SAR_BUILD_RUNNING pid={process.pid} elapsed={now - started:.0f}s")
            time.sleep(POLL_SECONDS)
        return_code = process.returncode
    if return_code != 0 or not all(
        manifest_complete(OUTPUT_ROOT / f"qwen3-30b-a3b-sar-ratio-{ratio}")
        for ratio in RATIOS
    ):
        raise RuntimeError(
            f"Omni SAR ratio build failed rc={return_code}; inspect {log_path}"
        )
    log("SAR_BUILD_DONE all_five_omni_ratio_checkpoints_complete=true")


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


def state_text() -> str:
    state = gpu_state()
    return ";".join(
        f"gpu{gpu}:free={state[gpu][1]}MiB,util={state[gpu][2]}%"
        for gpu in GPU_POOL
        if gpu in state
    )


def choose_group(
    reserved: set[int], task: Task, count: int
) -> tuple[tuple[int, ...], float] | None:
    state = gpu_state()
    allowed = [
        gpu
        for gpu in GPU_POOL
        if gpu not in reserved
        and gpu in state
        and (task.kind != "audio" or gpu not in AUDIO_BAD_GPUS)
        and state[gpu][1] >= MIN_FREE_MIB
        and state[gpu][2] <= MAX_UTILIZATION
    ]
    candidates: list[tuple[int, int, tuple[int, ...], float]] = []
    for group in itertools.combinations(allowed, count):
        free = [state[gpu][1] for gpu in group]
        total = [state[gpu][0] for gpu in group]
        dynamic = min(
            min((free_value - 512) / total_value for free_value, total_value in zip(free, total)),
            0.95,
        )
        if dynamic <= 0:
            continue
        candidates.append((min(free), sum(free), tuple(group), dynamic))
    if not candidates:
        return None
    _, _, group, dynamic = max(candidates)
    return group, dynamic


def read_rows(path: Path) -> set[tuple[str, int]]:
    rows: set[tuple[str, int]] = set()
    if not path.is_file():
        return rows
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.add((str(row["example_id"]), int(row["seed"])))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    return rows


def expected_keys(task: Task) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    sample_index = 0
    with task.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                example_id = row.get("example_id", sample_index)
                for seed in SEEDS:
                    keys.add((str(example_id), seed))
                sample_index += 1
    return keys


def generation_complete(task: Task) -> bool:
    expected = expected_keys(task)
    return bool(expected) and read_rows(task.output_file) == expected


def grading_complete(task: Task) -> bool:
    path = task.output_dir / "grading_results.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files") or []
        if not files:
            return False
        summary = files[0]
        expected = len(expected_keys(task))
        return (
            int(summary.get("total_rollouts", -1)) == expected
            and int(summary.get("missing_rollouts", 0)) == 0
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def task_complete(task: Task) -> bool:
    return generation_complete(task) and grading_complete(task)


def task_progress(task: Task) -> str:
    return f"{task.key}={len(read_rows(task.output_file))}/{len(expected_keys(task))}"


def launch_generation(task: Task, cards: tuple[int, ...], dynamic: float, attempt: int) -> Running:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"omni_{task.variant}_{task.benchmark}_attempt{attempt}.log"
    if task.kind == "text":
        command = [
            str(GEN_TEXT_PYTHON),
            str(TEXT_GEN),
            "--model",
            str(task.model),
            "--output-dir",
            str(task.output_dir),
            "--variant",
            task.variant,
            "--benchmark",
            task.benchmark,
            "--data",
            str(task.data),
            "--gpus",
            ",".join(map(str, cards)),
            "--seeds",
            ",".join(map(str, SEEDS)),
            "--n",
            str(len(SEEDS)),
            "--temperature",
            "0.7",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--repetition-penalty",
            "1.0",
            "--presence-penalty",
            "1.5",
            "--max-tokens",
            "32768",
            "--max-model-len",
            "40960",
            "--gpu-memory-utilization",
            f"{dynamic:.3f}",
            "--max-num-seqs",
            "8",
            "--request-chunk-size",
            "16",
        ]
    else:
        audio_utilization = 0.55 if len(cards) >= 4 else dynamic
        command = [
            str(GEN_AUDIO_PYTHON),
            str(AUDIO_GEN),
            "--model",
            str(task.model),
            "--output-dir",
            str(task.output_dir),
            "--variant",
            task.variant,
            "--benchmark",
            task.benchmark,
            "--data",
            str(task.data),
            "--gpus",
            ",".join(map(str, cards)),
            "--pipeline-parallel-size",
            "1",
            "--seeds",
            ",".join(map(str, SEEDS)),
            "--gpu-memory-utilization",
            f"{audio_utilization:.3f}",
            "--official-settings",
        ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "VLLM_USE_V1": "0",
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards)),
            },
            start_new_session=True,
        )
    log(
        f"GEN_START key={task.key} kind={task.kind} cards={','.join(map(str, cards))} "
        f"attempt={attempt} pid={process.pid} util={dynamic:.3f} {task_progress(task)}"
    )
    return Running(task, process, cards, "gen", attempt, log_path, time.monotonic())


def launch_grade(task: Task, cards: tuple[int, ...], attempt: int, log_path: Path) -> Running:
    if task.kind == "text":
        grade_card = next((card for card in cards if card in GRADE_ALLOWED_GPUS), None)
        if grade_card is None:
            raise RuntimeError(
                f"no allowed verifier GPU in generation group {cards}; "
                f"allowed={sorted(GRADE_ALLOWED_GPUS)}"
            )
        grade_cards = (grade_card,)
        command = [
            str(GRADE_PYTHON),
            str(TEXT_GRADE),
            "--eval-dir",
            str(task.output_dir),
            "--output",
            str(task.output_dir / "grading_results.json"),
            "--model",
            str(VERIFIER),
            "--gpus",
            str(grade_cards[0]),
            "--tp",
            "1",
            "--gpu-memory-utilization",
            "0.50",
            "--evaluator",
            "auto",
            "--write-details",
            "--verbose",
        ]
    else:
        grade_cards = ()
        command = [
            str(GRADE_PYTHON),
            str(AUDIO_GRADE),
            "--input",
            str(task.output_file),
            "--output",
            str(task.output_dir / "grading_results.json"),
            "--write-details",
        ]
    grade_log = log_path.with_name(log_path.stem + "_grade.log")
    with grade_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
    log(
        f"GRADE_START key={task.key} cards={','.join(map(str, grade_cards)) or 'cpu'} "
        f"attempt={attempt} pid={process.pid}"
    )
    return Running(task, process, grade_cards, "grade", attempt, grade_log, time.monotonic())


def make_tasks() -> list[Task]:
    tasks: list[Task] = []
    for ratio in RATIOS:
        for benchmark in (*TEXT_BENCHMARKS, *AUDIO_BENCHMARKS):
            tasks.append(Task(ratio, benchmark))
    return tasks


def status_line(
    pending: collections.deque[Task],
    running: dict[str, Running],
    completed: set[str],
    failed: set[str],
) -> str:
    active = ",".join(
        f"{key}:{job.stage}:pid{job.process.pid}:gpus{','.join(map(str, job.cards))}"
        for key, job in sorted(running.items())
    ) or "none"
    return (
        f"STATUS complete={len(completed)}/{len(make_tasks())} "
        f"pending={len(pending)} active={active} failed={len(failed)} "
        f"max_cards={MAX_CARDS} state={state_text()}"
    )


def run_evaluations() -> None:
    tasks = make_tasks()
    pending: collections.deque[Task] = collections.deque(
        task for task in tasks if not task_complete(task)
    )
    completed = {task.key for task in tasks if task_complete(task)}
    log(
        f"QUEUE_START tasks={len(tasks)} already_complete={len(completed)} "
        f"pending={len(pending)} max_cards={MAX_CARDS} seeds={SEEDS} backend=vllm"
    )

    running: dict[str, Running] = {}
    attempts: dict[str, int] = collections.defaultdict(int)
    failed: set[str] = set()
    last_report = time.monotonic()

    while pending or running:
        for key, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                continue

            if job.stage == "gen" and return_code == 0 and generation_complete(job.task):
                log(
                    f"GEN_DONE key={key} pid={job.process.pid} rc=0 "
                    f"elapsed={time.monotonic() - job.started:.0f}s {task_progress(job.task)}"
                )
                running[key] = launch_grade(job.task, job.cards, job.attempt, job.log_path)
                continue

            if job.stage == "grade" and return_code == 0 and grading_complete(job.task):
                completed.add(key)
                del running[key]
                log(
                    f"TASK_DONE key={key} pid={job.process.pid} rc=0 "
                    f"complete={len(completed)}/{len(tasks)} {task_progress(job.task)}"
                )
                continue

            del running[key]
            if attempts[key] < MAX_ATTEMPTS:
                pending.appendleft(job.task)
                log(
                    f"{job.stage.upper()}_RETRY key={key} pid={job.process.pid} rc={return_code} "
                    f"attempt_next={attempts[key] + 1} {task_progress(job.task)}"
                )
            else:
                failed.add(key)
                log(
                    f"{job.stage.upper()}_FAILED_FINAL key={key} pid={job.process.pid} "
                    f"rc={return_code} log={job.log_path} {task_progress(job.task)}"
                )

        reserved = {card for job in running.values() for card in job.cards}
        while pending:
            selected: tuple[int, Task, tuple[int, ...], float] | None = None
            for index, task in enumerate(pending):
                required = 2
                # After an audio generation failure, give the next attempt a
                # four-card fallback and lower official utilization.  This is
                # still within the six-card controller boundary.
                if task.kind == "audio" and attempts[task.key] >= 1:
                    required = 4
                choice = choose_group(reserved, task, required)
                if choice is not None:
                    cards, dynamic = choice
                    selected = (index, task, cards, dynamic)
                    break
            if selected is None:
                break
            index, task, cards, dynamic = selected
            del pending[index]
            attempts[task.key] += 1
            running[task.key] = launch_generation(task, cards, dynamic, attempts[task.key])
            reserved.update(cards)

        now = time.monotonic()
        if now - last_report >= MONITOR_SECONDS:
            last_report = now
            log(status_line(pending, running, completed, failed))
        if pending or running:
            time.sleep(POLL_SECONDS)

    log(status_line(pending, running, completed, failed))
    if failed or len(completed) != len(tasks):
        raise RuntimeError(
            f"Omni SAR ratio queue ended with failures={len(failed)} "
            f"completed={len(completed)}/{len(tasks)}"
        )
    log("QUEUE_DONE all_five_omni_ratios_all_five_benchmarks_complete=true")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another Omni SAR ratio follow-up controller is already running")

        log(
            f"CONTROLLER_START gpu_pool={GPU_POOL} max_cards={MAX_CARDS} "
            f"poll={POLL_SECONDS}s monitor={MONITOR_SECONDS}s audio_bad_gpus={sorted(AUDIO_BAD_GPUS)}"
        )
        wait_for_vl_queue()
        build_ratio_checkpoints()
        run_evaluations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
