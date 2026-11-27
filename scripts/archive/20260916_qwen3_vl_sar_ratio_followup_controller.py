#!/usr/bin/env python3
"""Finish the 30B gate, build VL SAR ratios, and run the VL eval matrix.

The controller intentionally owns only the requested follow-up:

1. wait until every Qwen3-30B-A3B ratio/benchmark item is complete;
2. build five no-truncation SAR checkpoints for each of Qwen3-VL-2B and
   Qwen3-VL-4B from their Instruct/Thinking pair; and
3. run all 10 checkpoints on GSM8K, MMLU, MMMU and MathVista-testmini.

It uses exactly the six-card pool approved for the current queue, one VL
vLLM generation task per card except that MathVista uses three TP=1 data-
parallel cards, four seeds (101/202/303/404), and a one-minute resource poll
with fifteen-minute status logging.  Generation and grading are resumable and
complete results are never regenerated unnecessarily.
"""

from __future__ import annotations

import collections
import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TEXT_RESULTS = RESULTS / "TEXT" / "Qwen3-30B-A3B"
VL_RESULTS = RESULTS / "VL"
LOG_ROOT = RESULTS / "_queue_logs" / "qwen3_vl_sar_ratio_followup"
CONTROLLER_LOG = LOG_ROOT / "controller.log"
LOCK_PATH = RESULTS / ".qwen3_vl_sar_ratio_followup.lock"

PYTHON = Path(os.environ.get("VL_SAR_PYTHON", "/home/hongbo/.conda/envs/ar/bin/python"))
BUILDER = ROOT / "scripts/build_qwen3_vl_sar_ratio_sweep.py"
GEN = ROOT / "eval/evals/gen_vllm.py"
GRADE = ROOT / "eval/evals/grade.py"
GRADE_VL = ROOT / "eval/evals/grade_new_vl.py"
VERIFIER = Path(os.environ.get("VL_SAR_VERIFIER", "/data/hongbo/sar/CompassVerifier-3B"))
MATHVISTA_MEDIA_ROOT = Path(
    os.environ.get("VL_SAR_MATHVISTA_MEDIA_ROOT", "/data/hongbo/datasets/MathVista")
)

# This pool is deliberately hard-coded to six cards.  Other cards may be used
# by unrelated jobs and are outside this controller's ownership.
GPU_POOL = (0, 1, 3, 4, 5, 7)
MAX_CARDS = 6
assert len(GPU_POOL) == MAX_CARDS

SEEDS = (101, 202, 303, 404)
RATIOS = (0.9, 0.8, 0.7, 0.6, 0.5)
BENCHMARKS = ("GSM8K", "MMLU", "MMMU", "MathVista-testmini")
VL_DATA = {
    "GSM8K": ROOT / "eval/data/VL_GSM8K/test.jsonl",
    "MMLU": ROOT / "eval/data/VL_MMLU/test.jsonl",
    "MMMU": ROOT / "eval/data/VL_MMMU/test.jsonl",
    "MathVista-testmini": ROOT / "eval/data/VL_MathVista/testmini.jsonl",
}

POLL_SECONDS = max(10, int(os.environ.get("VL_SAR_POLL_SECONDS", "60")))
MONITOR_SECONDS = max(60, int(os.environ.get("VL_SAR_MONITOR_SECONDS", "900")))
MAX_ATTEMPTS = max(1, int(os.environ.get("VL_SAR_MAX_ATTEMPTS", "3")))
IDLE_MEMORY_MIB = int(os.environ.get("VL_SAR_IDLE_MEMORY_MIB", "2000"))
IDLE_UTILIZATION = int(os.environ.get("VL_SAR_IDLE_UTILIZATION", "5"))
BUILD_THREADS = max(1, int(os.environ.get("VL_SAR_BUILD_THREADS", "32")))


@dataclass(frozen=True)
class Family:
    model_name: str
    base: Path
    thinking: Path
    output_root: Path
    output_prefix: str


FAMILIES = (
    Family(
        "Qwen3-VL-2B",
        Path("/data/hongbo/sar/Qwen3-VL-2B-Instruct"),
        Path("/data/hongbo/sar/Qwen3-VL-2B-Thinking"),
        Path("/mnt/disk2/hongbo/sar/Qwen3-VL-2B-sar-ratio-sweep"),
        "Qwen3-VL-2B-sar-ratio",
    ),
    Family(
        "Qwen3-VL-4B",
        Path("/data/hongbo/sar/Qwen3-VL-4B-Instruct"),
        Path("/data/hongbo/sar/Qwen3-VL-4B-Thinking"),
        Path("/mnt/disk2/hongbo/sar/Qwen3-VL-4B-sar-ratio-sweep"),
        "Qwen3-VL-4B-sar-ratio",
    ),
)


@dataclass(frozen=True)
class Task:
    family: str
    model_name: str
    variant: str
    ratio: float
    model: Path
    benchmark: str
    output_dir: Path
    expected_examples: int
    cards_required: int

    @property
    def key(self) -> str:
        return f"{self.model_name}/{self.variant}/{self.benchmark}"


@dataclass
class Running:
    task: Task
    cards: tuple[int, ...]
    attempt: int
    stage: str
    process: subprocess.Popen[Any]
    started: float
    log_path: Path


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ratio_label(ratio: float) -> str:
    return f"{ratio:.1f}" if abs(ratio * 10 - round(ratio * 10)) < 1e-9 else f"{ratio:g}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, OSError):
        return ""


def current_30b_controllers() -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if "20260913_qwen3_sar_followup_controller.py" in proc_cmdline(pid):
            matches.append(pid)
    return sorted(matches)


def nvidia_state() -> dict[int, tuple[int, int]]:
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
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    state: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            state[int(fields[0])] = (int(fields[1]), int(fields[2]))
        except ValueError:
            continue
    return state


def gpu_idle(gpu: int, state: dict[int, tuple[int, int]] | None = None) -> bool:
    state = nvidia_state() if state is None else state
    if not state:
        # If the query tool is unavailable, let vLLM report the real failure;
        # this avoids turning a monitoring outage into a permanent wait.
        return True
    memory, utilization = state.get(gpu, (10**9, 100))
    return memory <= IDLE_MEMORY_MIB and utilization <= IDLE_UTILIZATION


def pool_idle() -> bool:
    state = nvidia_state()
    return all(gpu_idle(gpu, state) for gpu in GPU_POOL)


def task_complete_30b(ratio: str, benchmark: str) -> bool:
    directory = TEXT_RESULTS / f"sar_ratio_{ratio}" / benchmark
    try:
        manifest = read_json(directory / "generation_manifest.json")
        if not isinstance(manifest, dict):
            return False
        output = Path(str(manifest["output_file"]))
        summary_payload = read_json(directory / "grading_results.json")
        if not isinstance(summary_payload, dict):
            return False
        files = summary_payload.get("files")
        if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
            return False
        summary = files[0]
        expected = int(summary["num_problems"]) * int(
            summary.get("expected_rollouts_per_problem", len(SEEDS))
        )
        return (
            manifest.get("backend") == "vllm"
            and manifest.get("modality") == "TEXT"
            and [int(x) for x in manifest["seeds"]] == list(SEEDS)
            and output.is_file()
            and int(manifest["num_rows"]) == int(manifest["num_examples"]) * len(SEEDS)
            and int(summary["total_rollouts"]) == expected
            and int(summary.get("missing_rollouts", 0)) == 0
            and int(summary.get("extra_rollouts", 0)) == 0
            and summary.get("verifier_enabled") is True
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def all_30b_complete() -> bool:
    return all(
        task_complete_30b(ratio_label(ratio), benchmark)
        for ratio in RATIOS
        for benchmark in (
            "AIME24",
            "AIME25",
            "AIME26",
            "AMC23",
            "HMMT25",
            "MATH-500",
            "Olympiad-Bench",
            "CMIMC25",
        )
    )


def model_complete(family: Family, ratio: float) -> bool:
    output = (
        family.output_root
        / f"{family.output_prefix}-{ratio_label(ratio)}"
    )
    marker_path = output / "sar_build_manifest.json"
    index_path = output / "model.safetensors.index.json"
    config_path = output / "config.json"
    try:
        marker = read_json(marker_path)
        index = read_json(index_path)
        if not isinstance(marker, dict) or not isinstance(index, dict):
            return False
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shards = sorted(set(weight_map.values()))
        return (
            marker.get("status") == "complete"
            and float(marker["ratio"]) == ratio
            and marker.get("base") == str(family.base.resolve())
            and marker.get("thinking") == str(family.thinking.resolve())
            and config_path.is_file()
            and all((output / shard).is_file() and (output / shard).stat().st_size > 0 for shard in shards)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_family(family: Family) -> None:
    missing = [ratio for ratio in RATIOS if not model_complete(family, ratio)]
    if not missing:
        log(f"BUILD_REUSE family={family.model_name} ratios=all output={family.output_root}")
        return
    if not family.base.is_dir() or not family.thinking.is_dir():
        raise FileNotFoundError(
            f"missing VL source family={family.model_name} base={family.base} thinking={family.thinking}"
        )
    build_log = LOG_ROOT / f"build_{family.model_name}_attempt1.log"
    log(
        f"BUILD_START family={family.model_name} missing={','.join(ratio_label(x) for x in missing)} "
        f"output={family.output_root} log={build_log}"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(BUILD_THREADS),
            "MKL_NUM_THREADS": str(BUILD_THREADS),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        str(PYTHON),
        str(BUILDER),
        "--base",
        str(family.base),
        "--thinking",
        str(family.thinking),
        "--output-root",
        str(family.output_root),
        "--output-prefix",
        family.output_prefix,
        "--ratios",
        ",".join(ratio_label(x) for x in RATIOS),
        "--device",
        "cpu",
        "--threads",
        str(BUILD_THREADS),
        "--resume",
    ]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with build_log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or any(not model_complete(family, ratio) for ratio in RATIOS):
        raise RuntimeError(
            f"VL SAR build failed family={family.model_name} rc={completed.returncode}; see {build_log}"
        )
    log(f"BUILD_DONE family={family.model_name} ratios=all output={family.output_root}")


def count_examples(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def make_tasks() -> list[Task]:
    tasks: list[Task] = []
    # Interleave the two families at each ratio so the three-card MathVista
    # waves do not spend their first several rounds on only one model size.
    for ratio in RATIOS:
        for family in FAMILIES:
            model = family.output_root / f"{family.output_prefix}-{ratio_label(ratio)}"
            variant = f"sar_ratio_{ratio_label(ratio)}"
            for benchmark in BENCHMARKS:
                tasks.append(
                    Task(
                        family=family.model_name,
                        model_name=family.model_name,
                        variant=variant,
                        ratio=ratio,
                        model=model,
                        benchmark=benchmark,
                        output_dir=VL_RESULTS / family.model_name / variant / benchmark,
                        expected_examples=count_examples(VL_DATA[benchmark]),
                        cards_required=3 if benchmark == "MathVista-testmini" else 1,
                    )
                )
    return tasks


def generation_complete(task: Task) -> bool:
    try:
        manifest = read_json(task.output_dir / "generation_manifest.json")
        if not isinstance(manifest, dict):
            return False
        output = Path(str(manifest["output_file"]))
        if not output.is_absolute():
            output = task.output_dir / output
        rows = manifest.get("rows_per_example") or {}
        runner = manifest.get("runner") or {}
        return (
            manifest.get("backend") == "vllm"
            and isinstance(runner, dict)
            and runner.get("backend") == "vllm"
            and int(manifest.get("tensor_parallel_size", 0)) == 1
            and manifest.get("modality") == "VL"
            and manifest.get("model_name") == task.model_name
            and manifest.get("variant") == task.variant
            and manifest.get("benchmark") == task.benchmark
            and [int(x) for x in manifest["seeds"]] == list(SEEDS)
            and int(manifest["num_examples"]) == task.expected_examples
            and int(manifest["num_rows"]) == task.expected_examples * len(SEEDS)
            and output.is_file()
            and isinstance(rows, dict)
            and len(rows) == task.expected_examples
            and all(int(value) == len(SEEDS) for value in rows.values())
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def grading_complete(task: Task) -> bool:
    try:
        payload = read_json(task.output_dir / "grading_results.json")
        if not isinstance(payload, dict):
            return False
        if task.benchmark == "MathVista-testmini":
            pass1_by_seed = payload.get("pass1_by_seed")
            return (
                payload.get("task_name") == task.benchmark
                and int(payload["total_rollouts"])
                == task.expected_examples * len(SEEDS)
                and int(payload.get("missing_rollouts", 0)) == 0
                and int(payload.get("expected_rollouts_per_problem", len(SEEDS))) == len(SEEDS)
                and isinstance(pass1_by_seed, dict)
                and sorted(pass1_by_seed) == [str(x) for x in SEEDS]
            )
        files = payload.get("files")
        if not isinstance(files, list) or len(files) != 1:
            return False
        summary = files[0]
        manifest = summary.get("manifest", {})
        return (
            [int(x) for x in manifest.get("seeds", [])] == list(SEEDS)
            and int(summary["num_problems"]) == task.expected_examples
            and int(summary["total_rollouts"]) == task.expected_examples * len(SEEDS)
            and int(summary.get("missing_rollouts", 0)) == 0
            and int(summary.get("extra_rollouts", 0)) == 0
            and summary.get("verifier_enabled") is True
            and payload.get("verifier", {}).get("enabled") is True
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def task_complete(task: Task) -> bool:
    return generation_complete(task) and grading_complete(task)


def task_log_path(task: Task, attempt: int) -> Path:
    safe = task.key.replace("/", "_").replace("-", "_")
    return LOG_ROOT / f"{safe}_attempt{attempt}.log"


def generation_command(task: Task, cards: tuple[int, ...]) -> list[str]:
    visual = task.benchmark in {"MMMU", "MathVista-testmini"}
    card_text = ",".join(str(card) for card in cards)
    return [
        str(PYTHON),
        str(GEN),
        "--model",
        str(task.model),
        "--tasks",
        task.benchmark,
        "--data-dir",
        str(ROOT / "eval/data"),
        "--task-path",
        f"{task.benchmark}={VL_DATA[task.benchmark]}",
        "--output-dir",
        str(task.output_dir),
        "--modality",
        "VL",
        "--model-name",
        task.model_name,
        "--variant",
        task.variant,
        "--gpus",
        card_text,
        "--allow-other-gpus",
        "--tp",
        "1",
        "--seeds",
        ",".join(str(x) for x in SEEDS),
        "--n",
        "4",
        "--temperature",
        "0.7",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--repetition-penalty",
        "1.0",
        "--presence-penalty",
        "0.0" if visual else "1.5",
        "--max-tokens",
        "40960" if visual else "32768",
        "--max-model-len",
        "40960",
        "--gpu-memory-utilization",
        "0.95",
        "--dtype",
        "bfloat16",
        "--backend",
        "vllm",
        "--enforce-eager",
        "--trust-remote-code",
        "--response-mode",
        "full",
        "--overwrite",
    ] + (["--allowed-local-media-path", str(MATHVISTA_MEDIA_ROOT)] if visual else [])


def grading_command(task: Task, gpu: int) -> list[str]:
    if task.benchmark == "MathVista-testmini":
        return [str(PYTHON), str(GRADE_VL), "--eval-dir", str(task.output_dir)]
    return [
        str(PYTHON),
        str(GRADE),
        "--eval-dir",
        str(task.output_dir),
        "--model",
        str(VERIFIER),
        "--gpus",
        str(gpu),
        "--tp",
        "1",
        "--gpu-memory-utilization",
        "0.95",
        "--evaluator",
        "auto",
        "--write-details",
        "--verbose",
        "--allow-other-gpus",
    ]


def start_process(task: Task, cards: tuple[int, ...], attempt: int, stage: str) -> Running:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = task_log_path(task, attempt)
    command = generation_command(task, cards) if stage == "gen" else grading_command(task, cards[0])
    environment = os.environ.copy()
    if stage == "gen":
        # gen_vllm receives physical ids through --gpus and internally creates
        # one data-parallel worker per id.  Restricting CUDA_VISIBLE_DEVICES
        # here would reinterpret those physical ids as local ordinals.
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(cards[0])
    environment["PYTHONUNBUFFERED"] = "1"
    with path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(
        f"{stage.upper()}_START key={task.key} gpus={','.join(map(str, cards))} attempt={attempt} "
        f"pid={process.pid} log={path}"
    )
    return Running(task, cards, attempt, stage, process, time.monotonic(), path)


def launch_or_grade(task: Task, cards: tuple[int, ...], attempt: int) -> Running:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    if generation_complete(task):
        return start_process(task, cards, attempt, "grade")
    return start_process(task, cards, attempt, "gen")


def status_line(
    pending: collections.deque[Task],
    running: dict[str, Running],
    completed: set[str],
    failed: list[str],
    quarantined: set[int],
) -> str:
    active = ",".join(
        f"gpus{','.join(map(str, run.cards))}:{run.task.key}:{run.stage}:pid{run.process.pid}"
        for _, run in sorted(running.items())
    ) or "none"
    return (
        f"STATUS complete={len(completed)}/40 pending={len(pending)} active={active} "
        f"failed={len(failed)} quarantined={sorted(quarantined)} max_cards={MAX_CARDS}"
    )


def wait_for_gate() -> None:
    last_report = 0.0
    while True:
        complete = all_30b_complete()
        controllers = current_30b_controllers()
        idle = pool_idle() if complete and not controllers else False
        if complete and not controllers and idle:
            log("GATE_READY 30B=40/40 controller=stopped pool_idle=yes")
            return
        current = time.monotonic()
        if current - last_report >= MONITOR_SECONDS:
            last_report = current
            log(
                f"GATE_WAIT 30B_complete={complete} current_30b_controllers={controllers or 'none'} "
                f"pool_idle={idle}"
            )
        time.sleep(POLL_SECONDS)


def run_queue(tasks: list[Task]) -> None:
    pending: collections.deque[Task] = collections.deque()
    completed: set[str] = set()
    for task in tasks:
        if task_complete(task):
            completed.add(task.key)
        else:
            pending.append(task)
    log(f"QUEUE_START tasks={len(tasks)} already_complete={len(completed)} pending={len(pending)}")

    running: dict[str, Running] = {}
    attempts: dict[str, int] = collections.defaultdict(int)
    failed: list[str] = []
    quarantined: set[int] = set()
    last_report = time.monotonic()

    while pending or running:
        for key, run in list(running.items()):
            return_code = run.process.poll()
            if return_code is None:
                continue
            if run.stage == "gen" and return_code == 0 and generation_complete(run.task):
                log(
                    f"GEN_DONE key={run.task.key} gpus={','.join(map(str, run.cards))} pid={run.process.pid} "
                    f"elapsed={time.monotonic() - run.started:.0f}s"
                )
                # MathVista grading is a local CPU-side normalization pass;
                # retain only the first card for its companion grade process
                # so the other cards can immediately serve another task.
                running[key] = start_process(run.task, (run.cards[0],), run.attempt, "grade")
                continue

            succeeded = (
                run.stage == "grade"
                and return_code == 0
                and grading_complete(run.task)
            )
            if succeeded:
                completed.add(run.task.key)
                del running[key]
                log(
                    f"TASK_DONE key={run.task.key} gpus={','.join(map(str, run.cards))} pid={run.process.pid} "
                    f"elapsed={time.monotonic() - run.started:.0f}s complete={len(completed)}/40"
                )
                continue

            attempts[run.task.key] = max(attempts[run.task.key], run.attempt)
            if run.stage == "gen" and 4 in run.cards:
                quarantined.add(4)
                log(f"GPU_QUARANTINE gpu=4 reason=generation_failure key={run.task.key}")
            del running[key]
            if attempts[run.task.key] < MAX_ATTEMPTS:
                pending.appendleft(run.task)
                log(
                    f"{run.stage.upper()}_RETRY key={run.task.key} gpus={','.join(map(str, run.cards))} rc={return_code} "
                    f"next_attempt={attempts[run.task.key] + 1}"
                )
            else:
                failed.append(run.task.key)
                log(
                    f"{run.stage.upper()}_FAILED_FINAL key={run.task.key} gpus={','.join(map(str, run.cards))} "
                    f"rc={return_code} log={run.log_path}"
                )

        state = nvidia_state()
        reserved = {gpu for run in running.values() for gpu in run.cards}
        free_gpus = [
            gpu
            for gpu in GPU_POOL
            if gpu not in reserved and gpu not in quarantined and gpu_idle(gpu, state)
        ]

        def take_pending(cards_required: int) -> Task | None:
            for index, candidate in enumerate(pending):
                if candidate.cards_required == cards_required:
                    del pending[index]
                    return candidate
            return None

        # MathVista uses three independent TP=1 workers.  Schedule those
        # groups first so the large multimodal batches do not fall back to a
        # known-unsafe single-card configuration.  Any leftover cards then
        # run the one-card benchmarks.
        while free_gpus:
            cards_required = 3 if len(free_gpus) >= 3 else 1
            task = take_pending(cards_required)
            if task is None:
                if cards_required == 3:
                    task = take_pending(1)
                    cards_required = 1
                if task is None:
                    break
            if task.key in completed or task_complete(task):
                completed.add(task.key)
                continue
            cards = tuple(free_gpus[:cards_required])
            del free_gpus[:cards_required]
            attempt = attempts[task.key] + 1
            attempts[task.key] = attempt
            running[task.key] = launch_or_grade(task, cards, attempt)

        current = time.monotonic()
        if current - last_report >= MONITOR_SECONDS:
            last_report = current
            log(status_line(pending, running, completed, failed, quarantined))
        if pending or running:
            time.sleep(POLL_SECONDS)

    log(status_line(pending, running, completed, failed, quarantined))
    if failed or len(completed) != len(tasks):
        raise RuntimeError(
            f"VL SAR follow-up ended with failures={len(failed)} completed={len(completed)}/{len(tasks)}"
        )
    log("QUEUE_DONE all_10_weights_all_40_benchmarks_complete")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another Qwen3-VL SAR ratio follow-up controller is already running")

        log(
            f"CONTROLLER_START gpu_pool={GPU_POOL} max_cards={MAX_CARDS} poll={POLL_SECONDS}s "
            f"monitor={MONITOR_SECONDS}s seeds={SEEDS} ratios={RATIOS} backend=vllm"
        )
        wait_for_gate()
        for family in FAMILIES:
            build_family(family)
        tasks = make_tasks()
        run_queue(tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
