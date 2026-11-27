#!/usr/bin/env python3
"""Run the Qwen3-4B SAR ratio sweep with one vLLM worker per GPU.

There are five checkpoints and eight text benchmarks.  Each task generates
exactly four rollouts (seeds 101/202/303/404) and is graded before its GPU is
released.  At most six single-GPU tasks are kept in flight whenever pending
work is available.  The unavailable physical GPU 4 is excluded from the pool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA_ROOT = ROOT / "eval/data"
MODEL_ROOT = Path("/mnt/disk3/hongbo/sar-models/qwen3-4b-sar-ratio-sweep")
GEN_PYTHON = Path("/home/hongbo/.conda/envs/ar/bin/python")
GEN = ROOT / "eval/evals/gen_vllm.py"
GRADE = ROOT / "eval/evals/grade.py"
VERIFIER = Path("/data/hongbo/sar/CompassVerifier-3B")
SEEDS = (101, 202, 303, 404)
RATIOS = (0.9, 0.8, 0.7, 0.6, 0.5)
BENCHMARKS = (
    "AIME24",
    "AIME25",
    "AIME26",
    "AMC23",
    "HMMT25",
    "MATH-500",
    "Olympiad-Bench",
    "CMIMC25",
)
GPU_POOL = tuple(
    int(item.strip())
    for item in os.environ.get("SAR_SWEEP_GPU_POOL", "0,1,2,3,5,6,7,8,9").split(",")
    if item.strip()
)
MAX_CARDS = 6
MAX_CONCURRENT = min(
    MAX_CARDS,
    len(GPU_POOL),
    max(1, int(os.environ.get("SAR_SWEEP_MAX_CONCURRENT", str(MAX_CARDS)))),
)
GPU_IDLE_MEMORY_MIB = int(os.environ.get("SAR_SWEEP_GPU_IDLE_MEMORY_MIB", "2000"))
GPU_IDLE_UTILIZATION = int(os.environ.get("SAR_SWEEP_GPU_IDLE_UTILIZATION", "5"))
POLL_SECONDS = max(10, int(os.environ.get("SAR_SWEEP_POLL_SECONDS", "20")))
MAX_RETRIES = max(1, int(os.environ.get("SAR_SWEEP_MAX_RETRIES", "2")))
LOG_ROOT = RESULTS / "_queue_logs/sar_ratio_sweep"
CONTROLLER_LOG = RESULTS / "_queue_logs/sar_ratio_sweep_controller.log"


@dataclass(frozen=True)
class Task:
    ratio: float
    benchmark: str

    @property
    def label(self) -> str:
        return f"sar_ratio_{self.ratio:.1f}/{self.benchmark}"

    @property
    def variant(self) -> str:
        return f"sar_ratio_{self.ratio:.1f}"

    @property
    def model(self) -> Path:
        return MODEL_ROOT / f"qwen3-4b-sar-ratio-{self.ratio:.1f}"

    @property
    def output_dir(self) -> Path:
        return RESULTS / "TEXT/Qwen3-4B" / self.variant / self.benchmark

    @property
    def output_file(self) -> Path:
        return self.output_dir / (
            f"{self.benchmark.lower()}_t0.7_p0.95_k20_rp1.0_pp1.5_n4-MNT32768.jsonl"
        )


@dataclass
class Running:
    task: Task
    process: subprocess.Popen[Any]
    gpu: int
    attempt: int
    log_path: Path
    log_handle: Any


def log(message: str) -> None:
    CONTROLLER_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ratio_label(ratio: float) -> str:
    return f"{ratio:.1f}"


def gpu_state() -> dict[int, tuple[int, int]]:
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
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            gpu, memory, utilization = map(int, fields)
        except ValueError:
            continue
        state[gpu] = (memory, utilization)
    return state


def idle_gpus(reserved: set[int]) -> list[int]:
    state = gpu_state()
    return [
        gpu
        for gpu in GPU_POOL
        if gpu not in reserved
        and gpu in state
        and state[gpu][0] <= GPU_IDLE_MEMORY_MIB
        and state[gpu][1] <= GPU_IDLE_UTILIZATION
    ]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def generation_complete(task: Task) -> bool:
    manifest = read_json(task.output_dir / "generation_manifest.json")
    if manifest is None or not task.output_file.is_file():
        return False
    try:
        expected = int(manifest["num_examples"]) * len(SEEDS)
        if manifest.get("backend") != "vllm":
            return False
        if [int(seed) for seed in manifest["seeds"]] != list(SEEDS):
            return False
        if int(manifest.get("num_rows", -1)) != expected:
            return False
        rows = []
        with task.output_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) != expected:
            return False
        keys = {(str(row["example_id"]), int(row["seed"])) for row in rows}
        return len(keys) == expected
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def grading_complete(task: Task) -> bool:
    payload = read_json(task.output_dir / "grading_results.json")
    if payload is None:
        return False
    try:
        files = payload.get("files")
        if not isinstance(files, list) or len(files) != 1:
            return False
        summary = files[0]
        expected = int(summary["num_problems"]) * int(
            summary.get("expected_rollouts_per_problem", len(SEEDS))
        )
        return (
            int(summary["total_rollouts"]) == expected
            and int(summary.get("missing_rollouts", 0)) == 0
            and int(summary.get("extra_rollouts", 0)) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def task_complete(task: Task) -> bool:
    return generation_complete(task) and grading_complete(task)


def command_for(task: Task, gpu: int) -> list[str]:
    return [
        str(GEN_PYTHON),
        str(GEN),
        "--model",
        str(task.model),
        "--tasks",
        task.benchmark,
        "--data-dir",
        str(DATA_ROOT),
        "--output-dir",
        str(task.output_dir),
        "--modality",
        "TEXT",
        "--model-name",
        "Qwen3-4B",
        "--variant",
        task.variant,
        "--gpus",
        str(gpu),
        "--tensor-parallel-size",
        "1",
        "--allow-other-gpus",
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
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
        "1.5",
        "--max-tokens",
        "32768",
        "--max-model-len",
        "40960",
        "--gpu-memory-utilization",
        "0.95",
        "--backend",
        "vllm",
        "--enforce-eager",
        "--trust-remote-code",
        "--response-mode",
        "full",
        "--overwrite",
    ]


def grade_command(task: Task, gpu: int) -> list[str]:
    return [
        str(GEN_PYTHON),
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


def start_task(task: Task, gpu: int, attempt: int) -> Running:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / (
        f"{task.variant}_{task.benchmark}_gpu{gpu}_attempt{attempt}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    generation = " ".join(command_for(task, gpu))
    grading = " ".join(grade_command(task, gpu))
    script = f"set -euo pipefail; {generation}; {grading}"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/mnt/disk3/hongbo/hf_cache",
            "HF_HUB_CACHE": "/mnt/disk3/hongbo/hf_cache/hub",
            "TRANSFORMERS_CACHE": "/mnt/disk3/hongbo/hf_cache/hub",
            "VLLM_CACHE_DIR": "/mnt/disk3/hongbo/vllm_cache",
            "TORCH_COMPILE_CACHE": "/mnt/disk3/hongbo/vllm_cache/torch_compile_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/mnt/disk3/hongbo/vllm_cache/torchinductor",
            "TRITON_CACHE_DIR": "/mnt/disk3/hongbo/vllm_cache/triton",
        }
    )
    log(f"START {task.label} gpu={gpu} attempt={attempt}")
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", script],
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return Running(task, process, gpu, attempt, log_path, log_handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()
    max_concurrent = min(MAX_CARDS, len(GPU_POOL), max(1, args.max_concurrent))

    if not GEN_PYTHON.is_file() or not GEN.is_file() or not GRADE.is_file():
        raise FileNotFoundError("generation or grading entrypoint is missing")
    missing_models = [
        str(Task(ratio, BENCHMARKS[0]).model)
        for ratio in RATIOS
        if not (Task(ratio, BENCHMARKS[0]).model / "config.json").is_file()
    ]
    if missing_models:
        raise FileNotFoundError(f"SAR checkpoints are not ready: {missing_models}")

    all_tasks = [Task(ratio, benchmark) for ratio in RATIOS for benchmark in BENCHMARKS]
    pending = [task for task in all_tasks if not task_complete(task)]
    done = {task.label for task in all_tasks if task_complete(task)}
    attempts: dict[str, int] = {task.label: 0 for task in all_tasks}
    running: list[Running] = []
    failed: set[str] = set()
    log(
        f"QUEUE_START tasks={len(all_tasks)} pending={len(pending)} "
        f"done={len(done)} seeds={list(SEEDS)} gpu_pool={list(GPU_POOL)} "
        f"max_concurrent={max_concurrent} max_cards={MAX_CARDS}"
    )

    while pending or running:
        for job in list(running):
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.log_handle.close()
            running.remove(job)
            complete = task_complete(job.task)
            log(
                f"REAP {job.task.label} rc={return_code} gpu={job.gpu} "
                f"complete={complete} log={job.log_path}"
            )
            if return_code == 0 and complete:
                done.add(job.task.label)
                continue
            if job.attempt < MAX_RETRIES:
                attempts[job.task.label] = job.attempt + 1
                pending.append(job.task)
                log(f"RETRY {job.task.label} attempt={job.attempt + 1}/{MAX_RETRIES}")
            else:
                failed.add(job.task.label)
                log(f"FAILED {job.task.label} after {job.attempt} attempts")

        if failed:
            raise RuntimeError(f"tasks failed: {sorted(failed)}")

        reserved = {job.gpu for job in running}
        while pending and len(running) < max_concurrent:
            available = idle_gpus(reserved)
            if not available:
                break
            gpu = available[0]
            task = pending.pop(0)
            attempts[task.label] += 1
            job = start_task(task, gpu, attempts[task.label])
            running.append(job)
            reserved.add(gpu)

        state = gpu_state()
        gpu_text = ",".join(
            f"{gpu}:{state[gpu][0]}MiB/{state[gpu][1]}%"
            for gpu in GPU_POOL
            if gpu in state
        )
        log(
            f"STATUS pending={len(pending)} running={len(running)} "
            f"done={len(done)}/{len(all_tasks)} gpu={gpu_text}"
        )
        if pending or running:
            time.sleep(POLL_SECONDS)

    log("QUEUE_FINISH all SAR ratio sweep tasks completed and graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
