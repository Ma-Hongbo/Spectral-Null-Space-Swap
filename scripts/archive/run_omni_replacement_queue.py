"""Run the Omni replacement queue without disturbing active benchmark jobs.

The old VLM queue used to place six ARKitScenes jobs after the MathVista jobs.
This controller owns only the replacement Omni jobs: the three checkpoints
(Instruct, Thinking and the ratio-1 SAR merge) on GSM8K and MMLU.  It uses
free VRAM as the scheduling signal, reserves selected cards in-process, and
passes a per-pair ``gpu_memory_utilization`` value to the vLLM runner.

Generation is append-only and all four rollout seeds are retained.  Existing
complete Omni-MMLU outputs are detected and skipped; a missing grade is
repaired without regenerating trajectories.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GEN_PYTHON = Path(
    os.environ.get(
        "OMNI_GEN_PYTHON",
        "/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python",
    )
)
GRADE_PYTHON = Path(
    os.environ.get("OMNI_GRADE_PYTHON", "/home/hongbo/.conda/envs/ar/bin/python")
)
GEN = ROOT / "eval/evals/gen_omni_vllm.py"
GRADE = ROOT / "eval/evals/grade.py"
BASE_MODEL = Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Instruct")
THINKING_MODEL = Path("/data/hongbo/sar/Qwen3-Omni-30B-A3B-Thinking")
SAR_MODEL = Path(
    "/data/hongbo/sar/Qwen3-Omni-30B-A3B-Instruct-subspace-swap-ratio1-thinking"
)
SWAP = ROOT / "model/subspace_swap.py"
OUT_ROOT = ROOT / "results/OMNI/Qwen3-Omni-30B-A3B"
LOG_ROOT = ROOT / "results/_queue_logs"
DATA = {
    "GSM8K": ROOT / "eval/data/VL_GSM8K/test.jsonl",
    "MMLU": ROOT / "eval/data/VL_MMLU/test.jsonl",
}
SEEDS = (101, 202, 303, 404)
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7, 8, 9)

POLL_SECONDS = int(os.environ.get("OMNI_QUEUE_POLL_SECONDS", "20"))
MIN_PAIR_FREE_MIB = int(os.environ.get("OMNI_MIN_PAIR_FREE_MIB", "25000"))
MIN_BUILD_FREE_MIB = int(os.environ.get("OMNI_MIN_BUILD_FREE_MIB", "30000"))
RESERVE_MIB = int(os.environ.get("OMNI_VLLM_RESERVE_MIB", "512"))
MAX_GPU_UTILIZATION = 0.95
MAX_GENERATION_ATTEMPTS = 3


@dataclass(frozen=True)
class Task:
    variant: str
    model: Path
    benchmark: str

    @property
    def data(self) -> Path:
        return DATA[self.benchmark]

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / self.variant / self.benchmark

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
    with (LOG_ROOT / "omni_replacement_queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_state() -> dict[int, tuple[int, int, int]]:
    """Return ``gpu -> (total_mib, free_mib, utilization_percent)``."""
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
        for gpu, (total, free, util) in sorted(gpu_state().items())
        if gpu in ALLOWED_GPUS
    )


def choose_pair(reserved: set[int]) -> tuple[tuple[int, int], float, tuple[int, int, int, int, int]] | None:
    state = gpu_state()
    candidates = []
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


def choose_single(reserved: set[int], minimum_free: int = 6000) -> tuple[int, float, tuple[int, int]] | None:
    state = gpu_state()
    candidates = []
    for gpu in ALLOWED_GPUS:
        if gpu in reserved or gpu not in state:
            continue
        total, free, util = state[gpu]
        if free < minimum_free or free <= RESERVE_MIB:
            continue
        dynamic = min((free - RESERVE_MIB) / total, MAX_GPU_UTILIZATION)
        candidates.append((free, gpu, dynamic, total, util))
    if not candidates:
        return None
    free, gpu, dynamic, total, util = max(candidates)
    return gpu, dynamic, (free, util)


def _output_file(task: Task) -> Path:
    return task.output_dir / (
        f"{task.benchmark.lower()}_t0.7_p0.95_k20_rp1.0_pp1.5_n4-MNT32768.jsonl"
    )


def _sample_ids(task: Task) -> list[str]:
    rows = []
    with task.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(str(row.get("example_id", len(rows))))
    return rows


def generation_complete(task: Task) -> bool:
    path = _output_file(task)
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    ids = _sample_ids(task)
    expected = {(example_id, seed) for example_id in ids for seed in SEEDS}
    actual = {(str(row.get("example_id")), int(row.get("seed"))) for row in rows}
    return len(rows) == len(expected) and actual == expected


def grade_complete(task: Task) -> bool:
    path = task.output_dir / "grading_results.json"
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        return bool(files) and all(file.get("total_rollouts", 0) == len(_sample_ids(task)) * len(SEEDS) for file in files)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def checkpoint_complete() -> bool:
    try:
        index = json.loads((BASE_MODEL / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        required = {"config.json", "model.safetensors.index.json", "subspace_swap_stats.json"}
        if not all((SAR_MODEL / name).is_file() for name in required):
            return False
        if not all((SAR_MODEL / shard).is_file() for shard in shards):
            return False
        stats = json.loads((SAR_MODEL / "subspace_swap_stats.json").read_text())
        return stats.get("summary", {}).get("method") == "P_base_subspace + P_perp_thinking"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def base_shards() -> list[str]:
    index = json.loads((BASE_MODEL / "model.safetensors.index.json").read_text())
    return sorted(set(index["weight_map"].values()))


def _build_process(shard: str, gpu: int, log_path: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "model") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        str(GEN_PYTHON),
        str(SWAP),
        "--base", str(BASE_MODEL),
        "--thinking", str(THINKING_MODEL),
        "--out", str(SAR_MODEL),
        "--ratio", "1",
        "--device", "cuda:0",
        "--only-shard", shard,
        "--skip-finalize",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process


def build_sar_checkpoint() -> bool:
    if checkpoint_complete():
        log(f"SAR_CHECKPOINT_ALREADY_READY path={SAR_MODEL}")
        return True
    SAR_MODEL.mkdir(parents=True, exist_ok=True)
    pending = []
    for shard in base_shards():
        stats = SAR_MODEL / f"subspace_swap_stats_{shard[:-len('.safetensors')]}.json"
        if not ((SAR_MODEL / shard).is_file() and stats.is_file() and stats.stat().st_size):
            pending.append(shard)
    log(f"SAR_BUILD_START pending_shards={len(pending)} min_free_mib={MIN_BUILD_FREE_MIB}")
    batch_number = 0
    while pending:
        batch_number += 1
        selections: list[tuple[str, int, subprocess.Popen[Any]]] = []
        reserved: set[int] = set()
        state = gpu_state()
        for gpu in ALLOWED_GPUS:
            if not pending or gpu not in state:
                break
            total, free, util = state[gpu]
            if free < MIN_BUILD_FREE_MIB:
                continue
            shard = pending.pop(0)
            path = LOG_ROOT / f"omni_sar_build_{shard[:-len('.safetensors')]}.log"
            process = _build_process(shard, gpu, path)
            selections.append((shard, gpu, process))
            reserved.add(gpu)
            log(f"SAR_SHARD_START batch={batch_number} shard={shard} gpu={gpu} free_mib={free} util={util}")
        if not selections:
            log(f"WAIT_SAR_BUILD_GPU min_free_mib={MIN_BUILD_FREE_MIB} snapshot={snapshot()}")
            time.sleep(POLL_SECONDS)
            continue
        failed = False
        for shard, gpu, process in selections:
            rc = process.wait()
            stats = SAR_MODEL / f"subspace_swap_stats_{shard[:-len('.safetensors')]}.json"
            complete = (SAR_MODEL / shard).is_file() and stats.is_file() and stats.stat().st_size
            log(f"SAR_SHARD_{'DONE' if rc == 0 and complete else 'FAILED'} shard={shard} gpu={gpu} rc={rc}")
            if rc != 0 or not complete:
                failed = True
                if shard not in pending:
                    pending.append(shard)
        if failed:
            log(f"SAR_BUILD_BATCH_RETRY pending_shards={len(pending)}")
    while True:
        selection = choose_single(set(), minimum_free=MIN_BUILD_FREE_MIB)
        if selection is not None:
            gpu, dynamic, details = selection
            log(f"SAR_FINALIZE_START gpu={gpu} free_mib={details[0]} util={details[1]}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(ROOT / "model") + os.pathsep + env.get("PYTHONPATH", "")
            command = [
                str(GEN_PYTHON), str(SWAP),
                "--base", str(BASE_MODEL),
                "--thinking", str(THINKING_MODEL),
                "--out", str(SAR_MODEL),
                "--ratio", "1",
                "--device", "cuda:0",
                "--finalize-only",
            ]
            path = LOG_ROOT / "omni_sar_finalize.log"
            with path.open("a", encoding="utf-8") as handle:
                rc = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            if rc == 0 and checkpoint_complete():
                log(f"SAR_BUILD_DONE path={SAR_MODEL}")
                return True
            log(f"SAR_FINALIZE_FAILED rc={rc}; retrying")
        else:
            log(f"WAIT_SAR_FINALIZE_GPU min_free_mib={MIN_BUILD_FREE_MIB} snapshot={snapshot()}")
            time.sleep(POLL_SECONDS)


def task_command(task: Task, cards: tuple[int, int], dynamic: float) -> list[str]:
    return [
        str(GEN_PYTHON), str(GEN),
        "--model", str(task.model),
        "--output-dir", str(task.output_dir),
        "--variant", task.variant,
        "--benchmark", task.benchmark,
        "--data", str(task.data),
        "--gpus", ",".join(str(card) for card in cards),
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
        "--max-num-seqs", "8",
        "--request-chunk-size", "16",
    ]


def start_generation(task: Task, cards: tuple[int, int], dynamic: float, attempt: int) -> Running:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"omni_gen_{task.variant}_{task.benchmark}_{cards[0]}_{cards[1]}.log"
    handle = path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
    process = subprocess.Popen(
        task_command(task, cards, dynamic),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(
        f"VLLM_START task={task.label} pid={process.pid} GPUs={cards} TP=2 "
        f"gpu_memory_utilization={dynamic:.3f} attempt={attempt} full_trajectory=true"
    )
    return Running(process, task, cards, "generation", attempt, path)


def start_grade(task: Task, gpu: int, dynamic: float) -> Running:
    path = LOG_ROOT / f"omni_grade_{task.variant}_{task.benchmark}.log"
    handle = path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        str(GRADE_PYTHON), str(GRADE),
        "--eval-dir", str(task.output_dir),
        "--output", str(task.output_dir / "grading_results.json"),
        "--model", "/data/hongbo/sar/CompassVerifier-3B",
        "--gpus", str(gpu),
        "--tp", "1",
        "--gpu-memory-utilization", f"{dynamic:.3f}",
        "--evaluator", "auto",
        "--write-details",
        "--verbose",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(f"GRADE_START task={task.label} pid={process.pid} gpu={gpu} gpu_memory_utilization={dynamic:.3f}")
    return Running(process, task, (gpu,), "grading", 1, path)


def tasks() -> list[Task]:
    return [
        Task("instruct", BASE_MODEL, "GSM8K"),
        Task("thinking", THINKING_MODEL, "GSM8K"),
        Task("sar_ratio_1", SAR_MODEL, "GSM8K"),
        Task("instruct", BASE_MODEL, "MMLU"),
        Task("thinking", THINKING_MODEL, "MMLU"),
        Task("sar_ratio_1", SAR_MODEL, "MMLU"),
    ]


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for task in tasks():
        if not task.data.is_file():
            log(f"MISSING_DATA task={task.label} path={task.data}")
            return 1
    log("QUEUE_START replacement=ARKitScenes->Omni-GSM8K+MMLU seeds=101,202,303,404")

    # The SAR checkpoint is a prerequisite only for its two tasks.  Building
    # it first uses currently available VRAM and avoids a half-loaded SAR
    # engine later.  Base GSM8K can start as soon as this checkpoint is ready.
    sar_ready = build_sar_checkpoint()
    if not sar_ready:
        log("SAR_CHECKPOINT_NOT_READY; SAR tasks remain pending")

    pending: list[Task] = []
    for task in tasks():
        if task.variant == "sar_ratio_1" and not sar_ready:
            pending.append(task)
            continue
        if generation_complete(task) and grade_complete(task):
            log(f"SKIP_COMPLETE task={task.label} rows={len(_sample_ids(task)) * len(SEEDS)}")
            continue
        pending.append(task)
    log("PENDING " + ",".join(task.label for task in pending) if pending else "PENDING none")

    active: list[Running] = []
    attempts: dict[str, int] = {task.label: 0 for task in pending}
    finished: set[str] = set()
    while pending or active:
        changed = False
        for run in list(active):
            return_code = run.process.poll()
            if return_code is None:
                continue
            active.remove(run)
            changed = True
            if run.stage == "generation":
                if return_code == 0:
                    # Generation cards have just been released.  Grade on a
                    # single free card and keep the other card available.
                    reserved = {card for item in active for card in item.cards}
                    selection = choose_single(reserved)
                    if selection is None:
                        log(f"WAIT_GRADE_AFTER_GENERATION task={run.task.label} snapshot={snapshot()}")
                        pending.insert(0, run.task)
                        continue
                    gpu, dynamic, _ = selection
                    active.append(start_grade(run.task, gpu, dynamic))
                    log(f"VLLM_DONE task={run.task.label} pid={run.process.pid}")
                else:
                    attempts[run.task.label] += 1
                    log(f"VLLM_FAILED task={run.task.label} pid={run.process.pid} rc={return_code} attempt={attempts[run.task.label]}")
                    if attempts[run.task.label] < MAX_GENERATION_ATTEMPTS:
                        pending.insert(0, run.task)
                    else:
                        log(f"VLLM_GAVE_UP task={run.task.label}")
            else:
                if return_code == 0 and grade_complete(run.task):
                    finished.add(run.task.label)
                    log(f"GRADE_DONE task={run.task.label} pid={run.process.pid}")
                else:
                    log(f"GRADE_FAILED task={run.task.label} pid={run.process.pid} rc={return_code}")
                    if not generation_complete(run.task):
                        pending.insert(0, run.task)
        reserved = {card for item in active for card in item.cards}
        # Fill every currently usable TP2 pair.  A pair is reserved before
        # launching so two fast scheduler iterations cannot select it twice.
        while pending:
            candidate = pending[0]
            if candidate.variant == "sar_ratio_1" and not checkpoint_complete():
                break
            if generation_complete(candidate):
                pending.pop(0)
                if grade_complete(candidate):
                    finished.add(candidate.label)
                    log(f"SKIP_COMPLETE task={candidate.label}")
                    continue
                selection = choose_single(reserved)
                if selection is None:
                    pending.insert(0, candidate)
                    break
                gpu, dynamic, _ = selection
                pending.pop(0)
                active.append(start_grade(candidate, gpu, dynamic))
                reserved.add(gpu)
                continue
            selection = choose_pair(reserved)
            if selection is None:
                break
            cards, dynamic, details = selection
            pending.pop(0)
            attempts[candidate.label] += 1
            active.append(start_generation(candidate, cards, dynamic, attempts[candidate.label]))
            reserved.update(cards)
            log(f"PAIR_SELECTED task={candidate.label} free_mib={details[0]},{details[1]} util={details[2]},{details[3]}")
        if pending and not changed:
            log(f"WAIT_OMNI_SLOT pending={','.join(task.label for task in pending)} snapshot={snapshot()}")
        if pending or active:
            time.sleep(max(5, POLL_SECONDS))
    log("QUEUE_FINISHED replacement=ARKitScenes->Omni-GSM8K+MMLU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
