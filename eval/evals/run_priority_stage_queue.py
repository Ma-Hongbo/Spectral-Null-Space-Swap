"""Priority queue for the remaining VLM and Omni text evaluations.

The queue deliberately keeps the audio benchmarks out of this process.  Its
stage barrier is:

    all VLM MathVista-testmini -> Omni GSM8K/MMLU -> audio benchmarks

The first stage can use a partially occupied card when at least 25 GiB is
free.  A genuinely empty card gets 0.95; a card with another process gets a
conservative dynamic value (at most 0.60).  Generation processes keep their
full responses and all four seeds.
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


ROOT = Path("/home/hongbo/sar-reproduction")
RESULTS = ROOT / "results"
LOG_ROOT = RESULTS / "_queue_logs"
AR_PYTHON = Path("/home/hongbo/.conda/envs/ar/bin/python")
GEN_VL = ROOT / "eval/evals/gen_vllm.py"
GRADE_VL = ROOT / "eval/evals/grade_new_vl.py"
OMNI_QUEUE = ROOT / "eval/evals/run_omni_replacement_queue.py"
DATA_DIR = ROOT / "eval/data"
MATHVISTA_DATA = DATA_DIR / "VL_MathVista/testmini.jsonl"
MEDIA_ROOT = Path("/data/hongbo/datasets/MathVista")
SEEDS = (101, 202, 303, 404)
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7, 8, 9)

POLL_SECONDS = int(os.environ.get("PRIORITY_QUEUE_POLL_SECONDS", "20"))
MIN_FREE_MIB = int(os.environ.get("PRIORITY_MIN_FREE_MIB", "25000"))
RESERVE_MIB = int(os.environ.get("PRIORITY_RESERVE_MIB", "512"))
MAX_UTILIZATION = 0.95
OCCUPIED_UTILIZATION_CAP = 0.60

VL_ROOT = RESULTS / "VL"
VL_TASK = "MathVista-testmini"
VL_TASKS = (
    {
        "label": "Qwen3-VL-2B-instruct_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-2B-Instruct",
        "model_name": "Qwen3-VL-2B",
        "variant": "instruct",
        "output_dir": VL_ROOT / "Qwen3-VL-2B/instruct/MathVista-testmini",
    },
    {
        "label": "Qwen3-VL-2B-thinking_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-2B-Thinking",
        "model_name": "Qwen3-VL-2B",
        "variant": "thinking",
        "output_dir": VL_ROOT / "Qwen3-VL-2B/thinking/MathVista-testmini",
    },
    {
        "label": "Qwen3-VL-2B-sar_ratio_1_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-2B-Instruct-subspace-swap-ratio1-thinking",
        "model_name": "Qwen3-VL-2B",
        "variant": "sar_ratio_1",
        "output_dir": VL_ROOT / "Qwen3-VL-2B/sar_ratio_1/MathVista-testmini",
    },
    {
        "label": "Qwen3-VL-4B-instruct_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-4B-Instruct",
        "model_name": "Qwen3-VL-4B",
        "variant": "instruct",
        "output_dir": VL_ROOT / "Qwen3-VL-4B/instruct/MathVista-testmini",
    },
    {
        "label": "Qwen3-VL-4B-thinking_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-4B-Thinking",
        "model_name": "Qwen3-VL-4B",
        "variant": "thinking",
        "output_dir": VL_ROOT / "Qwen3-VL-4B/thinking/MathVista-testmini",
    },
    {
        "label": "Qwen3-VL-4B-sar_ratio_1_MathVista-testmini",
        "model": "/data/hongbo/sar/Qwen3-VL-4B-Instruct-subspace-swap-ratio1-thinking",
        "model_name": "Qwen3-VL-4B",
        "variant": "sar_ratio_1",
        "output_dir": VL_ROOT / "Qwen3-VL-4B/sar_ratio_1/MathVista-testmini",
    },
)


@dataclass
class Child:
    process: subprocess.Popen[Any]
    label: str
    kind: str
    gpu: int | None = None


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (LOG_ROOT / "priority_stage_queue.log").open("a", encoding="utf-8") as handle:
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
            gpu, total, free, utilization = (int(field) for field in fields)
        except ValueError:
            continue
        state[gpu] = (total, free, utilization)
    return state


def snapshot() -> str:
    return "; ".join(
        f"gpu{gpu}:free={free}MiB,util={util}%"
        for gpu, (_total, free, util) in sorted(gpu_state().items())
        if gpu in ALLOWED_GPUS
    )


def process_args() -> list[tuple[int, str]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,args="], text=True, timeout=8
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_text, args = line.split(None, 1)
            pid = int(pid_text)
        except (ValueError, TypeError):
            continue
        if "gen_vllm.py" in args or "run_omni_replacement_queue.py" in args:
            rows.append((pid, args))
    return rows


def active_for(output_dir: Path) -> bool:
    needle = str(output_dir)
    return any(needle in args for _pid, args in process_args())


def output_file(eval_dir: Path) -> Path | None:
    manifest = eval_dir / "generation_manifest.json"
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("output_file")
            if isinstance(value, str):
                return Path(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    candidates = sorted(
        path for path in eval_dir.glob("*.jsonl")
        if not path.name.endswith("grading_details.jsonl")
    )
    return candidates[-1] if candidates else None


def generation_complete(task: dict[str, Any]) -> bool:
    path = output_file(task["output_dir"])
    if path is None or not path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected = 1000 * len(SEEDS)
    if len(rows) != expected:
        return False
    pairs = {(str(row.get("example_id")), int(row.get("seed"))) for row in rows}
    return len(pairs) == expected


def grade_complete(task: dict[str, Any]) -> bool:
    path = task["output_dir"] / "grading_results.json"
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        int(result.get("total_rollouts", -1)) == 1000 * len(SEEDS)
        and int(result.get("missing_rollouts", 1)) == 0
    )


def choose_gpu(reserved: set[int]) -> tuple[int, float, int, int] | None:
    candidates: list[tuple[int, int, float, int, int]] = []
    for gpu, (total, free, utilization) in gpu_state().items():
        if gpu not in ALLOWED_GPUS or gpu in reserved or free < MIN_FREE_MIB:
            continue
        raw = min((free - RESERVE_MIB) / total, MAX_UTILIZATION)
        if raw <= 0:
            continue
        empty = free >= int(total * 0.90) and utilization == 0
        dynamic = MAX_UTILIZATION if empty else min(raw, OCCUPIED_UTILIZATION_CAP)
        candidates.append((free, gpu, dynamic, utilization, total))
    if not candidates:
        return None
    free, gpu, dynamic, utilization, total = max(candidates)
    return gpu, dynamic, free, utilization


def start_grade(task: dict[str, Any]) -> Child:
    label = task["label"]
    log_path = LOG_ROOT / f"grade_{label}.log"
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        [str(AR_PYTHON), str(GRADE_VL), "--eval-dir", str(task["output_dir"])],
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(f"GRADE_START task={label} pid={process.pid} device=CPU")
    return Child(process, label, "vl_grade")


def start_sar_mathvista(gpu: int, dynamic: float) -> Child:
    task = VL_TASKS[-1]
    task["output_dir"].mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"gen_Qwen3-VL-4B_sar_ratio_1_MathVista-testmini_gpu{gpu}.log"
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        str(AR_PYTHON), str(GEN_VL),
        "--model", task["model"],
        "--tasks", VL_TASK,
        "--data-dir", str(DATA_DIR),
        "--task-path", f"{VL_TASK}={MATHVISTA_DATA}",
        "--output-dir", str(task["output_dir"]),
        "--modality", "VL",
        "--model-name", task["model_name"],
        "--variant", task["variant"],
        "--gpus", str(gpu),
        "--allow-other-gpus",
        "--tp", "1",
        "--seeds", ",".join(str(seed) for seed in SEEDS),
        "--n", "4",
        "--temperature", "0.7",
        "--top-p", "0.95",
        "--top-k", "20",
        "--repetition-penalty", "1.0",
        "--presence-penalty", "0.0",
        "--max-tokens", "40960",
        "--max-model-len", "40960",
        "--gpu-memory-utilization", f"{dynamic:.3f}",
        "--backend", "vllm",
        "--response-mode", "full",
        "--allowed-local-media-path", str(MEDIA_ROOT),
        "--overwrite",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(
        f"VLLM_START task={task['label']} pid={process.pid} gpu={gpu} "
        f"gpu_memory_utilization={dynamic:.3f} full_trajectory=true seeds=101,202,303,404"
    )
    return Child(process, task["label"], "vl_generation", gpu)


def start_omni_queue() -> Child:
    log_path = LOG_ROOT / "priority_omni_replacement_queue.log"
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("OMNI_QUEUE_POLL_SECONDS", "20")
    env.setdefault("OMNI_MIN_PAIR_FREE_MIB", "25000")
    env.setdefault("OMNI_VLLM_RESERVE_MIB", "512")
    process = subprocess.Popen(
        [str(AR_PYTHON), str(OMNI_QUEUE)],
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    log(
        f"STAGE2_START pid={process.pid} queue={OMNI_QUEUE} "
        "tasks=Omni-{instruct,thinking,sar_ratio_1}x{GSM8K,MMLU} "
        "audio_stage=blocked_until_stage2_done"
    )
    return Child(process, "Omni-text-stage", "omni_queue")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not MATHVISTA_DATA.is_file():
        log(f"ERROR missing MathVista data: {MATHVISTA_DATA}")
        return 1
    log(
        "QUEUE_START order=MathVista -> Omni-GSM8K+MMLU -> "
        "MMAU+MMAR+MMSU; gpu4=excluded; dynamic_vllm=true"
    )
    log("INITIAL_GPU " + snapshot())

    children: list[Child] = []
    sar_retry_after = 0.0
    omni_started = False
    omni_finished = False
    grade_started: set[str] = set()

    while True:
        now = time.time()
        for child in list(children):
            rc = child.process.poll()
            if rc is None:
                continue
            children.remove(child)
            if child.kind == "vl_generation":
                complete = generation_complete(VL_TASKS[-1])
                log(
                    f"VLLM_EXIT task={child.label} pid={child.process.pid} "
                    f"rc={rc} complete={complete}"
                )
                if rc != 0 or not complete:
                    sar_retry_after = now + 120
            elif child.kind == "vl_grade":
                task = next(item for item in VL_TASKS if item["label"] == child.label)
                log(
                    f"GRADE_EXIT task={child.label} pid={child.process.pid} "
                    f"rc={rc} complete={grade_complete(task)}"
                )
            else:
                omni_finished = True
                log(
                    f"STAGE2_EXIT pid={child.process.pid} rc={rc}; "
                    "STAGE3_AUDIO remains pending by requested order"
                )

        # CPU-only grading is independent of the GPU generation jobs.
        for task in VL_TASKS:
            if generation_complete(task) and not grade_complete(task):
                if task["label"] in grade_started:
                    continue
                if any(child.kind == "vl_grade" and child.label == task["label"] for child in children):
                    grade_started.add(task["label"])
                    continue
                children.append(start_grade(task))
                grade_started.add(task["label"])

        sar_task = VL_TASKS[-1]
        if (
            not generation_complete(sar_task)
            and not active_for(sar_task["output_dir"])
            and now >= sar_retry_after
            and not any(child.kind == "vl_generation" for child in children)
        ):
            selection = choose_gpu(set())
            if selection is not None:
                gpu, dynamic, free, utilization = selection
                children.append(start_sar_mathvista(gpu, dynamic))
                log(
                    f"SINGLE_GPU_SELECTED task={sar_task['label']} gpu={gpu} "
                    f"free_mib={free} external_util={utilization}"
                )
            elif not active_for(sar_task["output_dir"]):
                log(f"WAIT_VL_SLOT task={sar_task['label']} snapshot={snapshot()}")

        vl_barrier = all(generation_complete(task) and grade_complete(task) for task in VL_TASKS)
        if vl_barrier and not omni_started and not any(
            "run_omni_replacement_queue.py" in args for _pid, args in process_args()
        ):
            children.append(start_omni_queue())
            omni_started = True

        if omni_finished:
            log("QUEUE_FINISHED_STAGE2; audio tasks intentionally not started")
            return 0

        if not vl_barrier:
            status = ",".join(
                f"{task['label']}="
                f"{'done' if generation_complete(task) and grade_complete(task) else 'pending'}"
                for task in VL_TASKS
            )
            log(f"STATUS stage=MathVista {status} snapshot={snapshot()}")
        time.sleep(max(5, POLL_SECONDS))


if __name__ == "__main__":
    raise SystemExit(main())
