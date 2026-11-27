"""Keep the newly requested VLM benchmark queue full as GPU groups free up.

The three initial groups are already occupied by long-running jobs from the
experiment.  Once one of those parent processes exits, this scheduler starts
the next unstarted (model, benchmark) pair on that same group.  GPU 4 is
intentionally absent because it is the known ERROR card.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from collections import deque
from pathlib import Path


ROOT = Path("/home/hongbo/sar-reproduction")
PYTHON = Path("/home/hongbo/.conda/envs/ar/bin/python")
RUNNER = ROOT / "eval/evals/gen_vllm.py"
GRADER = ROOT / "eval/evals/grade_new_vl.py"
RESULTS = ROOT / "results"
DATA_DIR = ROOT / "eval/data"
MATHVISTA_DATA = ROOT / "eval/data/VL_MathVista/testmini.jsonl"
MATHVISTA_MEDIA_ROOT = Path("/data/hongbo/datasets/MathVista")


MODELS = (
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B", "instruct", "instruct"),
    ("Qwen3-VL-2B-Thinking", "Qwen3-VL-2B", "thinking", "thinking"),
    (
        "Qwen3-VL-2B-Instruct-subspace-swap-ratio1-thinking",
        "Qwen3-VL-2B",
        "sar_ratio_1",
        "sar_ratio_1",
    ),
    ("Qwen3-VL-4B-Instruct", "Qwen3-VL-4B", "instruct", "instruct"),
    ("Qwen3-VL-4B-Thinking", "Qwen3-VL-4B", "thinking", "thinking"),
    (
        "Qwen3-VL-4B-Instruct-subspace-swap-ratio1-thinking",
        "Qwen3-VL-4B",
        "sar_ratio_1",
        "sar_ratio_1",
    ),
)


def _tasks() -> deque[tuple[str, str, str, str, str]]:
    queue: deque[tuple[str, str, str, str, str]] = deque()
    model_tasks = []
    for model_path, model_name, variant, result_variant in MODELS:
        model_tasks.append((model_path, model_name, variant, result_variant, "MathVista-testmini"))
    # The first MathVista task was launched before this scheduler.  Do not
    # duplicate it; it remains the first completed item in the overall queue.
    # A resumed controller can also be handed active 2B Thinking/SAR jobs;
    # those paths are supplied through NEW_VL_SKIP_MODEL_PATHS so that the
    # remaining 4B jobs are the next items instead of being regenerated.
    skip_paths = {
        "Qwen3-VL-2B-Instruct",
        *filter(None, os.environ.get("NEW_VL_SKIP_MODEL_PATHS", "").split(",")),
    }
    queue.extend(item for item in model_tasks if item[0] not in skip_paths)
    return queue


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ready(benchmark: str) -> bool:
    return benchmark == "MathVista-testmini" and MATHVISTA_DATA.is_file()


def _output_dir(model_name: str, variant: str, benchmark: str) -> Path:
    return RESULTS / "VL" / model_name / variant / benchmark


def _command(item: tuple[str, str, str, str, str], gpus: str) -> list[str]:
    model_path, model_name, variant, result_variant, benchmark = item
    data_path = MATHVISTA_DATA
    media_root = MATHVISTA_MEDIA_ROOT
    return [
        str(PYTHON), str(RUNNER),
        "--model", str(Path("/data/hongbo/sar") / model_path),
        "--tasks", benchmark,
        "--data-dir", str(DATA_DIR),
        "--task-path", f"{benchmark}={data_path}",
        "--output-dir", str(_output_dir(model_name, result_variant, benchmark)),
        "--modality", "VL",
        "--model-name", model_name,
        "--variant", variant,
        "--gpus", gpus,
        "--allow-other-gpus",
        "--tp", "1",
        "--seeds", "101,202,303,404",
        "--n", "4",
        "--temperature", "0.7",
        "--top-p", "0.95",
        "--top-k", "20",
        "--repetition-penalty", "1.0",
        "--presence-penalty", "0.0",
        "--max-tokens", "40960",
        "--max-model-len", "40960",
        "--gpu-memory-utilization", "0.50",
        "--backend", "vllm",
        "--response-mode", "full",
        "--allowed-local-media-path", str(media_root),
        "--overwrite",
    ]


def _start_grader(item: tuple[str, str, str, str, str], log_dir: Path) -> None:
    _, model_name, result_variant, _, benchmark = item
    output = _output_dir(model_name, result_variant, benchmark)
    log_path = log_dir / f"grade_{model_name}_{result_variant}_{benchmark}.log"
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(GRADER), "--eval-dir", str(output)],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    print(f"started grader pid={process.pid} benchmark={benchmark} output={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathvista-pid", type=int, default=2220062)
    parser.add_argument("--mmmu-pid", type=int, default=2188992)
    parser.add_argument("--text-pid", type=int, default=1994181)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--log-dir", type=Path, default=RESULTS / "_queue_logs")
    parser.add_argument("--lock", type=Path, default=RESULTS / ".new_vl_queue.lock")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("new VLM queue is already running")

        pending = _tasks()
        groups = [
            {"name": "A", "gpus": "1,2,7,9", "pid": args.mathvista_pid},
            {"name": "B", "gpus": "3,6,8", "pid": args.mmmu_pid},
            {"name": "C", "gpus": "0,5", "pid": args.text_pid},
        ]
        active: dict[str, subprocess.Popen[bytes]] = {}
        while pending or active:
            for group in groups:
                name = group["name"]
                process = active.get(name)
                if process is not None:
                    if process.poll() is None:
                        continue
                    print(f"queue task on group {name} exited rc={process.returncode}", flush=True)
                    del active[name]
                    _start_grader(process._codex_item, args.log_dir)  # type: ignore[attr-defined]
                    continue
                if not pending:
                    continue
                if _pid_alive(int(group["pid"])):
                    continue
                candidate = pending[0]
                if not _ready(candidate[4]):
                    continue
                pending.popleft()
                log_path = args.log_dir / f"gen_{candidate[1]}_{candidate[3]}_{candidate[4]}_{name}.log"
                handle = log_path.open("a", encoding="utf-8")
                command = _command(candidate, str(group["gpus"]))
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                handle.close()
                process._codex_item = candidate  # type: ignore[attr-defined]
                active[name] = process
                print(
                    f"started group={name} pid={process.pid} gpus={group['gpus']} "
                    f"model={candidate[1]} variant={candidate[3]} benchmark={candidate[4]}",
                    flush=True,
                )
            if pending or active:
                time.sleep(max(5, args.poll_seconds))
        print("new VLM queue completed", flush=True)


if __name__ == "__main__":
    main()
