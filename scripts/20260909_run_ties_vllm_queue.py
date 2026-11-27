#!/usr/bin/env python3
"""Dynamically evaluate the five anchor-based TIES checkpoints with vLLM.

Every checkpoint in this queue follows the same construction:

    output = Instruct + TIES(Thinking - Instruct)

The queue is deliberately independent from the older MI queues.  It observes
GPU framebuffer use before launching a task, keeps each selected card reserved
until generation and grading finish, and retries transient vLLM failures.  GPU
4 currently reports ``reset_status=Yes`` and is excluded from the pool; the
other nine cards, including GPU 2, passed per-device CUDA checks.  Existing
jobs on the other cards are never signaled.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/home/hongbo/sar-reproduction")
RESULTS = ROOT / "results"
TIES_ROOT = Path("/mnt/disk3/hongbo/sar/ties-models")
DATA_ROOT = ROOT / "eval/data"
MATHVISTA_DATA = Path("/mnt/disk3/hongbo/sar/local_eval_data/VL_MathVista/testmini.jsonl")
MATHVISTA_MEDIA = Path("/mnt/disk3/hongbo/datasets/MathVista")
AUDIO_ROOT = Path("/mnt/disk3/hongbo/audio_benchmarks/prepared")
VERIFIER = Path("/mnt/disk3/hongbo/sar/CompassVerifier-3B")

PYTHON = Path("/home/hongbo/.conda/envs/ar/bin/python")
OMNI_PYTHON = Path("/home/hongbo/.conda/envs/qwen3-omni-vllm/bin/python")
GEN_VLLM = ROOT / "eval/evals/gen_vllm.py"
GEN_OMNI_TEXT = ROOT / "eval/evals/gen_omni_vllm.py"
GEN_OMNI_AUDIO = ROOT / "eval/evals/gen_omni_audio_vllm.py"
GRADE = ROOT / "eval/evals/grade.py"
GRADE_VL = ROOT / "eval/evals/grade_new_vl.py"
GRADE_AUDIO = ROOT / "eval/evals/grade_omni_audio.py"
UPDATE_OMNI_THINKING_GSM8K = ROOT / "scripts/20260909_update_omni_thinking_gsm8k_matrix.py"

SEEDS = (101, 202, 303, 404)
# GPU 2 is healthy again and is included.  GPU 4 currently reports
# reset_status=Yes and fails CUDA initialization even though nvidia-smi can
# query it, so it must remain excluded until an administrator resets it.
GPU_POOL = tuple(
    int(item.strip())
    for item in os.environ.get("TIES_GPU_POOL", "0,1,2,3,5,6,7,8,9").split(",")
    if item.strip()
)
GPU_IDLE_MEMORY_MIB = 2000
GPU_IDLE_UTILIZATION = 5
POLL_SECONDS = max(10, int(os.environ.get("TIES_QUEUE_POLL_SECONDS", "20")))
STATUS_SECONDS = max(60, int(os.environ.get("TIES_QUEUE_STATUS_SECONDS", "600")))
MAX_RETRIES = max(1, int(os.environ.get("TIES_QUEUE_MAX_RETRIES", "2")))
GPU_MEMORY_UTILIZATION = "0.95"

CONTROLLER_LOG = RESULTS / "_queue_logs/20260909_ties_vllm_controller.log"
LOG_ROOT = RESULTS / "_queue_logs/20260909_ties_vllm"


@dataclass(frozen=True)
class Task:
    key: str
    kind: str
    model_name: str
    model: Path
    benchmark: str
    output_dir: Path
    gpu_count: int


@dataclass
class Running:
    task: Task
    process: Any
    cards: tuple[int, ...]
    attempt: int
    log_path: Path
    log_handle: Any | None


class AdoptedProcess:
    """Poll a task shell inherited from a previous queue controller."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> int | None:
        try:
            result = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(self.pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        state = result.stdout.strip()
        if not state or state.startswith("Z"):
            return 0
        return None


def log(message: str) -> None:
    CONTROLLER_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def csv(values: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(value) for value in values)


def env_for(kind: str) -> dict[str, str]:
    env = os.environ.copy()
    cache_root = Path("/mnt/disk3/hongbo/vllm_cache")
    hf_root = Path("/mnt/disk3/hongbo/hf_cache")
    env.update(
        {
            "HF_HOME": str(hf_root),
            "HF_HUB_CACHE": str(hf_root / "hub"),
            "TRANSFORMERS_CACHE": str(hf_root / "hub"),
            "HF_DATASETS_CACHE": str(hf_root / "datasets"),
            "VLLM_CACHE_DIR": str(cache_root),
            "TORCH_COMPILE_CACHE": str(cache_root / "torch_compile_cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "XDG_CACHE_HOME": str(Path("/mnt/disk3/hongbo/.cache")),
            "VLLM_NO_USAGE_STATS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    # vLLM 0.11.2 in the ar environment is V1-only.  The Omni environment
    # intentionally keeps V0 because its Qwen branch is registered there.
    if kind.startswith("omni"):
        env["VLLM_USE_V1"] = "0"
    else:
        env["VLLM_USE_V1"] = "1"
    for path in (
        hf_root,
        hf_root / "hub",
        hf_root / "datasets",
        cache_root,
        cache_root / "torch_compile_cache",
        cache_root / "torchinductor",
        cache_root / "triton",
        Path(env["XDG_CACHE_HOME"]),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return env


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
        if gpu in GPU_POOL:
            state[gpu] = (memory, utilization)
    return state


def idle_cards(reserved: set[int]) -> list[int]:
    state = gpu_state()
    return [
        gpu
        for gpu in GPU_POOL
        if gpu not in reserved
        and gpu in state
        and state[gpu][0] <= GPU_IDLE_MEMORY_MIB
        and state[gpu][1] <= GPU_IDLE_UTILIZATION
    ]


def choose_cards(count: int, reserved: set[int]) -> tuple[int, ...] | None:
    cards = idle_cards(reserved)
    if len(cards) < count:
        return None
    # Stable ordering keeps logs easy to compare.  The first available group
    # is sufficient because all cards in the pool have the same role here.
    return tuple(cards[:count])


def reservation_cards(running: list[Running]) -> set[int]:
    """Keep only genuinely active cards reserved for split VL workers.

    MathVista-testmini is sharded as independent TP=1 workers.  Once one of
    those workers finishes, its card has no CUDA process while the remaining
    shard continues on the other cards.  Releasing only those demonstrably
    idle cards lets the queue start another task immediately.  Likewise,
    once a normal generation has completed, its sequential verifier uses only
    the first card; the remaining cards can be assigned to a waiting task.
    All cards remain reserved during model loading and generation.
    """
    state = gpu_state()
    reserved: set[int] = set()
    for item in running:
        task = item.task
        if task.kind == "vl" and task.benchmark == "MathVista-testmini":
            active = {
                card
                for card in item.cards
                if card not in state
                or state[card][0] > GPU_IDLE_MEMORY_MIB
                or state[card][1] > GPU_IDLE_UTILIZATION
            }
            # During initialisation (or after all workers have exited), keep
            # the complete allocation reserved until the parent is reaped.
            if active and len(active) < len(item.cards):
                reserved.update(active)
                continue
        # gen_vllm.py and gen_omni_vllm.py run grading in the same shell after
        # writing a complete manifest.  The normal verifier is pinned to the
        # first card, so release the other cards during this short grading
        # phase and let the queue fill them with pending work.
        if generation_complete(task) and not grade_complete(task):
            reserved.add(item.cards[0])
            continue
        reserved.update(item.cards)
    return reserved


def pending_multicard_requirement(pending: list[Task]) -> int | None:
    """Return the largest ready multi-card allocation still waiting.

    A handoff can leave single-card jobs running on every card while a
    larger multi-card model waits forever for a simultaneous group.  When the
    largest ready allocation can be made, prefer it before filling the same
    group with smaller jobs.  If fewer cards are free, smaller jobs may still
    use them so that available hardware is not left idle.
    """
    requirements = [
        task.gpu_count
        for task in pending
        if task.gpu_count > 1 and model_ready(task.model)
    ]
    return max(requirements) if requirements else None


def model_ready(model: Path) -> bool:
    if not (model / "config.json").is_file():
        return False
    # The merger creates metadata and weight shards incrementally.  Requiring
    # the final stats/index pair prevents vLLM from observing a half-written
    # checkpoint when a GPU becomes idle during the merge.
    if not (model / "ties_merge_stats.json").is_file():
        return False
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        return False
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        shard_names = sorted(set(str(value) for value in weight_map.values()))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return False
    return bool(shard_names) and all((model / name).is_file() for name in shard_names)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def manifest_output(directory: Path, manifest: dict[str, Any]) -> Path | None:
    value = manifest.get("output_file")
    if not isinstance(value, str):
        return None
    path = Path(value)
    return path if path.is_absolute() else directory / path


def generation_complete(task: Task) -> bool:
    manifest = read_json(task.output_dir / "generation_manifest.json")
    if manifest is None:
        return False
    try:
        if manifest.get("backend") != "vllm":
            return False
        seeds = [int(seed) for seed in manifest["seeds"]]
        if seeds != list(SEEDS):
            return False
        examples = int(manifest["num_examples"])
        output = manifest_output(task.output_dir, manifest)
        if output is None or not output.is_file() or output.stat().st_size <= 0:
            return False
        expected = examples * len(SEEDS)
        if int(manifest.get("num_rows", -1)) != expected:
            return False
        rows_per_example = manifest.get("rows_per_example")
        if isinstance(rows_per_example, dict):
            if len(rows_per_example) != examples:
                return False
            if any(int(value) != len(SEEDS) for value in rows_per_example.values()):
                return False
        # A second lightweight check catches a truncated JSONL whose manifest
        # was written before a process was interrupted.
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != expected:
            return False
        keys = {(str(row.get("example_id")), int(row.get("seed"))) for row in rows}
        return len(keys) == expected
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def grade_complete(task: Task) -> bool:
    result = read_json(task.output_dir / "grading_results.json")
    if result is None:
        return False
    try:
        total = int(result.get("total_rollouts", -1))
        if task.benchmark == "MathVista-testmini":
            expected = int(result.get("expected_total_rollouts", 0))
            return (
                total == expected
                and int(result.get("missing_rollouts", 1)) == 0
                and sorted(result.get("pass1_by_seed", {}).keys())
                == [str(seed) for seed in SEEDS]
            )
        if task.kind == "omni_audio":
            overall = result.get("overall", {})
            return int(overall.get("total", -1)) == 4000 or total == 4000
        files = result.get("files")
        if not isinstance(files, list) or len(files) != 1:
            return False
        summary = files[0]
        expected = int(summary.get("num_problems", 0)) * int(
            summary.get("expected_rollouts_per_problem", len(SEEDS))
        )
        return (
            int(summary.get("total_rollouts", -1)) == expected
            and int(summary.get("missing_rollouts", 0)) == 0
            and int(summary.get("extra_rollouts", 0)) == 0
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def output_file(task: Task) -> Path | None:
    manifest = read_json(task.output_dir / "generation_manifest.json")
    if manifest:
        return manifest_output(task.output_dir, manifest)
    if task.kind == "omni_audio":
        # The audio runner uses this canonical name before its first manifest
        # write.  This lets the generation+grading reservation be built in a
        # single scheduler pass.
        return task.output_dir / (
            f"{task.benchmark.lower()}_t0.6_p0.95_k20_rp1.0_pp0.0_n4-MNT16384.jsonl"
        )
    return None


def generation_command(task: Task, cards: tuple[int, ...]) -> list[str]:
    seeds = csv(list(SEEDS))
    if task.kind in {"text", "vl"}:
        # MathVista can produce a large multimodal encoder batch.  For both VL
        # checkpoints, keep TP=1 and shard examples over the selected cards so
        # one vision batch does not exhaust a 46-GiB card.  Sampling and
        # model-length settings remain identical to the standard task config.
        tensor_parallel_size = 1 if (
            task.kind == "vl"
            and task.model_name in {"Qwen3-VL-2B", "Qwen3-VL-4B"}
            and task.benchmark == "MathVista-testmini"
        ) else (len(cards) if task.gpu_count > 1 else 1)
        command = [
            str(PYTHON),
            str(GEN_VLLM),
            "--model",
            str(task.model),
            "--tasks",
            task.benchmark,
            "--data-dir",
            str(DATA_ROOT),
            "--output-dir",
            str(task.output_dir),
            "--modality",
            "TEXT" if task.kind == "text" else "VL",
            "--model-name",
            task.model_name,
            "--variant",
            "ties",
            "--gpus",
            csv(cards),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--seeds",
            seeds,
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
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            "--max-model-len",
            "40960",
            "--backend",
            "vllm",
            "--enforce-eager",
            "--trust-remote-code",
            "--response-mode",
            "full",
            "--allow-other-gpus",
            "--overwrite",
        ]
        if task.kind == "text":
            command += [
                "--presence-penalty",
                "1.5",
                "--max-tokens",
                "32768",
            ]
        elif task.benchmark in {"MMMU", "MathVista-testmini"}:
            command += [
                "--presence-penalty",
                "0.0",
                "--max-tokens",
                "40960",
            ]
        else:
            command += [
                "--presence-penalty",
                "1.5",
                "--max-tokens",
                "32768",
            ]
        if task.benchmark == "AIME26":
            command += [
                "--task-path",
                "AIME26=/home/hongbo/MultiLingual/data/aime2026.jsonl",
            ]
        if task.benchmark == "MathVista-testmini":
            command += [
                "--task-path",
                f"MathVista-testmini={MATHVISTA_DATA}",
                "--allowed-local-media-path",
                str(MATHVISTA_MEDIA),
            ]
        return command

    if task.kind == "omni_text":
        data = DATA_ROOT / "VL_GSM8K/test.jsonl" if task.benchmark == "GSM8K" else DATA_ROOT / "VL_MMLU/test.jsonl"
        return [
            str(OMNI_PYTHON),
            str(GEN_OMNI_TEXT),
            "--model",
            str(task.model),
            "--output-dir",
            str(task.output_dir),
            "--benchmark",
            task.benchmark,
            "--variant",
            "ties",
            "--data",
            str(data),
            "--gpus",
            csv(cards),
            "--seeds",
            seeds,
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
            GPU_MEMORY_UTILIZATION,
            "--max-num-seqs",
            "8",
            "--request-chunk-size",
            "16",
        ]

    if task.kind == "omni_audio":
        data = AUDIO_ROOT / {
            "MMAR": "MMAR/samples.jsonl",
            "MMSU": "MMSU/samples.jsonl",
            "MMAU_testmini": "MMAU/samples.jsonl",
        }[task.benchmark]
        return [
            str(OMNI_PYTHON),
            str(GEN_OMNI_AUDIO),
            "--model",
            str(task.model),
            "--output-dir",
            str(task.output_dir),
            "--benchmark",
            task.benchmark,
            "--variant",
            "ties",
            "--data",
            str(data),
            "--gpus",
            csv(cards),
            "--pipeline-parallel-size",
            "1",
            "--seeds",
            seeds,
            "--temperature",
            "0.6",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--repetition-penalty",
            "1.0",
            "--presence-penalty",
            "0.0",
            "--max-tokens",
            "16384",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            "--max-num-seqs",
            "8",
            # Keep eight active multimodal sequences, but submit a larger
            # request window so vLLM's continuous batcher can replace short
            # audio generations while one request is still decoding.
            "--request-chunk-size",
            "32",
        ]

    raise ValueError(f"unknown task kind: {task.kind}")


def grading_command(task: Task, cards: tuple[int, ...]) -> list[str]:
    first = str(cards[0])
    if task.benchmark == "MathVista-testmini":
        return [str(PYTHON), str(GRADE_VL), "--eval-dir", str(task.output_dir)]
    if task.kind == "omni_audio":
        input_path = output_file(task)
        if input_path is None:
            raise RuntimeError(f"cannot resolve audio output for {task.key}")
        return [
            str(PYTHON),
            str(GRADE_AUDIO),
            "--input",
            str(input_path),
            "--output",
            str(task.output_dir / "grading_results.json"),
            "--write-details",
        ]
    return [
        str(PYTHON),
        str(GRADE),
        "--eval-dir",
        str(task.output_dir),
        "--model",
        str(VERIFIER),
        "--gpus",
        first,
        "--tp",
        "1",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--evaluator",
        "auto",
        "--write-details",
        "--verbose",
        "--allow-other-gpus",
    ]


def task_command(task: Task, cards: tuple[int, ...]) -> tuple[list[str], str]:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    generation_done = generation_complete(task)
    grading_done = grade_complete(task)

    # The standalone Omni Thinking GSM8K generation is already complete. Keep
    # its formal grading inside this controller so it can use the first card
    # released by the queue, instead of waiting for all 29 TIES tasks to drain.
    # The matrix updater stays in the same reservation to avoid a second
    # watcher racing the next task onto the card.
    if task.kind == "priority_grade":
        if not generation_done:
            raise RuntimeError(f"priority grading artifacts are not ready: {task.key}")
        commands = []
        if not grading_done:
            commands.append(shlex.join(grading_command(task, cards)))
        commands.append(
            shlex.join([str(PYTHON), str(UPDATE_OMNI_THINKING_GSM8K)])
        )
        return [
            "bash",
            "-lc",
            "set -euo pipefail; " + " && ".join(commands),
        ], "; ".join(commands)

    commands: list[str] = []
    if not generation_done:
        commands.append(shlex.join(generation_command(task, cards)))
    if not grading_done:
        commands.append(shlex.join(grading_command(task, cards)))
    if not commands:
        commands.append("true")
    # Keep the whole generation+grading pair inside one reservation.  The
    # shell is only used as a strict sequencer; every argument is shlex-quoted.
    return ["bash", "-lc", "set -euo pipefail; " + " && ".join(commands)], "; ".join(commands)


def make_tasks() -> list[Task]:
    tasks: list[Task] = []

    # This is a completed four-seed vLLM generation awaiting only formal
    # CompassVerifier grading. It is intentionally inserted before the 29
    # regular TIES tasks so the first available valid GPU is used immediately.
    tasks.append(
        Task(
            key="omni-thinking-GSM8K-formal-grade",
            kind="priority_grade",
            model_name="Qwen3-Omni-30B-A3B",
            model=TIES_ROOT / "Qwen3-Omni-30B-A3B-TIES",
            benchmark="GSM8K",
            output_dir=RESULTS / "OMNI/Qwen3-Omni-30B-A3B/thinking/GSM8K",
            gpu_count=1,
        )
    )

    def add(
        key: str,
        kind: str,
        model_name: str,
        model_dir: str,
        benchmark: str,
        modality_dir: str,
        gpu_count: int,
    ) -> None:
        tasks.append(
            Task(
                key=key,
                kind=kind,
                model_name=model_name,
                model=TIES_ROOT / model_dir,
                benchmark=benchmark,
                output_dir=RESULTS / modality_dir / model_name / "ties" / benchmark,
                gpu_count=gpu_count,
            )
        )

    # Put 2-GPU work first so a newly free pair is preferentially used by a
    # multi-card task.  If only one card is free, the scheduler falls through
    # to the first compatible single-card task instead of leaving that card
    # idle.
    for benchmark in ("AIME24", "AIME25", "AIME26", "AMC23", "HMMT25", "MATH-500", "Olympiad-Bench", "CMIMC25"):
        add("text30-" + benchmark, "text", "Qwen3-30B-A3B", "Qwen3-30B-A3B-TIES", benchmark, "TEXT", 2)
    for benchmark in ("GSM8K", "MMLU"):
        add("omni-text-" + benchmark, "omni_text", "Qwen3-Omni-30B-A3B", "Qwen3-Omni-30B-A3B-TIES", benchmark, "OMNI", 2)
    # Keep at least eight healthy cards occupied while the three incomplete
    # audio tasks are resumed.  TP=4 is valid for Qwen3-Omni, and the
    # existing partial rows remain append-only; a task may therefore contain
    # rows from its earlier TP=2 attempt and its resumed TP=4 attempt.
    audio_gpu_count = {"MMAR": 4, "MMSU": 2, "MMAU_testmini": 2}
    for benchmark in ("MMAR", "MMSU", "MMAU_testmini"):
        add(
            "omni-audio-" + benchmark,
            "omni_audio",
            "Qwen3-Omni-30B-A3B",
            "Qwen3-Omni-30B-A3B-TIES",
            benchmark,
            "OMNI",
            audio_gpu_count[benchmark],
        )
    for benchmark in ("AIME24", "AIME25", "AIME26", "AMC23", "HMMT25", "MATH-500", "Olympiad-Bench", "CMIMC25"):
        add("text4-" + benchmark, "text", "Qwen3-4B", "Qwen3-4B-TIES", benchmark, "TEXT", 1)
    for model_name, model_dir in (
        ("Qwen3-VL-2B", "Qwen3-VL-2B-TIES"),
        ("Qwen3-VL-4B", "Qwen3-VL-4B-TIES"),
    ):
        for benchmark in ("GSM8K", "MMLU", "MMMU", "MathVista-testmini"):
            add(
                f"{model_name}-" + benchmark,
                "vl",
                model_name,
                model_dir,
                benchmark,
                "VL",
                (
                    3
                    if benchmark == "MathVista-testmini"
                    else 1
                ),
            )
    return tasks


def status_line(pending: list[Task], running: list[Running], done: set[str], failed: set[str]) -> None:
    state = gpu_state()
    gpu_text = ", ".join(
        f"{gpu}:{state[gpu][0]}MiB/{state[gpu][1]}%" for gpu in GPU_POOL if gpu in state
    )
    running_text = ", ".join(
        f"{item.task.key}[pid={item.process.pid},gpu={csv(item.cards)}]" for item in running
    ) or "none"
    log(
        "STATUS "
        f"pending={len(pending)} running={len(running)} done={len(done)} failed={len(failed)} "
        f"running_tasks={running_text} gpu={gpu_text}"
    )


def parse_adopt_spec(spec: str, task_map: dict[str, Task]) -> tuple[Task, int, tuple[int, ...]]:
    try:
        key, pid_text, cards_text = spec.split(":", 2)
        task = task_map[key]
        pid = int(pid_text)
        cards = tuple(int(value) for value in cards_text.split(",") if value)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"invalid --adopt {spec!r}; expected TASK_KEY:PID:GPU[,GPU...]"
        ) from exc
    if not cards or any(card not in GPU_POOL for card in cards):
        raise ValueError(f"adopted cards must be in GPU_POOL: {spec!r}")
    return task, pid, cards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adopt",
        action="append",
        default=[],
        metavar="TASK_KEY:PID:GPU[,GPU...]",
        help="adopt a task shell left running by an older queue controller",
    )
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    pending = list(tasks)
    running: list[Running] = []
    done: set[str] = set()
    failed: set[str] = set()
    attempts: dict[str, int] = {}
    last_status = 0.0

    log(
        "QUEUE_START scheme=Instruct_anchor+TIES(Thinking-Instruct) "
        f"tasks={len(tasks)} (29 TIES + priority Omni grade) "
        f"seeds={list(SEEDS)} gpu_pool={list(GPU_POOL)} "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}"
    )

    task_map = {task.key: task for task in tasks}
    for spec in args.adopt:
        task, pid, cards = parse_adopt_spec(spec, task_map)
        if task in pending:
            pending.remove(task)
        if generation_complete(task) and grade_complete(task):
            done.add(task.key)
            log(f"ADOPT_SKIP_COMPLETE task={task.key} pid={pid}")
            continue
        attempts[task.key] = 1
        adopted_log = LOG_ROOT / f"{task.key.replace('/', '_')}_adopted_pid{pid}_g{csv(cards)}.log"
        running.append(
            Running(
                task=task,
                process=AdoptedProcess(pid),
                cards=cards,
                attempt=1,
                log_path=adopted_log,
                log_handle=None,
            )
        )
        log(
            f"ADOPT task={task.key} pid={pid} gpu={csv(cards)} "
            "reservation=kept_until_exit"
        )

    while pending or running:
        # Reap finished children first so their cards can be reused in this
        # same poll cycle.
        for item in list(running):
            return_code = item.process.poll()
            if return_code is None:
                continue
            if item.log_handle is not None:
                item.log_handle.close()
            running.remove(item)
            complete = generation_complete(item.task) and grade_complete(item.task)
            log(
                f"REAP task={item.task.key} pid={item.process.pid} rc={return_code} "
                f"gpu={csv(item.cards)} complete={complete} log={item.log_path}"
            )
            if return_code == 0 and complete:
                done.add(item.task.key)
                continue
            attempt = attempts.get(item.task.key, item.attempt)
            if attempt < MAX_RETRIES:
                attempts[item.task.key] = attempt + 1
                pending.append(item.task)
                log(f"RETRY task={item.task.key} attempt={attempt + 1}/{MAX_RETRIES}")
            else:
                failed.add(item.task.key)
                log(f"FAILED_FINAL task={item.task.key} log={item.log_path}")

        # Remove tasks completed by a previous invocation or a partial manual
        # run before allocating a GPU to them.
        for task in list(pending):
            if generation_complete(task) and grade_complete(task):
                pending.remove(task)
                done.add(task.key)
                log(f"SKIP_COMPLETE task={task.key}")

        reserved = reservation_cards(running)
        launched = True
        while launched:
            launched = False
            largest_pending_multicard = pending_multicard_requirement(pending)
            available_idle = idle_cards(reserved)
            for task in list(pending):
                if not model_ready(task.model):
                    continue
                if task.kind == "priority_grade" and not generation_complete(task):
                    continue
                if (
                    largest_pending_multicard is not None
                    and task.gpu_count < largest_pending_multicard
                    and len(available_idle) >= largest_pending_multicard
                ):
                    continue
                cards = choose_cards(task.gpu_count, reserved)
                if cards is None:
                    continue
                command, command_text = task_command(task, cards)
                attempt = attempts.get(task.key, 0) + 1
                attempts[task.key] = attempt
                log_path = LOG_ROOT / f"{task.key.replace('/', '_')}_attempt{attempt}_g{csv(cards)}.log"
                log_handle = log_path.open("ab", buffering=0)
                log(
                    f"LAUNCH task={task.key} kind={task.kind} model={task.model} "
                    f"gpu={csv(cards)} attempt={attempt} log={log_path}"
                )
                log(f"COMMAND task={task.key} {command_text}")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env_for(task.kind),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as exc:
                    log_handle.close()
                    log(f"LAUNCH_FAILED task={task.key} error={type(exc).__name__}: {exc}")
                    if attempt < MAX_RETRIES:
                        pending.remove(task)
                        pending.append(task)
                    else:
                        pending.remove(task)
                        failed.add(task.key)
                    continue
                pending.remove(task)
                running.append(Running(task, process, cards, attempt, log_path, log_handle))
                reserved.update(cards)
                launched = True
                # Re-read nvidia-smi on the next outer loop after each launch;
                # the reserved set prevents this process from double-booking.
                break

        now = time.time()
        if now - last_status >= STATUS_SECONDS:
            status_line(pending, running, done, failed)
            last_status = now
        if pending or running:
            time.sleep(POLL_SECONDS)

    status_line(pending, running, done, failed)
    log(f"QUEUE_FINISHED done={len(done)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
