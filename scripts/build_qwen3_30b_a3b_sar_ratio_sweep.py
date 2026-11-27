#!/usr/bin/env python3
"""Build the five Qwen3-30B-A3B SAR ratio checkpoints.

The construction matches the existing Qwen3-4B ratio sweep:

    W_ratio = P_S(W_instruct) + (I - P_S)(W_thinking)

where S is the leading-rank subspace of the Instruct matrix and
``rank(S) = ceil(ratio * min(W.shape))``.  The SVD is done once per target
matrix and reused for all five ratios.  This is the no-truncation
subspace-swap form used by the existing ratio=1 Qwen3 checkpoints.

The builder is resumable at shard granularity.  It writes each shard through
a temporary file and only writes a completion marker after every ratio has a
valid complete checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RATIOS = (0.9, 0.8, 0.7, 0.6, 0.5)


def parse_ratios(value: str) -> list[float]:
    ratios: list[float] = []
    for raw in value.split(","):
        ratio = float(raw.strip())
        if not 0.0 < ratio <= 1.0:
            raise argparse.ArgumentTypeError(f"ratio must be in (0, 1], got {ratio}")
        if ratio not in ratios:
            ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return ratios


def ratio_label(ratio: float) -> str:
    return f"{ratio:.1f}" if abs(ratio * 10 - round(ratio * 10)) < 1e-9 else f"{ratio:g}"


def load_weight_map(model_dir: Path) -> dict[str, str]:
    # Import lazily so --help and syntax checks do not load the model.
    from model.subspace_swap import build_tensor_map

    mapping, _ = build_tensor_map(str(model_dir))
    return mapping


def is_target(name: str) -> bool:
    from model.subspace_swap import is_target as sar_is_target

    return sar_is_target(name)


def valid_safetensor(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return bool(handle.keys())
    except (OSError, RuntimeError, ValueError):
        return False


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_metadata(source_dir: Path, output_dir: Path) -> None:
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix == ".safetensors" or source.name == "model.safetensors.index.json":
            continue
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)


def model_complete(output_dir: Path, shard_names: list[str]) -> bool:
    marker_path = output_dir / "sar_build_manifest.json"
    index_path = output_dir / "model.safetensors.index.json"
    if not marker_path.is_file() or not index_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if marker.get("status") != "complete":
        return False
    return all(valid_safetensor(output_dir / shard) for shard in shard_names)


def write_index_and_metadata(
    output_dir: Path,
    weight_map: dict[str, str],
    source_dtype: torch.dtype,
) -> None:
    shards = sorted(set(weight_map.values()))
    total_size = sum((output_dir / shard).stat().st_size for shard in shards)
    atomic_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dtype_name = str(source_dtype).removeprefix("torch.")
    config["torch_dtype"] = dtype_name
    atomic_json(config_path, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Qwen3 Instruct checkpoint")
    parser.add_argument("--thinking", type=Path, required=True, help="Qwen3 Thinking checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=list(DEFAULT_RATIOS),
        help="comma-separated retained-rank ratios",
    )
    parser.add_argument("--device", default="cpu", help="SVD device; CPU keeps all GPUs free")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, int(os.environ.get("SAR_BUILD_THREADS", "32"))),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_dir = args.base.expanduser().resolve()
    thinking_dir = args.thinking.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for model_dir in (base_dir, thinking_dir):
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json: {model_dir}")
        if not (model_dir / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"missing model index: {model_dir}")

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Some embedding environments initialize the inter-op pool before
            # importing this script.  The intra-op setting is still useful.
            pass

    base_map = load_weight_map(base_dir)
    thinking_map = load_weight_map(thinking_dir)
    # Omni's Instruct checkpoint contains the talker/code2wav tensors while
    # Thinking is thinker-only.  Keep the Instruct output layout and align
    # common tensors by name; only common target matrices participate in the
    # subspace swap, while Instruct-only tensors remain unchanged.
    thinking_only = set(thinking_map) - set(base_map)
    if thinking_only:
        examples = sorted(thinking_only)[:5]
        raise RuntimeError(
            "Thinking contains tensors absent from the Instruct output layout; "
            f"examples: {examples}"
        )

    shard_groups: dict[str, list[str]] = {}
    for name, shard in sorted(base_map.items()):
        shard_groups.setdefault(shard, []).append(name)
    shard_names = sorted(shard_groups)
    target_count = sum(
        1 for name in base_map if name in thinking_map and is_target(name)
    )
    output_dirs = {
        ratio: output_root / f"qwen3-30b-a3b-sar-ratio-{ratio_label(ratio)}"
        for ratio in args.ratios
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for output_dir in output_dirs.values():
        if output_dir.exists() and not args.resume:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"output already exists: {output_dir}; use --resume to continue"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_metadata(thinking_dir, output_dir)

    complete = [ratio for ratio, directory in output_dirs.items() if model_complete(directory, shard_names)]
    if len(complete) == len(output_dirs):
        print("all requested checkpoints are already complete", flush=True)
        return 0

    started = time.time()
    print(
        json.dumps(
            {
                "base": str(base_dir),
                "thinking": str(thinking_dir),
                "output_root": str(output_root),
                "ratios": args.ratios,
                "device": args.device,
                "threads": args.threads,
                "shards": len(shard_names),
                "target_matrices": target_count,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    source_dtype: torch.dtype | None = None
    processed_targets = 0
    for shard_index, shard in enumerate(shard_names, start=1):
        needed = [
            ratio
            for ratio, output_dir in output_dirs.items()
            if not valid_safetensor(output_dir / shard)
        ]
        if not needed:
            print(f"resume skip shard={shard} ({shard_index}/{len(shard_names)})", flush=True)
            continue

        shard_started = time.time()
        base_state = load_file(str(base_dir / shard), device="cpu")
        thinking_cache: dict[str, dict[str, torch.Tensor]] = {}
        output_states: dict[float, dict[str, torch.Tensor]] = {ratio: {} for ratio in needed}

        for name in shard_groups[shard]:
            base_weight = base_state[name]
            if source_dtype is None and base_weight.is_floating_point():
                source_dtype = base_weight.dtype
            thinking_shard = thinking_map.get(name)
            if is_target(name) and thinking_shard is not None:
                if thinking_shard not in thinking_cache:
                    thinking_cache[thinking_shard] = load_file(
                        str(thinking_dir / thinking_shard), device="cpu"
                    )
                thinking_weight = thinking_cache[thinking_shard][name]
                if thinking_weight.shape != base_weight.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: {base_weight.shape} vs {thinking_weight.shape}"
                    )
                base32 = base_weight.to(device=args.device, dtype=torch.float32)
                thinking32 = thinking_weight.to(device=args.device, dtype=torch.float32)
                u, _, vh = torch.linalg.svd(base32, full_matrices=False)
                full_rank = min(base32.shape)
                for ratio in needed:
                    k = max(1, math.ceil(ratio * full_rank))
                    uk = u[:, :k]
                    vk = vh[:k, :].transpose(0, 1)
                    base_coeff = uk.transpose(0, 1) @ base32 @ vk
                    thinking_coeff = uk.transpose(0, 1) @ thinking32 @ vk
                    base_proj = uk @ base_coeff @ vk.transpose(0, 1)
                    thinking_proj = uk @ thinking_coeff @ vk.transpose(0, 1)
                    mixed = thinking32 - thinking_proj + base_proj
                    output_dtype = base_weight.dtype if base_weight.is_floating_point() else torch.float16
                    output_states[ratio][name] = mixed.to(
                        device="cpu", dtype=output_dtype
                    ).contiguous()
                processed_targets += 1
                if processed_targets % 50 == 0:
                    print(
                        f"processed_targets={processed_targets}/{target_count} "
                        f"shard={shard} tensor={name} elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )
                del base32, thinking32, u, vh
            else:
                for ratio in needed:
                    output_states[ratio][name] = base_weight

        for ratio in needed:
            output_dir = output_dirs[ratio]
            temporary = output_dir / f".{shard}.tmp-{os.getpid()}"
            if temporary.exists():
                temporary.unlink()
            save_file(output_states[ratio], str(temporary), metadata={"format": "pt"})
            os.replace(temporary, output_dir / shard)
            print(
                f"saved ratio={ratio_label(ratio)} shard={shard} "
                f"tensors={len(output_states[ratio])} elapsed={time.time() - shard_started:.0f}s",
                flush=True,
            )
        del base_state, thinking_cache, output_states

    if source_dtype is None:
        source_dtype = torch.bfloat16
    for ratio, output_dir in output_dirs.items():
        if not all(valid_safetensor(output_dir / shard) for shard in shard_names):
            raise RuntimeError(f"missing or invalid shard after build: {output_dir}")
        write_index_and_metadata(output_dir, base_map, source_dtype)
        atomic_json(
            output_dir / "subspace_swap_stats.json",
            {
                "method": "P_base_subspace + P_perp_thinking",
                "ratio": ratio,
                "truncate": False,
                "base": str(base_dir),
                "thinking": str(thinking_dir),
                "projected_matrices": target_count,
                "shards": len(shard_names),
                "note": "Weights use the Instruct leading subspace and Thinking complement.",
            },
        )
        atomic_json(
            output_dir / "sar_build_manifest.json",
            {
                "status": "complete",
                "schema_version": 1,
                "method": "P_base_subspace + P_perp_thinking",
                "ratio": ratio,
                "truncate": False,
                "base": str(base_dir),
                "thinking": str(thinking_dir),
                "projected_matrices": target_count,
                "shards": len(shard_names),
                "elapsed_s": time.time() - started,
            },
        )

    print(
        f"completed ratios={','.join(ratio_label(ratio) for ratio in args.ratios)} "
        f"projected_matrices={target_count} elapsed={time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
