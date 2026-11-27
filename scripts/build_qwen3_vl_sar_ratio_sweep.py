#!/usr/bin/env python3
"""Build a Qwen3-VL Instruct/Thinking no-truncation SAR ratio sweep.

For every target linear weight this constructs

    W_ratio = P_S(W_instruct) + (I - P_S)(W_thinking),

where ``S`` is the leading-rank subspace of the Instruct weight and
``rank(S) = ceil(ratio * min(W.shape))``.  The same SVD is reused for all
requested ratios.  Vision attention/MLP matrices and language-model
attention/MLP matrices are selected by ``model.subspace_swap.is_target``;
embeddings, norms, routers and other non-target tensors come from Instruct.

The builder is shard-resumable.  A checkpoint receives a completion marker
only after all of its shards, index and metadata have been validated.
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

from model.subspace_swap import build_tensor_map, is_target  # noqa: E402


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def valid_safetensor(path: Path, expected_names: set[str] | None = None) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = set(handle.keys())
        return expected_names is None or names == expected_names
    except (OSError, RuntimeError, ValueError):
        return False


def copy_metadata(source_dir: Path, output_dir: Path) -> None:
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix == ".safetensors" or source.name == "model.safetensors.index.json":
            continue
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)


def read_index(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid weight_map in {path}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


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
    config["torch_dtype"] = str(source_dtype).removeprefix("torch.")
    atomic_json(config_path, config)


def model_complete(
    output_dir: Path,
    weight_map: dict[str, str],
    expected_ratio: float,
    base_dir: Path,
    thinking_dir: Path,
) -> bool:
    marker_path = output_dir / "sar_build_manifest.json"
    index_path = output_dir / "model.safetensors.index.json"
    config_path = output_dir / "config.json"
    if not marker_path.is_file() or not index_path.is_file() or not config_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        output_map = read_index(index_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict):
        return False
    if (
        marker.get("status") != "complete"
        or _marker_ratio(marker) != expected_ratio
        or marker.get("base") != str(base_dir)
        or marker.get("thinking") != str(thinking_dir)
        or output_map != weight_map
    ):
        return False
    by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    return all(valid_safetensor(output_dir / shard, names) for shard, names in by_shard.items())


def _marker_ratio(marker: dict[str, Any]) -> float | None:
    try:
        return float(marker["ratio"])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Qwen3-VL Instruct checkpoint")
    parser.add_argument("--thinking", type=Path, required=True, help="Qwen3-VL Thinking checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="directory prefix, e.g. Qwen3-VL-2B-sar-ratio",
    )
    parser.add_argument("--ratios", type=parse_ratios, default=list(DEFAULT_RATIOS))
    parser.add_argument("--device", default="cpu", help="SVD device; CPU leaves GPUs free")
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

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    base_map = dict(build_tensor_map(str(base_dir))[0])
    thinking_map = dict(build_tensor_map(str(thinking_dir))[0])
    if set(base_map) != set(thinking_map):
        missing = sorted(set(base_map) ^ set(thinking_map))
        raise RuntimeError(f"base/thinking tensor maps differ; examples: {missing[:5]}")

    shard_groups: dict[str, list[str]] = {}
    for name, shard in sorted(base_map.items()):
        shard_groups.setdefault(shard, []).append(name)
    shard_names = sorted(shard_groups)
    target_count = sum(1 for name in base_map if is_target(name))
    output_dirs = {
        ratio: output_root / f"{args.output_prefix}-{ratio_label(ratio)}"
        for ratio in args.ratios
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for output_dir in output_dirs.values():
        if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
            raise FileExistsError(f"output already exists: {output_dir}; use --resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_metadata(thinking_dir, output_dir)

    complete = [
        ratio
        for ratio, directory in output_dirs.items()
        if model_complete(directory, base_map, ratio, base_dir, thinking_dir)
    ]
    if len(complete) == len(output_dirs):
        print("all requested VL SAR checkpoints are already complete", flush=True)
        return 0

    started = time.time()
    print(
        json.dumps(
            {
                "base": str(base_dir),
                "thinking": str(thinking_dir),
                "output_root": str(output_root),
                "output_prefix": args.output_prefix,
                "ratios": args.ratios,
                "device": args.device,
                "threads": args.threads,
                "shards": shard_names,
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
            if not valid_safetensor(
                output_dir / shard,
                {name for name in shard_groups[shard]},
            )
        ]
        if not needed:
            print(f"resume skip shard={shard} ({shard_index}/{len(shard_names)})", flush=True)
            continue

        shard_started = time.time()
        base_state = load_file(str(base_dir / shard), device="cpu")
        thinking_state = load_file(str(thinking_dir / thinking_map[next(iter(shard_groups[shard]))]), device="cpu")
        output_states: dict[float, dict[str, torch.Tensor]] = {ratio: {} for ratio in needed}

        for name in shard_groups[shard]:
            base_weight = base_state[name]
            if source_dtype is None and base_weight.is_floating_point():
                source_dtype = base_weight.dtype
            if is_target(name):
                thinking_shard = thinking_map[name]
                if thinking_shard != shard:
                    # The released VL checkpoints use aligned shard names, but
                    # this keeps the builder correct if a future checkpoint
                    # maps a tensor to a different shard.
                    thinking_state_for_name = load_file(
                        str(thinking_dir / thinking_shard), device="cpu"
                    )[name]
                else:
                    thinking_state_for_name = thinking_state[name]
                if thinking_state_for_name.shape != base_weight.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: {base_weight.shape} vs "
                        f"{thinking_state_for_name.shape}"
                    )
                base32 = base_weight.to(device=args.device, dtype=torch.float32)
                thinking32 = thinking_state_for_name.to(device=args.device, dtype=torch.float32)
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
                    output_dtype = base_weight.dtype if base_weight.is_floating_point() else torch.bfloat16
                    output_states[ratio][name] = mixed.to(
                        device="cpu", dtype=output_dtype
                    ).contiguous()
                processed_targets += 1
                if processed_targets % 25 == 0:
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
                f"tensors={len(output_states[ratio])} "
                f"elapsed={time.time() - shard_started:.0f}s",
                flush=True,
            )
        del base_state, thinking_state, output_states

    if source_dtype is None:
        source_dtype = torch.bfloat16
    for ratio, output_dir in output_dirs.items():
        by_shard = {
            shard: {name for name in names}
            for shard, names in shard_groups.items()
        }
        if not all(valid_safetensor(output_dir / shard, names) for shard, names in by_shard.items()):
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
                "note": "VL vision and language attention/MLP targets; other tensors use Instruct.",
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
        f"completed prefix={args.output_prefix} "
        f"ratios={','.join(ratio_label(ratio) for ratio in args.ratios)} "
        f"projected_matrices={target_count} elapsed={time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
