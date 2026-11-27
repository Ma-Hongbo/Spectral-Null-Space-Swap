#!/usr/bin/env python3
"""Build several Qwen3-4B no-truncation SAR checkpoints in one pass.

The existing ``sar_ratio_1`` checkpoints use the subspace-swap definition

    W = P_S(W_instruct) + (I - P_S)(W_thinking),

where S is the tensor-product subspace from the leading singular vectors of
the Instruct weight.  This builder keeps that definition and reuses the SVD
of each Instruct matrix for every requested ratio, so a rank sweep does not
repeat the expensive SVD five times.
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
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.subspace_swap import build_tensor_map, is_target  # noqa: E402


def parse_ratios(value: str) -> list[float]:
    ratios: list[float] = []
    for raw in value.split(","):
        ratio = float(raw.strip())
        if not 0.0 < ratio <= 1.0:
            raise argparse.ArgumentTypeError(
                f"ratio must be in (0, 1], got {ratio}"
            )
        if ratio not in ratios:
            ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return ratios


def ratio_label(ratio: float) -> str:
    return f"{ratio:.1f}" if abs(ratio * 10 - round(ratio * 10)) < 1e-9 else f"{ratio:g}"


def load_weight_map(model_dir: Path) -> dict[str, str]:
    mapping, _ = build_tensor_map(str(model_dir))
    return mapping


def copy_metadata(thinking_dir: Path, output_dir: Path) -> None:
    for filename in thinking_dir.iterdir():
        if filename.suffix in {".json", ".txt", ".jinja"} and filename.name != "model.safetensors.index.json":
            shutil.copy2(filename, output_dir / filename.name)


def projected_weight(
    base_weight: torch.Tensor,
    thinking_weight: torch.Tensor,
    ratio: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base = base_weight.to(device=device, dtype=torch.float32)
    thinking = thinking_weight.to(device=device, dtype=torch.float32)
    full_rank = min(base.shape)
    k = max(1, math.ceil(ratio * full_rank))

    # This is the same no-truncation subspace swap used by the current
    # ratio=1 checkpoint.  The expensive SVD is intentionally outside this
    # function and is shared by all ratios for one matrix.
    u, _, vh = torch.linalg.svd(base, full_matrices=False)
    uk = u[:, :k]
    vk = vh[:k, :].transpose(0, 1)
    base_coeff = uk.transpose(0, 1) @ base @ vk
    thinking_coeff = uk.transpose(0, 1) @ thinking @ vk
    base_proj = uk @ base_coeff @ vk.transpose(0, 1)
    thinking_proj = uk @ thinking_coeff @ vk.transpose(0, 1)
    mixed = thinking - thinking_proj + base_proj

    subspace_error = (uk.transpose(0, 1) @ mixed @ vk - base_coeff).norm()
    complement_error = (
        (mixed - uk @ (uk.transpose(0, 1) @ mixed @ vk) @ vk.transpose(0, 1))
        - (thinking - thinking_proj)
    ).norm()
    stats = {
        "rank": k,
        "full_rank": full_rank,
        "base_subspace_norm": float(base_proj.norm()),
        "thinking_subspace_norm": float(thinking_proj.norm()),
        "subspace_error": float(subspace_error),
        "complement_error": float(complement_error),
    }
    return mixed.to(device="cpu", dtype=torch.float16).contiguous(), stats


def write_index(output_dir: Path, weight_map: dict[str, str]) -> None:
    total_size = sum((output_dir / shard).stat().st_size for shard in set(weight_map.values()))
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["torch_dtype"] = "float16"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--thinking", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=parse_ratios("0.9,0.8,0.7,0.6,0.5"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = args.base.expanduser().resolve()
    thinking_dir = args.thinking.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for model_dir in (base_dir, thinking_dir):
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json: {model_dir}")

    base_map = load_weight_map(base_dir)
    thinking_map = load_weight_map(thinking_dir)
    if set(base_map) != set(thinking_map):
        missing = sorted(set(base_map) ^ set(thinking_map))
        raise RuntimeError(f"base/thinking tensor maps differ; examples: {missing[:5]}")

    shards: dict[str, list[str]] = {}
    for name, shard in sorted(base_map.items()):
        shards.setdefault(shard, []).append(name)

    output_dirs = {
        ratio: output_root / f"qwen3-4b-sar-ratio-{ratio_label(ratio)}"
        for ratio in args.ratios
    }
    for output_dir in output_dirs.values():
        if output_dir.exists() and not args.overwrite:
            existing = list(output_dir.glob("model-*.safetensors"))
            if existing or (output_dir / "config.json").exists():
                raise FileExistsError(
                    f"output already exists: {output_dir}; use --overwrite to replace it"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_metadata(thinking_dir, output_dir)

    # Keep at most one Thinking shard in RAM, matching the established builder.
    thinking_cache: dict[str, dict[str, torch.Tensor]] = {}

    def get_thinking_tensor(name: str) -> torch.Tensor:
        shard = thinking_map[name]
        if shard not in thinking_cache:
            thinking_cache.clear()
            thinking_cache[shard] = load_file(str(thinking_dir / shard), device="cpu")
        return thinking_cache[shard][name]

    per_ratio_stats: dict[float, dict[str, dict[str, Any]]] = {
        ratio: {} for ratio in args.ratios
    }
    projected_count = 0
    started = time.time()
    print(
        json.dumps(
            {
                "base": str(base_dir),
                "thinking": str(thinking_dir),
                "output_root": str(output_root),
                "ratios": args.ratios,
                "device": args.device,
                "shards": sorted(shards),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for shard in sorted(shards):
        shard_started = time.time()
        base_state = load_file(str(base_dir / shard), device="cpu")
        output_states: dict[float, dict[str, torch.Tensor]] = {
            ratio: {} for ratio in args.ratios
        }
        for name in shards[shard]:
            base_weight = base_state[name]
            if is_target(name):
                thinking_weight = get_thinking_tensor(name)
                if thinking_weight.shape != base_weight.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: {base_weight.shape} vs {thinking_weight.shape}"
                    )
                base = base_weight.to(device=args.device, dtype=torch.float32)
                thinking = thinking_weight.to(device=args.device, dtype=torch.float32)
                full_rank = min(base.shape)
                u, _, vh = torch.linalg.svd(base, full_matrices=False)
                for ratio in args.ratios:
                    k = max(1, math.ceil(ratio * full_rank))
                    uk = u[:, :k]
                    vk = vh[:k, :].transpose(0, 1)
                    base_coeff = uk.transpose(0, 1) @ base @ vk
                    thinking_coeff = uk.transpose(0, 1) @ thinking @ vk
                    base_proj = uk @ base_coeff @ vk.transpose(0, 1)
                    thinking_proj = uk @ thinking_coeff @ vk.transpose(0, 1)
                    mixed = thinking - thinking_proj + base_proj
                    subspace_error = (uk.transpose(0, 1) @ mixed @ vk - base_coeff).norm()
                    complement_error = (
                        (mixed - uk @ (uk.transpose(0, 1) @ mixed @ vk) @ vk.transpose(0, 1))
                        - (thinking - thinking_proj)
                    ).norm()
                    output_states[ratio][name] = mixed.to(
                        device="cpu", dtype=torch.float16
                    ).contiguous()
                    per_ratio_stats[ratio][name] = {
                        "rank": k,
                        "full_rank": full_rank,
                        "base_subspace_norm": float(base_proj.norm()),
                        "thinking_subspace_norm": float(thinking_proj.norm()),
                        "subspace_error": float(subspace_error),
                        "complement_error": float(complement_error),
                    }
                projected_count += 1
                if projected_count % 50 == 0:
                    print(
                        f"[{projected_count}] {name} full_rank={full_rank} "
                        f"elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )
            else:
                value = base_weight.to(torch.float16) if base_weight.is_floating_point() else base_weight
                for ratio in args.ratios:
                    output_states[ratio][name] = value

        for ratio, output_dir in output_dirs.items():
            save_file(
                output_states[ratio],
                str(output_dir / shard),
                metadata={"format": "pt"},
            )
            print(
                f"saved ratio={ratio_label(ratio)} shard={shard} "
                f"tensors={len(output_states[ratio])} "
                f"elapsed={time.time() - shard_started:.0f}s",
                flush=True,
            )
        del base_state, output_states
        thinking_cache.clear()

    for ratio, output_dir in output_dirs.items():
        write_index(output_dir, base_map)
        stats = per_ratio_stats[ratio]
        summary = {
            "method": "P_base_subspace + P_perp_thinking",
            "ratio": ratio,
            "truncate": False,
            "projected_matrices": len(stats),
            "max_subspace_error": max((item["subspace_error"] for item in stats.values()), default=0.0),
            "max_complement_error": max((item["complement_error"] for item in stats.values()), default=0.0),
            "elapsed_s": time.time() - started,
        }
        (output_dir / "subspace_swap_stats.json").write_text(
            json.dumps({"summary": summary, "per_weight": stats}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    print(
        f"completed ratios={','.join(ratio_label(ratio) for ratio in args.ratios)} "
        f"projected_matrices={projected_count} elapsed={time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
