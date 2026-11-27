#!/usr/bin/env python3
"""Streaming TIES-Merging for Hugging Face safetensors checkpoints.

The implementation follows the TIES recipe from
https://arxiv.org/abs/2306.01708:

1. build task vectors relative to a common base;
2. trim each task vector to the largest ``density`` fraction by magnitude;
3. elect a coordinate-wise sign by signed mass; and
4. take the disjoint mean of entries agreeing with the elected sign.

The model files are processed shard-by-shard and tensors are read through
``safe_open`` so that a complete 30B checkpoint never has to be materialized
in RAM.  A deterministic magnitude sample is used for the global trim
threshold; the actual merge mask is then applied to every tensor in a second
pass.  This is a practical streaming equivalent of the flattened-vector
operation in the reference implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
WEIGHT_SUFFIXES = (".safetensors",)
DEFAULT_DENSITY = 0.20
DEFAULT_SCALE = 1.0
DEFAULT_SAMPLE_PER_TENSOR = 4096


def _read_index(model_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model_dir / INDEX_NAME
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid weight_map in {index_path}")
        return {str(k): str(v) for k, v in weight_map.items()}, payload

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(
            f"{model_dir} has no index and does not contain exactly one "
            "safetensors file"
        )
    with safe_open(str(shards[0]), framework="pt", device="cpu") as handle:
        mapping = {name: shards[0].name for name in handle.keys()}
    return mapping, {"metadata": {"total_size": shards[0].stat().st_size}}


def _open_shard(path: Path):
    return safe_open(str(path), framework="pt", device="cpu")


def _shard_groups(weight_map: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        groups[shard].append(name)
    for names in groups.values():
        names.sort()
    return dict(groups)


def _valid_shard(path: Path, expected: set[str]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with _open_shard(path) as handle:
            return set(handle.keys()) == expected
    except Exception:
        return False


def _iter_base_names(base_map: dict[str, str]) -> Iterator[tuple[str, list[str]]]:
    for shard, names in sorted(_shard_groups(base_map).items()):
        yield shard, names


def _handles_for_shard(
    model_dir: Path,
    model_map: dict[str, str],
    names: list[str],
):
    """Open the minimum set of model shards needed for one base shard."""
    needed = sorted({model_map[name] for name in names if name in model_map})
    stack = contextlib.ExitStack()
    handles = {
        shard: stack.enter_context(_open_shard(model_dir / shard))
        for shard in needed
    }
    return stack, handles


def _sample_abs_values(tensor: torch.Tensor, limit: int) -> np.ndarray:
    values = tensor.detach().float().abs().reshape(-1)
    if values.numel() == 0:
        return np.empty(0, dtype=np.float32)
    if values.numel() > limit:
        # Evenly spaced indices make the sample independent of the Python hash
        # seed and preserve the tensor's element-count weighting.
        indices = torch.linspace(
            0,
            values.numel() - 1,
            steps=limit,
            dtype=torch.long,
        )
        values = values.index_select(0, indices)
    return values.cpu().numpy().astype(np.float32, copy=False)


def _magnitude_threshold(samples: list[np.ndarray], density: float) -> tuple[float, int]:
    if not samples:
        return math.inf, 0
    joined = np.concatenate(samples)
    if joined.size == 0 or not np.any(joined > 0):
        return math.inf, int(joined.size)
    quantile = float(np.quantile(joined, 1.0 - density, method="linear"))
    return quantile, int(joined.size)


def _trim(delta: torch.Tensor, threshold: float) -> torch.Tensor:
    if not math.isfinite(threshold):
        return torch.zeros_like(delta, dtype=torch.float32)
    delta = delta.float()
    return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))


def _ties_delta(
    deltas: list[torch.Tensor],
    thresholds: list[float],
) -> tuple[torch.Tensor, dict[str, int]]:
    trimmed = [_trim(delta, threshold) for delta, threshold in zip(deltas, thresholds)]
    if not trimmed:
        raise ValueError("TIES requires at least one task vector")

    stacked = torch.stack(trimmed, dim=0)
    active = stacked != 0
    signed_mass = stacked.sum(dim=0)
    elected = torch.sign(signed_mass)

    # The reference implementation resolves zero signs by a majority rule.
    # For an exact tie (common when an anchor model is included), use the first
    # non-zero task in the declared order so the result is deterministic.
    unresolved = elected == 0
    if unresolved.any():
        for row in stacked:
            candidate = torch.sign(row)
            fill = unresolved & (candidate != 0)
            elected = torch.where(fill, candidate, elected)
            unresolved = elected == 0
            if not unresolved.any():
                break

    agree = active & (torch.sign(stacked) == elected.unsqueeze(0))
    selected = torch.where(agree, stacked, torch.zeros_like(stacked))
    counts = (selected != 0).sum(dim=0).clamp_min(1).float()
    merged = selected.sum(dim=0) / counts
    stats = {
        "elements": int(merged.numel()),
        "trimmed_nonzero": int(active.sum().item()),
        "selected_nonzero": int((selected != 0).sum().item()),
        "unresolved_zero_signs": int(unresolved.sum().item()),
    }
    return merged, stats


def _copy_metadata(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if source.name == INDEX_NAME or source.suffix in WEIGHT_SUFFIXES:
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(source.name)
    config_path = output_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["torch_dtype"] = "bfloat16"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return sorted(copied)


def _write_index(output_dir: Path, base_map: dict[str, str]) -> None:
    shards = sorted(set(base_map.values()))
    total_size = sum((output_dir / shard).stat().st_size for shard in shards)
    payload = {
        "metadata": {"total_size": total_size},
        "weight_map": base_map,
    }
    (output_dir / INDEX_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_output(output_dir: Path, base_map: dict[str, str]) -> dict[str, Any]:
    index_path = output_dir / INDEX_NAME
    if not index_path.is_file():
        raise RuntimeError(f"missing output index: {index_path}")
    output_map, _ = _read_index(output_dir)
    if output_map != base_map:
        raise RuntimeError("output weight_map does not match base weight_map")
    by_shard = _shard_groups(base_map)
    for shard, names in by_shard.items():
        if not _valid_shard(output_dir / shard, set(names)):
            raise RuntimeError(f"invalid output shard: {output_dir / shard}")
    total_size = sum((output_dir / shard).stat().st_size for shard in by_shard)
    return {
        "shards": len(by_shard),
        "tensors": len(base_map),
        "bytes": total_size,
    }


def _collect_thresholds(
    base_dir: Path,
    base_map: dict[str, str],
    model_dirs: list[Path],
    model_maps: list[dict[str, str]],
    density: float,
    sample_per_tensor: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    samples: list[list[np.ndarray]] = [[] for _ in model_dirs]
    common_counts = [0 for _ in model_dirs]
    floating_counts = [0 for _ in model_dirs]
    started = time.time()
    for shard, names in _iter_base_names(base_map):
        base_stack, base_handles = _handles_for_shard(base_dir, base_map, names)
        model_stacks = []
        model_handles = []
        try:
            for model_dir, model_map in zip(model_dirs, model_maps):
                stack, handles = _handles_for_shard(model_dir, model_map, names)
                model_stacks.append(stack)
                model_handles.append(handles)
            for name in names:
                base_tensor = base_handles[base_map[name]].get_tensor(name)
                if not base_tensor.is_floating_point():
                    continue
                for index, (model_map, handles) in enumerate(zip(model_maps, model_handles)):
                    model_shard = model_map.get(name)
                    if model_shard is None:
                        continue
                    model_tensor = handles[model_shard].get_tensor(name)
                    if tuple(model_tensor.shape) != tuple(base_tensor.shape):
                        raise RuntimeError(
                            f"shape mismatch for {name}: {tuple(base_tensor.shape)} "
                            f"vs {tuple(model_tensor.shape)}"
                        )
                    if not model_tensor.is_floating_point():
                        continue
                    delta = model_tensor.float() - base_tensor.float()
                    samples[index].append(_sample_abs_values(delta, sample_per_tensor))
                    common_counts[index] += int(delta.numel())
                    floating_counts[index] += int(delta.numel())
                    del model_tensor, delta
                del base_tensor
        finally:
            for stack in model_stacks:
                stack.close()
            base_stack.close()
        gc.collect()
        print(
            f"[ties] threshold scan shard={shard} elapsed={(time.time() - started) / 60:.1f}m",
            flush=True,
        )

    thresholds: list[float] = []
    stats: list[dict[str, Any]] = []
    for index, arrays in enumerate(samples):
        threshold, sample_count = _magnitude_threshold(arrays, density)
        nonzero_sample = int(sum(int(np.count_nonzero(x)) for x in arrays))
        thresholds.append(threshold)
        stats.append(
            {
                "model": str(model_dirs[index]),
                "common_elements": common_counts[index],
                "floating_elements": floating_counts[index],
                "sample_elements": sample_count,
                "sample_nonzero": nonzero_sample,
                "trim_threshold_abs": threshold if math.isfinite(threshold) else None,
                "trim_density": density,
                "threshold_is_all_zero": not math.isfinite(threshold),
            }
        )
    return thresholds, stats


def merge(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base).expanduser().resolve()
    model_dirs = [Path(value).expanduser().resolve() for value in args.models]
    output_dir = Path(args.output).expanduser().resolve()
    if not base_dir.is_dir() or any(not path.is_dir() for path in model_dirs):
        raise FileNotFoundError("base and all model arguments must be directories")
    if not 0.0 < args.density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    base_map, base_index = _read_index(base_dir)
    model_maps = []
    model_indices = []
    for model_dir in model_dirs:
        model_map, model_index = _read_index(model_dir)
        model_maps.append(model_map)
        model_indices.append(model_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds, threshold_stats = _collect_thresholds(
        base_dir,
        base_map,
        model_dirs,
        model_maps,
        args.density,
        args.sample_per_tensor,
    )
    print(
        "[ties] thresholds="
        + json.dumps(
            [None if not math.isfinite(x) else x for x in thresholds],
            ensure_ascii=False,
        ),
        flush=True,
    )

    base_groups = _shard_groups(base_map)
    shard_stats: list[dict[str, Any]] = []
    total_stats = {
        "common_floating_merged": 0,
        "base_only_copied": 0,
        "nonfloating_copied": 0,
        "trimmed_nonzero": 0,
        "selected_nonzero": 0,
        "unresolved_zero_signs": 0,
    }
    started = time.time()
    for shard, names in sorted(base_groups.items()):
        output_path = output_dir / shard
        expected = set(names)
        if args.resume and _valid_shard(output_path, expected):
            print(f"[ties] reuse shard={shard} tensors={len(names)}", flush=True)
            shard_stats.append({"shard": shard, "tensors": len(names), "reused": True})
            continue

        base_stack, base_handles = _handles_for_shard(base_dir, base_map, names)
        model_stacks = []
        model_handles = []
        output_state: dict[str, torch.Tensor] = {}
        local = {
            "common_floating_merged": 0,
            "base_only_copied": 0,
            "nonfloating_copied": 0,
            "trimmed_nonzero": 0,
            "selected_nonzero": 0,
            "unresolved_zero_signs": 0,
        }
        try:
            for model_dir, model_map in zip(model_dirs, model_maps):
                stack, handles = _handles_for_shard(model_dir, model_map, names)
                model_stacks.append(stack)
                model_handles.append(handles)
            for name in names:
                base_tensor = base_handles[base_map[name]].get_tensor(name)
                if not base_tensor.is_floating_point():
                    output_state[name] = base_tensor
                    local["nonfloating_copied"] += 1
                    continue

                deltas: list[torch.Tensor] = []
                present = True
                for model_map, handles in zip(model_maps, model_handles):
                    model_shard = model_map.get(name)
                    if model_shard is None:
                        present = False
                        break
                    model_tensor = handles[model_shard].get_tensor(name)
                    if tuple(model_tensor.shape) != tuple(base_tensor.shape):
                        raise RuntimeError(
                            f"shape mismatch for {name}: {tuple(base_tensor.shape)} "
                            f"vs {tuple(model_tensor.shape)}"
                        )
                    if not model_tensor.is_floating_point():
                        present = False
                        break
                    deltas.append(model_tensor.float() - base_tensor.float())
                    del model_tensor

                if not present:
                    output_state[name] = base_tensor
                    local["base_only_copied"] += 1
                    for delta in deltas:
                        del delta
                    continue

                merged_delta, coord_stats = _ties_delta(deltas, thresholds)
                merged = base_tensor.float() + float(args.scale) * merged_delta
                output_state[name] = merged.to(dtype=base_tensor.dtype).contiguous()
                local["common_floating_merged"] += 1
                for key in ("trimmed_nonzero", "selected_nonzero", "unresolved_zero_signs"):
                    local[key] += coord_stats[key]
                del deltas, merged_delta, merged, base_tensor
        finally:
            for stack in model_stacks:
                stack.close()
            base_stack.close()

        temp_path = output_dir / (shard + ".part")
        if temp_path.exists():
            temp_path.unlink()
        save_file(output_state, str(temp_path), metadata={"format": "pt"})
        os.replace(temp_path, output_path)
        del output_state
        gc.collect()
        for key in total_stats:
            total_stats[key] += local[key]
        shard_stats.append({"shard": shard, "tensors": len(names), "reused": False, **local})
        print(
            f"[ties] saved shard={shard} tensors={len(names)} "
            f"merged={local['common_floating_merged']} "
            f"elapsed={(time.time() - started) / 60:.1f}m",
            flush=True,
        )

    metadata_source = Path(args.metadata_source or model_dirs[0]).expanduser().resolve()
    copied_metadata = _copy_metadata(metadata_source, output_dir)
    _write_index(output_dir, base_map)
    validation = _validate_output(output_dir, base_map)
    summary = {
        "schema_version": 1,
        "method": "TIES-Merging",
        "paper": "https://arxiv.org/abs/2306.01708",
        "trim": "topk20",
        "density": args.density,
        "sign_election": "mass",
        "aggregation": "dis-mean",
        "scale": args.scale,
        "formula": "base + scale * disjoint_mean(trim(model_i - base))",
        "anchor_role": args.anchor_label,
        "base_model": str(base_dir),
        "task_vector_definition": args.task_vector_label,
        "task_models": [str(path) for path in model_dirs],
        "metadata_source": str(metadata_source),
        "output_model": str(output_dir),
        "base_tensors": len(base_map),
        "model_tensor_counts": [len(mapping) for mapping in model_maps],
        "base_only_tensors": sorted(set(base_map) - set.intersection(*(set(x) for x in model_maps)))[:100],
        "task_only_tensor_counts": [len(set(mapping) - set(base_map)) for mapping in model_maps],
        "threshold_sample_per_tensor": args.sample_per_tensor,
        "thresholds": threshold_stats,
        "shards": shard_stats,
        "totals": total_stats,
        "copied_metadata": copied_metadata,
        "validation": validation,
        "elapsed_seconds": time.time() - started,
    }
    stats_path = output_dir / "ties_merge_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Explicit anchor checkpoint used as the task-vector reference",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Fine-tuned/task checkpoints, in deterministic merge order",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--metadata-source", help="Checkpoint supplying tokenizer/config files")
    parser.add_argument(
        "--anchor-label",
        default="explicit anchor",
        help="Human-readable role recorded in ties_merge_stats.json",
    )
    parser.add_argument(
        "--task-vector-label",
        default="model_i - anchor",
        help="Human-readable task-vector definition recorded in the stats",
    )
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--sample-per-tensor", type=int, default=DEFAULT_SAMPLE_PER_TENSOR)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.sample_per_tensor <= 0:
        raise SystemExit("--sample-per-tensor must be positive")
    try:
        merge(args)
    except Exception as exc:
        print(f"[ties] ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
