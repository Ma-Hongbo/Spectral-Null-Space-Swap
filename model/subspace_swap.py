"""Construct a no-truncation spectral subspace swap checkpoint.

For every SAR target linear weight, let S be the tensor-product subspace
spanned by the leading singular vectors of the base weight W_base.  Given the
corresponding Thinking weight W_thinking, this script writes

    W_mix = P_S(W_base) + (I - P_S)(W_thinking).

Thus P_S(W_mix) = P_S(W_base), while its Frobenius-orthogonal complement is
identical to that of W_thinking.  The rank ratio only defines S; neither the
Thinking matrix nor its update is low-rank truncated.
"""

import argparse
import glob
import json
import math
import os
import shutil
import time

import torch
from safetensors.torch import load_file, save_file


ATTENTION_TARGET_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)

MLP_TARGET_SUFFIXES = (
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

VISION_TARGET_SUFFIXES = (
    "attn.qkv.weight",
    "attn.proj.weight",
    "linear_fc1.weight",
    "linear_fc2.weight",
)

TARGET_SUFFIXES = ATTENTION_TARGET_SUFFIXES + MLP_TARGET_SUFFIXES


def is_target(name: str) -> bool:
    if name.endswith(ATTENTION_TARGET_SUFFIXES):
        return True
    if ".visual." in name and name.endswith(VISION_TARGET_SUFFIXES):
        return True
    return ".mlp." in name and name.endswith((
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    ))


def load_index(model_dir: str) -> dict[str, str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            return json.load(handle)["weight_map"]
    return {"__single__": "model.safetensors"}


def build_tensor_map(model_dir: str) -> tuple[dict[str, str], None]:
    """Return the safetensors shard containing each checkpoint tensor."""
    weight_map = load_index(model_dir)
    if "__single__" in weight_map:
        state = load_file(os.path.join(model_dir, "model.safetensors"))
        return {name: "model.safetensors" for name in state}, None
    return weight_map, None


def tensor_map(model_dir: str) -> dict[str, str]:
    """Return the safetensors file holding each tensor.

    ``build_tensor_map`` expands a single ``model.safetensors`` file as well
    as a sharded Hugging Face index.  This matters for Qwen3-VL-2B, whose
    checkpoint is intentionally distributed as one safetensors file.
    """
    mapping, _ = build_tensor_map(model_dir)
    return mapping


def swap_weight(
    w_base: torch.Tensor,
    w_thinking: torch.Tensor,
    ratio: float,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Return P_S(W_base) + (I-P_S)(W_thinking), computed in FP32."""
    base = w_base.to(device=device, dtype=torch.float32)
    thinking = w_thinking.to(device=device, dtype=torch.float32)
    full_rank = min(base.shape)
    k = max(1, math.ceil(ratio * full_rank))

    u, _, vh = torch.linalg.svd(base, full_matrices=False)
    uk = u[:, :k]
    vk = vh[:k, :].transpose(0, 1)

    # Orthogonal Frobenius projections onto span(U_k) \u2297 span(V_k).
    base_coeff = uk.transpose(0, 1) @ base @ vk
    thinking_coeff = uk.transpose(0, 1) @ thinking @ vk
    base_proj = uk @ base_coeff @ vk.transpose(0, 1)
    thinking_proj = uk @ thinking_coeff @ vk.transpose(0, 1)
    mixed = thinking - thinking_proj + base_proj

    # Numerical checks make the intended decomposition auditable.
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
    return mixed.to("cpu", torch.float16), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Qwen3-4B checkpoint")
    parser.add_argument("--thinking", required=True, help="Qwen3-4B-Thinking-2507 checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=0.10,
                        help="fraction of each base weight's singular directions defining the subspace")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-shard", action="append", default=None,
                        help="build only this safetensors shard (repeatable); use with --skip-finalize")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="write selected weight shards and worker stats only")
    parser.add_argument("--finalize-only", action="store_true",
                        help="write index/config and merge worker stats after parallel shard builds")
    args = parser.parse_args()

    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be in (0, 1]")
    os.makedirs(args.out, exist_ok=True)

    base_map = tensor_map(args.base)
    thinking_map = tensor_map(args.thinking)
    shards: dict[str, list[str]] = {}
    for name, shard in base_map.items():
        shards.setdefault(shard, []).append(name)

    if args.finalize_only and (args.only_shard or args.skip_finalize):
        parser.error("--finalize-only cannot be combined with --only-shard or --skip-finalize")
    selected_shards = sorted(shards)
    if args.only_shard:
        unknown = set(args.only_shard) - set(shards)
        if unknown:
            parser.error(f"unknown safetensors shard(s): {sorted(unknown)}")
        selected_shards = sorted(set(args.only_shard))

    thinking_cache: dict[str, dict[str, torch.Tensor]] = {}

    def thinking_tensor(name: str) -> torch.Tensor:
        shard = thinking_map[name]
        if shard not in thinking_cache:
            thinking_cache.clear()
            thinking_cache[shard] = load_file(os.path.join(args.thinking, shard))
        return thinking_cache[shard][name]

    per_weight: dict[str, dict] = {}
    projected = 0
    started = time.time()
    if not args.finalize_only:
        for shard in selected_shards:
            names = shards[shard]
            base_state = load_file(os.path.join(args.base, shard))
            output_state: dict[str, torch.Tensor] = {}
            for name in names:
                base_weight = base_state[name]
                if is_target(name) and name in thinking_map:
                    thinking_weight = thinking_tensor(name)
                    if thinking_weight.shape != base_weight.shape:
                        raise ValueError(f"shape mismatch for {name}: {base_weight.shape} vs {thinking_weight.shape}")
                    output_state[name], per_weight[name] = swap_weight(
                        base_weight, thinking_weight, args.ratio, args.device
                    )
                    projected += 1
                    if projected % 50 == 0:
                        print(f"[{projected}] {name} k={per_weight[name]['rank']}/"
                              f"{per_weight[name]['full_rank']} ({time.time() - started:.0f}s)", flush=True)
                else:
                    # Keep every non-target tensor (embeddings, norms, lm_head, etc.)
                    # from the base checkpoint, matching build_perponly.py.
                    output_state[name] = base_weight.to(torch.float16) if base_weight.is_floating_point() else base_weight
            save_file(output_state, os.path.join(args.out, shard), metadata={"format": "pt"})
            print(f"saved {shard} ({len(output_state)} tensors)", flush=True)

        if args.skip_finalize:
            shard_label = "_".join(selected_shards).replace(".safetensors", "")
            with open(os.path.join(args.out, f"subspace_swap_stats_{shard_label}.json"), "w") as handle:
                json.dump({"per_weight": per_weight}, handle, indent=2)
            return

    missing = [shard for shard in shards if not os.path.isfile(os.path.join(args.out, shard))]
    if missing:
        raise RuntimeError(f"cannot finalize: missing output shard(s): {missing}")

    # The thinking checkpoint supplies the tokenizer/chat template and config.
    for filename in os.listdir(args.thinking):
        if filename.endswith((".json", ".txt", ".jinja")) and filename != "model.safetensors.index.json":
            shutil.copy(os.path.join(args.thinking, filename), os.path.join(args.out, filename))

    total_size = sum(os.path.getsize(os.path.join(args.out, shard)) for shard in shards)
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": base_map}, handle, indent=2)

    config_path = os.path.join(args.out, "config.json")
    config = json.load(open(config_path))
    config["torch_dtype"] = "float16"
    json.dump(config, open(config_path, "w"), indent=2)

    if args.finalize_only:
        for stats_path in glob.glob(os.path.join(args.out, "subspace_swap_stats_*.json")):
            per_weight.update(json.load(open(stats_path)).get("per_weight", {}))
        projected = len(per_weight)
    summary = {
        "method": "P_base_subspace + P_perp_thinking",
        "ratio": args.ratio,
        "truncate": False,
        "projected_matrices": projected,
        "max_subspace_error": max((x["subspace_error"] for x in per_weight.values()), default=0.0),
        "max_complement_error": max((x["complement_error"] for x in per_weight.values()), default=0.0),
        "elapsed_s": time.time() - started,
    }
    with open(os.path.join(args.out, "subspace_swap_stats.json"), "w") as handle:
        json.dump({"summary": summary, "per_weight": per_weight}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
