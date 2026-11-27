"""Build perp-only (W0+dW_perp) model for MoE architectures.
Adapted from build_perponly.py for Qwen3-MoE expert weights.

Decomposes all linear projection weights (attention + MoE experts + router)
into subspace (parallel to W0's row/col space) and null-space (perp) components,
keeping only the null-space: W_out = W0 + dW_perp.

With --mode=subonly, keeps only the subspace: W_out = W0 + dW_par.

Expert weights are small (768x2048), so SVD is fast per expert.
128 experts × 3 projections × 48 layers = 18,432 SVDs — expect ~20-40 min on GPU.
"""
import argparse, json, os, shutil, time
import torch
from safetensors.torch import load_file, save_file

# Match both dense-style and MoE-style weight names
TARGET_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "gate_proj.weight",   # matches both mlp.gate_proj and mlp.experts.N.gate_proj
    "up_proj.weight",
    "down_proj.weight",
    "mlp.gate.weight",    # MoE router
)

def is_target(name):
    return name.endswith(TARGET_SUFFIXES)


def load_index(model_dir):
    return json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]


def project(W0, dW, mode="perp"):
    """SVD-based projection. Returns dW_perp (mode='perp') or dW_par (mode='subonly')."""
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
    V = Vh.transpose(0, 1)
    dpar = U @ (U.transpose(0, 1) @ dW @ V) @ V.transpose(0, 1)
    if mode == "perp":
        return dW - dpar
    else:
        return dpar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base (Instruct) model dir")
    ap.add_argument("--rl", required=True, help="RL/Thinking model dir")
    ap.add_argument("--out", required=True, help="Output model dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--mode", default="perp", choices=["perp", "subonly"],
                    help="perp = W0+dW_perp (default), subonly = W0+dW_par")
    args = ap.parse_args()

    DT = getattr(torch, args.dtype)
    os.makedirs(args.out, exist_ok=True)

    base_map = load_index(args.base)
    rl_map = load_index(args.rl)

    # Group weights by shard file
    shards = {}
    for name, sh in base_map.items():
        shards.setdefault(sh, []).append(name)

    rl_cache = {}
    def get_rl(name):
        sh = rl_map[name]
        if sh not in rl_cache:
            rl_cache.clear()
            rl_cache[sh] = load_file(os.path.join(args.rl, sh))
        return rl_cache[sh][name]

    new_index = {}
    n_projected = 0
    n_skipped = 0
    t0 = time.time()

    for sh in sorted(shards):
        base_sd = load_file(os.path.join(args.base, sh))
        out = {}
        sh_projected = 0

        for name in shards[sh]:
            w0 = base_sd[name]
            if is_target(name) and name in rl_map and w0.dim() == 2:
                W0 = w0.to(args.device, torch.float32)
                WR = get_rl(name).to(args.device, torch.float32)
                dW = WR - W0
                delta = project(W0, dW, mode=args.mode)
                out[name] = (W0 + delta).to("cpu", DT)
                del W0, WR, dW, delta
                n_projected += 1
                sh_projected += 1
            else:
                out[name] = w0.to(DT) if w0.is_floating_point() else w0
                if is_target(name):
                    n_skipped += 1
            new_index[name] = sh

        save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
        elapsed = time.time() - t0
        print(f"saved {sh} ({sh_projected} projected, {elapsed:.0f}s elapsed)", flush=True)
        del base_sd, out
        torch.cuda.empty_cache()

    # Write index
    total_size = sum(os.path.getsize(os.path.join(args.out, s)) for s in shards)
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": new_index},
        open(os.path.join(args.out, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    # Copy config, tokenizer, etc. from RL model
    for fn in os.listdir(args.rl):
        if fn.endswith((".json", ".txt", ".jinja", ".py", ".model")) and \
           not fn.startswith("model-0") and fn != "model.safetensors.index.json":
            shutil.copy(os.path.join(args.rl, fn), os.path.join(args.out, fn))

    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["torch_dtype"] = args.dtype
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    elapsed = time.time() - t0
    print(f"\n{args.mode}-only -> {args.out}")
    print(f"Projected {n_projected} weights, skipped {n_skipped}, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
