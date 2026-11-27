#!/usr/bin/env python3
"""Build the random-subspace control for the SUB (in-subspace) side:
W = W0 + Q(Q^T dW T)T^T with Q,T RANDOM orthonormal bases of the SAME dimensions
as the SVD basis of W0 (r = min(m,n) per weight). Same-dimension random control
for the norm decomposition (activation/function-space experiment).
Non-target weights from base; config/tokenizer copied from RL (build_perponly IO)."""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
          "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
def is_target(n):
    return n.endswith(TARGET)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--rl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    DT = getattr(torch, a.dtype)
    gen = torch.Generator().manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.rl, "model.safetensors.index.json")))["weight_map"]
    bidx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]
    shards = {}
    for n, sh in idx.items():
        shards.setdefault(sh, []).append(n)
    bcache = {}
    def get_base(name):
        sh = bidx[name]
        if sh not in bcache:
            bcache.clear()
            bcache[sh] = load_file(os.path.join(a.base, sh))
        return bcache[sh][name]
    for sh in sorted(shards):
        rl_sd = load_file(os.path.join(a.rl, sh)); out = {}
        for name in shards[sh]:
            if is_target(name):
                w0 = get_base(name).to(torch.float32)
                dW = rl_sd[name].to(torch.float32) - w0
                m, n = w0.shape
                r = min(m, n)
                Q = torch.linalg.qr(torch.randn(m, r, generator=gen), mode="reduced")[0]
                T = torch.linalg.qr(torch.randn(n, r, generator=gen), mode="reduced")[0]
                out[name] = (w0 + Q @ (Q.T @ dW @ T) @ T.T).to(DT)
            else:
                out[name] = get_base(name).to(DT)
        save_file(out, os.path.join(a.out, sh))
        print(f'saved {sh} ({len(out)} tensors)', flush=True)
    for f in os.listdir(a.rl):
        p = os.path.join(a.rl, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy2(p, os.path.join(a.out, f))
    print('built', a.out)

if __name__ == '__main__':
    main()
