#!/usr/bin/env python3
"""Frobenius distances among the 4 variants (Qwen3-4B 2507 pair, CPU-only).
d(null, inst)=|DW_perp|, d(null, think)=|DW_par|, d(sub, inst)=|DW_par|, d(sub, think)=|DW_perp|.
Reports totals + per-type breakdown + normalized ratios by |DW| = d(inst, think)."""
import os, math
import torch
from transformers import AutoModelForCausalLM

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


env = dict(l.strip().split('=', 1) for l in open(os.path.join(ROOT, 'paths2507.env')) if '=' in l and not l.startswith('#'))
os.environ['HF_HUB_OFFLINE'] = '1'
PATHS = dict(inst=env['BASE2'], think=env['RL2'],
             null=os.path.join(ROOT, "models/perponly_2507"),
             sub=os.path.join(ROOT, "models/subonly_2507"))
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def grab(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    w = {}
    for l in range(36):
        for name in TYPES:
            mod = m.model.layers[l].self_attn if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else m.model.layers[l].mlp
            w[(l, name)] = getattr(mod, name).weight.detach().float()
    del m
    return w

print('loading 4 models (CPU fp32)...', flush=True)
W = {k: grab(p) for k, p in PATHS.items()}
print('loaded', flush=True)

tot = {k: 0.0 for k in ['dW2', 'null_inst2', 'null_think2', 'sub_inst2', 'sub_think2']}
bytype = {t: {k: 0.0 for k in tot} for t in TYPES}
for (l, name) in W['inst']:
    dW = W['think'][(l, name)] - W['inst'][(l, name)]
    n_i = W['null'][(l, name)] - W['inst'][(l, name)]
    n_t = W['null'][(l, name)] - W['think'][(l, name)]
    s_i = W['sub'][(l, name)] - W['inst'][(l, name)]
    s_t = W['sub'][(l, name)] - W['think'][(l, name)]
    vals = dict(dW2=dW.norm().item()**2, null_inst2=n_i.norm().item()**2,
                null_think2=n_t.norm().item()**2, sub_inst2=s_i.norm().item()**2,
                sub_think2=s_t.norm().item()**2)
    for k in tot:
        tot[k] += vals[k]
        bytype[name][k] += vals[k]

dW = math.sqrt(tot['dW2'])
def r(v):
    return math.sqrt(tot[v]) / dW
print(f"\nd(inst, think) = |DW| = {dW:.4f}")
print(f"d(null, inst)  = |DW_perp| = {math.sqrt(tot['null_inst2']):.4f}   ({r('null_inst2')*100:.1f}% of |DW|)")
print(f"d(null, think) = |DW_par|  = {math.sqrt(tot['null_think2']):.4f}   ({r('null_think2')*100:.1f}% of |DW|)")
print(f"d(sub, inst)   = |DW_par|  = {math.sqrt(tot['sub_inst2']):.4f}   ({r('sub_inst2')*100:.1f}% of |DW|)")
print(f"d(sub, think)  = |DW_perp| = {math.sqrt(tot['sub_think2']):.4f}   ({r('sub_think2')*100:.1f}% of |DW|)")
print(f"\n--> null is closer to {'Thinking' if tot['null_think2'] < tot['null_inst2'] else 'Instruct'}"
      f" ({min(r('null_think2'), r('null_inst2'))*100:.1f}% vs {max(r('null_think2'), r('null_inst2'))*100:.1f}% of |DW|)")
print(f"--> sub  is closer to {'Instruct' if tot['sub_inst2'] < tot['sub_think2'] else 'Thinking'}")
print('\nper-type |DW_par|/|DW| (null-think) and |DW_perp|/|DW| (null-inst):')
for t in TYPES:
    p = math.sqrt(bytype[t]['null_think2']) / math.sqrt(bytype[t]['dW2'])
    q = math.sqrt(bytype[t]['null_inst2']) / math.sqrt(bytype[t]['dW2'])
    print(f'  {t:10s} par={p*100:5.1f}%  perp={q*100:5.1f}%')
