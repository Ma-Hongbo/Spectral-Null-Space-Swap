#!/usr/bin/env python3
"""Distance of Instruct / Thinking from their parent Qwen3-4B-Base (CPU-only).
Reads weights directly from safetensors shards (no config needed; base cache is another user's).
dW_inst = W_inst - W_base (SFT), dW_think = W_think - W_base (SFT+RL).
Reports norms, rho vs |W_base|, per-type, cos(dW_inst, dW_think), and the RL delta |W_t - W_i|."""
import os, glob, math, re
import torch
from safetensors import safe_open

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

env = dict(l.strip().split('=', 1) for l in open(os.path.join(ROOT, 'paths2507.env')) if '=' in l and not l.startswith('#'))
# Resolve the base checkpoint from the HF cache (snapshot hashes differ per machine);
# override with S3_QWEN3_4B_BASE if it lives elsewhere.
BASE = os.environ.get("S3_QWEN3_4B_BASE") or sorted(glob.glob(
    f"{HF_CACHE}/hub/models--Qwen--Qwen3-4B-Base/snapshots/*"))[-1]
PATHS = dict(base=BASE, inst=env['BASE2'], think=env['RL2'])
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def grab(dirpath):
    w = {}
    for i in (1, 2, 3):
        shard = f'{dirpath}/model-0000{i}-of-00003.safetensors'
        with safe_open(shard, framework='pt') as f:
            for k in f.keys():
                m = re.match(r'model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.weight$', k)
                if m and m.group(3) in TYPES:
                    w[(int(m.group(1)), m.group(3))] = f.get_tensor(k).float()
    return w

print('reading shards...', flush=True)
W = {k: grab(p) for k, p in PATHS.items()}
print(f'loaded: {[(k, len(v)) for k, v in W.items()]}', flush=True)
assert set(W['base']) == set(W['inst']) == set(W['think']) and len(W['base']) == 252

tot = {k: 0.0 for k in ['w0_2', 'inst_2', 'think_2']}
bytype = {t: {k: 0.0 for k in tot} for t in TYPES}
ip_it = 0.0
for key in W['base']:
    l, name = key
    w0 = W['base'][key]
    di = W['inst'][key] - w0
    dt = W['think'][key] - w0
    ip_it += float((di * dt).sum())
    vals = dict(w0_2=w0.norm().item()**2, inst_2=di.norm().item()**2, think_2=dt.norm().item()**2)
    for k in tot:
        tot[k] += vals[k]
        bytype[name][k] += vals[k]

w0 = math.sqrt(tot['w0_2']); di = math.sqrt(tot['inst_2']); dt = math.sqrt(tot['think_2'])
print(f'\n|W_base|       = {w0:10.4f}')
print(f'|dW_inst|  (SFT)      = {di:10.4f}   rho = {di/w0*100:.2f}% of |W_base|')
print(f'|dW_think| (SFT+RL)   = {dt:10.4f}   rho = {dt/w0*100:.2f}% of |W_base|')
print(f'\n--> ratio |dW_think|/|dW_inst| = {dt/di:.3f}x   ({"Thinking" if dt>di else "Instruct"} fine-tuned the weights more)')
print(f'--> cos(dW_inst, dW_think) = {ip_it/(di*dt):+.4f}  (alignment of the two post-training updates)')
rl = math.sqrt(sum((W['think'][k] - W['inst'][k]).norm().item()**2 for k in W['base']))
print(f'--> |W_think - W_inst| (pure RL delta, our usual dW) = {rl:.4f}  ({rl/w0*100:.2f}% of |W_base|)')
print('\nper-type rho (% of |W0|):')
for t in TYPES:
    a = math.sqrt(bytype[t]['inst_2']) / math.sqrt(bytype[t]['w0_2']) * 100
    b = math.sqrt(bytype[t]['think_2']) / math.sqrt(bytype[t]['w0_2']) * 100
    print(f'  {t:10s} inst={a:6.2f}%  think={b:6.2f}%  ratio={b/a:5.2f}x')
