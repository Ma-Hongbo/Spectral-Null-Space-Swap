#!/usr/bin/env python3
"""Random-subspace control for the norm decomposition (Qwen3-4B 2507 pair, CPU).
For each weight: dW = W_think - W_inst, W0 = W_inst, compact SVD -> U,V (r=min(m,n)).
P_S(X) = U U^T X V V^T. Measured share f = ||P_S(dW)||^2/||dW||^2.
Random-subspace baseline: for a random r-dim two-sided projection E||P_R(dW)||^2 = (r^2/(m n))||dW||^2
for ANY fixed dW -> baseline = r^2/(m n). Ratio f/baseline > 1 = preferential subspace alignment.
Also computed at rho=0.8 (k=ceil(0.8 r)), the S^3 default."""
import re, math, statistics
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
PATHS = dict(inst=env['BASE2'], think=env['RL2'])
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def grab(dirpath):
    w = {}
    for i in (1, 2, 3):
        with safe_open(f'{dirpath}/model-0000{i}-of-00003.safetensors', framework='pt') as f:
            for k in f.keys():
                m = re.match(r'model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.weight$', k)
                if m and m.group(3) in TYPES:
                    w[(int(m.group(1)), m.group(3))] = f.get_tensor(k).float()
    return w

print('reading shards...', flush=True)
W = {k: grab(p) for k, p in PATHS.items()}

rows = []
for key in W['inst']:
    l, name = key
    w0 = W['inst'][key]
    dW = W['think'][key] - w0
    m, n = w0.shape
    U, S, Vh = torch.linalg.svd(w0, full_matrices=False)   # U m x r, Vh r x n
    r = min(m, n)
    tot = float((dW * dW).sum())
    f1 = float((U @ (U.T @ dW @ Vh.T) @ Vh).square().sum()) / tot
    b1 = r * r / (m * n)
    k = math.ceil(0.8 * r)
    Uk, Vk = U[:, :k], Vh[:k]
    f08 = float((Uk @ (Uk.T @ dW @ Vk.T) @ Vk).square().sum()) / tot
    b08 = k * k / (m * n)
    rows.append(dict(layer=l, type=name, shape=[m, n], f1=f1, b1=b1, f08=f08, b08=b08))

def rep(rs):
    f1 = sum(x['f1'] * x['shape'][0] * x['shape'][1] for x in rs) / sum(x['shape'][0] * x['shape'][1] for x in rs)
    b1 = sum(x['b1'] * x['shape'][0] * x['shape'][1] for x in rs) / sum(x['shape'][0] * x['shape'][1] for x in rs)
    f08 = sum(x['f08'] * x['shape'][0] * x['shape'][1] for x in rs) / sum(x['shape'][0] * x['shape'][1] for x in rs)
    b08 = sum(x['b08'] * x['shape'][0] * x['shape'][1] for x in rs) / sum(x['shape'][0] * x['shape'][1] for x in rs)
    return f1, b1, f1 / b1, f08, b08, f08 / b08

print(f'\n{"type":10s} {"f1(meas)":>8s} {"b1(rand)":>8s} {"ratio1":>7s} | {"f08(meas)":>8s} {"b08(rand)":>8s} {"ratio08":>7s}')
for t in TYPES:
    rs = [x for x in rows if x['type'] == t]
    f1, b1, r1, f08, b08, r8 = rep(rs)
    print(f'{t:10s} {f1*100:7.1f}% {b1*100:7.1f}% {r1:7.3f} | {f08*100:7.1f}% {b08*100:7.1f}% {r8:7.3f}')
f1, b1, r1, f08, b08, r8 = rep(rows)
print(f'{"ALL":10s} {f1*100:7.1f}% {b1*100:7.1f}% {r1:7.3f} | {f08*100:7.1f}% {b08*100:7.1f}% {r8:7.3f}')
# per-layer ratio profile (mean over types)
print('\nper-layer ratio f/baseline at rho=1 (mean over 7 types):')
for l in range(0, 36, 4):
    line = []
    for y in range(l, min(l + 4, 36)):
        rs = [x for x in rows if x['layer'] == y]
        line.append(f'{statistics.mean(x["f1"] / x["b1"] for x in rs):.2f}')
    print(f'  L{l:2d}-{min(l+3,35):2d}: ' + ' '.join(line))
