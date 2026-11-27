#!/usr/bin/env python3
"""Verify the corollary condition <DeltaW_perp, G_perp> < 0 for Qwen3-4B 2507 pair.

G = gradient of attention entropy H at W0 (Instruct-2507), H = mean over layers 32-35
(the paper's measurement window), measured on the paper's fixed CoT traces
(/tmp/aime_cot.json, 10 traces x 1024-token windows, causal attention).

Per (layer, weight-type): compact SVD of W0; projection side by shape (tall m>n -> left
null, wide m<n -> right/row-space null), matching build_perponly.py conventions.
Reports per-layer cosine(DeltaW_perp, G_perp), inner product, and |G_par|/|G| (Assumption
sanity). Saves results to sar/grad_cos_results.json.
"""
import json, os, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


env = dict(l.strip().split('=', 1) for l in open(os.path.join(ROOT, 'paths2507.env')) if '=' in l and not l.startswith('#'))
BASE, RL = env['BASE2'], env['RL2']
os.environ['HF_HUB_OFFLINE'] = '1'
DEV = 'cuda:0'
CHUNK = 1
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
OUT = os.path.join(ROOT, 'grad_cos_results.json')

t0 = time.time()
tok = AutoTokenizer.from_pretrained(RL)
recs = json.load(open('/tmp/aime_cot.json'))
seqs = []
for r in recs:
    t = tok(r['problem'] + '\n\n' + r['cot']).input_ids
    if len(t) >= 512:
        seqs.append(t[:512])
    if len(seqs) >= 10:
        break
print(f'{len(seqs)} traces x 512 tokens (reduced from 1024 for GPU memory)', flush=True)

# claim the GPU instantly (beat racing jobs on this shared box); freed after model load
HOLD = torch.zeros(26_000_000_000, dtype=torch.uint8, device=DEV)
print('CLAIMED 26GB', flush=True)

m0 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                          attn_implementation='eager').to(DEV).eval()
print('loaded BASE on GPU', flush=True)
m1 = AutoModelForCausalLM.from_pretrained(RL, torch_dtype=torch.float32,
                                          attn_implementation='eager').to('cpu').eval()
print('loaded RL on CPU', flush=True)
del HOLD
torch.cuda.empty_cache()
print('released hold', flush=True)

layers0 = m0.model.layers
layers1 = m1.model.layers
nl = len(layers0)
tgt = {}
for l in range(nl):
    for name in TYPES:
        if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            w0 = getattr(layers0[l].self_attn, name).weight
            w1 = getattr(layers1[l].self_attn, name).weight
        else:
            w0 = getattr(layers0[l].mlp, name).weight
            w1 = getattr(layers1[l].mlp, name).weight
        w0.requires_grad_(True)
        tgt[(l, name)] = dict(w0=w0, w1=w1.detach().float(), shape=tuple(w0.shape))
print(f'targets: {len(tgt)} weights ({nl} layers x {len(TYPES)} types)', flush=True)

def entropy(attns):
    Hs = []
    for a in attns:
        p = a.float()
        Hs.append((-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).mean())
    return sum(Hs) / len(Hs)

# capture attention probs ONLY for layers 32-35 (the paper's window) via hooks
probs_holder = []
hooks = []
for l in [32, 33, 34, 35]:
    hooks.append(m0.model.layers[l].self_attn.register_forward_hook(
        lambda mod, inp, out, h=probs_holder: h.append(out[1])))

G = {(l, name): torch.zeros_like(tgt[(l, name)]['w0']) for (l, name) in tgt}
keys = list(tgt.keys())
q = len(keys) // 4
groups = [keys[i * q:(i + 1) * q] for i in range(4)]   # quarter gradient buffers
nchunks = math.ceil(len(seqs) / CHUNK)
for c in range(0, len(seqs), CHUNK):
    ids = torch.tensor(seqs[c:c + CHUNK], device=DEV)
    for gi, gkeys in enumerate(groups):
        probs_holder.clear()
        m0(ids)
        H = entropy(list(probs_holder))       # L32-35 mean entropy
        gs = torch.autograd.grad(H, [tgt[k]['w0'] for k in gkeys], retain_graph=False,
                                 allow_unused=True)
        for k, g in zip(gkeys, gs):
            if g is not None:
                G[k] += g.detach() / nchunks
        del gs, H
        torch.cuda.empty_cache()
        print(f'chunk {c // CHUNK + 1}/{nchunks} group {gi + 1}/4 done ({time.time() - t0:.0f}s)', flush=True)
    del ids
    torch.cuda.empty_cache()
for h in hooks:
    h.remove()

results = []
no_path = []
for (l, name), g in G.items():
    if g.norm() < 1e-30:          # no gradient path (e.g. last-layer o/MLP never feeds attention)
        no_path.append((l, name))
        continue
    w0 = tgt[(l, name)]['w0'].detach().float()
    dW = tgt[(l, name)]['w1'].to(DEV) - w0
    m, n = w0.shape
    U, S, Vh = torch.linalg.svd(w0, full_matrices=False)   # U: m x r, Vh: r x n
    if m > n:   # tall -> left-null (method convention)
        P_dW = U @ (U.T @ dW); P_g = U @ (U.T @ g)
    else:       # wide -> right/row-space null
        P_dW = dW @ Vh.T @ Vh; P_g = g @ Vh.T @ Vh
    dWp = (dW - P_dW).flatten(); Gp = (g - P_g).flatten()
    ip = float(torch.dot(dWp, Gp))
    cos = float(ip / (float(dWp.norm()) * float(Gp.norm()) + 1e-30))
    gpar_frac = float(P_g.norm() / (g.norm() + 1e-30))
    results.append(dict(layer=l, type=name, shape=[m, n], ip=ip, cos=cos, gpar_frac=gpar_frac))
print(f'no gradient path (skipped): {len(no_path)} weights: {no_path[:8]}...', flush=True)

json.dump(dict(results=results, elapsed_s=time.time() - t0), open(OUT, 'w'))

attn = [r for r in results if r['type'] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
mlp = [r for r in results if r['type'] in ('gate_proj', 'up_proj', 'down_proj')]
late = [r for r in results if 32 <= r['layer'] <= 35]
def summ(rs, tag):
    ips = sum(r['ip'] for r in rs); coses = [r['cos'] for r in rs]
    neg = sum(1 for r in rs if r['cos'] < 0)
    print(f'{tag:12s} n={len(rs):3d}  sum<dWp,Gp>={ips:+.4f}  mean_cos={sum(coses)/len(coses):+.4f}  neg_frac={neg}/{len(rs)}  mean|Gp|/|G|={sum(r["gpar_frac"] for r in rs)/len(rs):.3f}', flush=True)
summ(results, 'ALL')
summ(attn, 'ATTN')
summ(mlp, 'MLP')
summ(late, 'L32-35')
for r in results:
    if 32 <= r['layer'] <= 35:
        print(f"  L{r['layer']:2d} {r['type']:10s} cos={r['cos']:+.4f} ip={r['ip']:+.4e} |Gp|/|G|={r['gpar_frac']:.3f}", flush=True)
print('DONE', OUT, flush=True)
