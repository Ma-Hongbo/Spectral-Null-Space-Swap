#!/usr/bin/env python3
"""Per-problem <DeltaW_perp, G_perp> for Qwen3-4B 2507 (the per-trace breakdown).

grad_cos_7datasets.py averages the gradient over its 10 traces BEFORE projecting, so it
can only report one number per dataset. This reports one number per PROBLEM, which is
what tells you whether the dataset-level figure is a stable quantity or the mean of
wildly varying per-problem values.

Method. Since DeltaW_perp lies in the orthogonal complement of the projected subspace,
the projector is self-adjoint and idempotent, so

    <DeltaW_perp, G_perp> = <DeltaW_perp, G>

exactly (verified in fp32 to ~1e-9 relative to |dWp|*|g| at the real matrix sizes). So
DeltaW_perp is precomputed once per matrix, and each trace only needs a dot product --
no projection of g, and no 252 gradient buffers held on the GPU. Peak GPU drops from
~31 GB (fp32 model + all gradient buffers) to ~17 GB.

Because the dataset mean of the per-trace values is the gradient-mean value, the
`mean` row here must reproduce grad_cos_7datasets.py's ALL column. That is the check.

  python grad_cos_perprob.py --datasets aime24,aime25 --nseq 10
Output: {OUTDIR}/grad_cos_perprob/{ds}.json
"""
import json, os, math, time, glob
import torch

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


argv = os.sys.argv
def arg(f, d=None):
    return argv[argv.index(f) + 1] if f in argv else d

DATA_DIR = arg('--data', os.path.join(ROOT, 'data'))
OUTDIR = arg('--out', ROOT)
WORK = arg('--work', '/tmp/cot7d')
NSEQ = int(arg('--nseq', 10))
SEQ = 512
DATASETS = (arg('--datasets') or 'aime24,aime25').split(',')
os.environ['HF_HUB_OFFLINE'] = '1'
DEV = 'cuda:0'

env = dict(l.strip().split('=', 1) for l in open(f'{OUTDIR}/paths2507.env')
           if '=' in l and not l.startswith('#'))
BASE, RL = env['BASE2'], env['RL2']
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
LAYERS = [32, 33, 34, 35]
DEST = f'{OUTDIR}/grad_cos_perprob'
os.makedirs(DEST, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(RL)

def load_seqs(ds):
    """Reuse the CoT that grad_cos_7datasets.py generated; never regenerate here."""
    p = f'{WORK}/{ds}.json'
    if not os.path.exists(p):
        raise SystemExit(f'FATAL: {p} missing -- run grad_cos_7datasets.py --datasets {ds} first')
    recs = json.load(open(p))
    seqs, used = [], []
    for r in recs:
        t = tok(r['problem'] + '\n\n' + r['cot']).input_ids
        if len(t) >= SEQ:
            seqs.append(t[:SEQ]); used.append(r.get('id', f'{ds}-{len(used)}'))
        if len(seqs) >= NSEQ:
            break
    if len(seqs) < NSEQ:
        raise SystemExit(f'FATAL: {p} gives only {len(seqs)} traces >= {SEQ} tokens')
    return seqs, used

m0 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                          attn_implementation='eager').to(DEV).eval()
print('loaded BASE fp32', flush=True)
m1 = AutoModelForCausalLM.from_pretrained(RL, torch_dtype=torch.float32,
                                          attn_implementation='eager').to('cpu').eval()
layers0, layers1 = m0.model.layers, m1.model.layers
tgt = {}
for l in range(len(layers0)):
    for name in TYPES:
        if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            w0 = getattr(layers0[l].self_attn, name).weight
            w1 = getattr(layers1[l].self_attn, name).weight
        else:
            w0 = getattr(layers0[l].mlp, name).weight
            w1 = getattr(layers1[l].mlp, name).weight
        w0.requires_grad_(True)
        tgt[(l, name)] = dict(w0=w0, w1=w1.detach().float())
del m1
torch.cuda.empty_cache()
DIM_TOT = sum(tgt[k]['w0'].shape[0] * tgt[k]['w0'].shape[1] for k in tgt)
DIM_LATE = sum(tgt[k]['w0'].shape[0] * tgt[k]['w0'].shape[1] for k in tgt if k[0] in LAYERS)
print(f'dim total {DIM_TOT:,} (1/sqrt {1/math.sqrt(DIM_TOT):.2e})  '
      f'late {DIM_LATE:,} (1/sqrt {1/math.sqrt(DIM_LATE):.2e})', flush=True)

# Precompute DeltaW_perp once per matrix, kept on CPU (14.5 GB RAM on a 1 TB box).
# The projection basis is kept too (~14.5 GB more): the inner product needs only
# <dWp, g>, but the COSINE needs |G_perp| per trace, and that does require projecting g.
dWp, basis, tall, dwp_norm2 = {}, {}, {}, {}
for key, d in tgt.items():
    w0 = d['w0'].detach().float()
    dW = d['w1'].to(DEV) - w0
    m, n = w0.shape
    U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
    P = (U @ (U.T @ dW)) if m > n else (dW @ Vh.T @ Vh)
    dWp[key] = (dW - P).flatten().cpu()
    basis[key] = (U if m > n else Vh).cpu()
    tall[key] = m > n
    dwp_norm2[key] = float((dW - P).norm()) ** 2
    del P, dW, U, Vh
    torch.cuda.empty_cache()
print(f'dWp + basis precomputed for {len(dWp)} matrices ({time.time()-t0:.0f}s)', flush=True)

def entropy(attns):
    return sum((-(a.float() * torch.log(a.float().clamp_min(1e-12))).sum(-1)).mean()
               for a in attns) / len(attns)

holder = []
hooks = [layers0[l].self_attn.register_forward_hook(
    lambda mod, i, o, h=holder: h.append(o[1])) for l in LAYERS]
keys = list(tgt.keys()); q = len(keys) // 4
groups = [keys[i*q:(i+1)*q] for i in range(4)]

for ds in DATASETS:
    out = f'{DEST}/{ds}.json'
    if os.path.exists(out):
        print(f'[{ds}] done, skip', flush=True); continue
    seqs, ids_used = load_seqs(ds)
    print(f'\n=== {ds}: {len(seqs)} traces ===', flush=True)

    per_trace, per_trace_late = [], []
    per_cos, per_cos_late = [], []
    detail = []
    for si, ids in enumerate(seqs):
        ids = torch.tensor([ids], device=DEV)
        acc = {k: 0.0 for k in keys}
        gp_norm2 = {k: 0.0 for k in keys}
        for gkeys in groups:
            holder.clear(); m0(ids)
            H = entropy(list(holder))
            gs = torch.autograd.grad(H, [tgt[k]['w0'] for k in gkeys],
                                     retain_graph=False, allow_unused=True)
            for k, g in zip(gkeys, gs):
                if g is None:
                    continue
                B = basis[k].to(DEV)
                Pg = (B @ (B.T @ g)) if tall[k] else (g @ B.T @ B)
                Gp = (g - Pg).flatten()
                acc[k] = float(torch.dot(dWp[k].to(DEV), Gp))
                gp_norm2[k] = float(Gp.norm()) ** 2
                del Pg, Gp, B
            del gs, H
            torch.cuda.empty_cache()
        del ids
        total = sum(acc.values())
        late = sum(v for k, v in acc.items() if k[0] in LAYERS)
        # concatenated cosine for this trace: stack DeltaW_perp and G_perp into single
        # vectors. Random baseline 1/sqrt(D).
        dw_n = math.sqrt(sum(dwp_norm2.values()))
        gp_n = math.sqrt(sum(gp_norm2.values()))
        dw_nl = math.sqrt(sum(v for k, v in dwp_norm2.items() if k[0] in LAYERS))
        gp_nl = math.sqrt(sum(v for k, v in gp_norm2.items() if k[0] in LAYERS))
        cos_all = total / (dw_n * gp_n + 1e-30)
        cos_late = late / (dw_nl * gp_nl + 1e-30)
        per_trace.append(total); per_trace_late.append(late)
        per_cos.append(cos_all); per_cos_late.append(cos_late)
        detail.append(dict(trace=ids_used[si], all=total, late=late,
                           cos_all=cos_all, cos_late=cos_late,
                           rows=[dict(layer=k[0], type=k[1], ip=v) for k, v in acc.items()]))
        print(f'  [{ds}] {ids_used[si]:>12}  ALL={total:+.5f}  L32-35={late:+.5f}  '
              f'cos={cos_all:+.3e}  cosL={cos_late:+.3e}  (1/sqrtD={1/math.sqrt(DIM_TOT):.1e})',
              flush=True)

    mean_all = sum(per_trace) / len(per_trace)
    mean_late = sum(per_trace_late) / len(per_trace_late)
    sd_all = (sum((x - mean_all) ** 2 for x in per_trace) / max(1, len(per_trace) - 1)) ** .5
    sd_late = (sum((x - mean_late) ** 2 for x in per_trace_late) / max(1, len(per_trace_late) - 1)) ** .5
    mean_cos = sum(per_cos) / len(per_cos)
    mean_cos_l = sum(per_cos_late) / len(per_cos_late)
    res = dict(dataset=ds, nseq=len(seqs), seq=SEQ, per_trace=per_trace,
               per_trace_late=per_trace_late, mean_all=mean_all, mean_late=mean_late,
               sd_all=sd_all, sd_late=sd_late, per_cos=per_cos, per_cos_late=per_cos_late,
               mean_cos=mean_cos, mean_cos_late=mean_cos_l,
               dim_total=DIM_TOT, dim_late=DIM_LATE, detail=detail, elapsed_s=time.time() - t0)
    tmp = out + '.tmp'
    json.dump(res, open(tmp, 'w')); os.replace(tmp, out)
    print(f'[{ds}] mean ALL={mean_all:+.5f} (sd {sd_all:.5f})  '
          f'mean L32-35={mean_late:+.5f} (sd {sd_late:.5f})', flush=True)
    print(f'[{ds}] catcos ALL={mean_cos:+.3e} vs 1/sqrt(D)={1/math.sqrt(DIM_TOT):.2e} '
          f'-> {abs(mean_cos)*math.sqrt(DIM_TOT):.2f}x random  |  '
          f'L32-35 catcos={mean_cos_l:+.3e} -> {abs(mean_cos_l)*math.sqrt(DIM_LATE):.2f}x',
          flush=True)
    print(f'[{ds}] CHECK mean must match grad_cos_7datasets.py ALL for {ds}', flush=True)

for h in hooks:
    h.remove()
print('DONE', flush=True)
