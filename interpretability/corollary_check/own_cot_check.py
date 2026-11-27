#!/usr/bin/env python3
"""Own-trace version of the corollary check for Qwen3-4B 2507 pair.

Part A: generate each variant's OWN CoT traces on the same 10 AIME problems
(base/sub/null/full, temp 0.6, top_p 0.95, max_new 4608) -> /tmp/own_cot_{tag}.json
(skipped if files exist).
Part B per trace set (the input distribution of each variant):
  (1) G = grad of L32-35 mean attention entropy H at W0 (Instruct), fp32 autograd,
      hooks on L32-35 only, 4 gradient groups, 512-token windows;
  (2) cos(DeltaW_perp, G_perp) per (layer, type), projection side by shape;
  (3) direct fixed-input check: H(W0 + DeltaW_perp) - H(W0) on the SAME windows,
      both models fp32 (null = models/perponly_2507).
Outputs: os.path.join(ROOT, "own_cot_results_{tag}.json")
"""
import json, math, os, sys, time
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
MODELS = {
    'base': BASE,
    'sub':  os.path.join(ROOT, "models/subonly_2507"),
    'null': os.path.join(ROOT, "models/perponly_2507"),
    'full': RL,
}
os.environ['HF_HUB_OFFLINE'] = '1'
DEV = 'cuda:0'
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
OUTDIR = ROOT

# ================= Part A: own traces =================
tok = AutoTokenizer.from_pretrained(RL, padding_side='left')
probs = []
for f in ('/tmp/aime24.jsonl', '/tmp/aime25.jsonl'):
    probs += [json.loads(l) for l in open(f)]
probs = probs[:10]
print(f'{len(probs)} problems', flush=True)

for tag, path in MODELS.items():
    fout = f'/tmp/own_cot_{tag}.json'
    if os.path.exists(fout):
        print(f'{tag}: traces exist, skip', flush=True)
        continue
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to(DEV).eval()
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p['problem']}],
                                     tokenize=False, add_generation_prompt=True) for p in probs]
    enc = tok(texts, return_tensors='pt', padding=True).to(DEV)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=4608, do_sample=True, temperature=0.6,
                         top_p=0.95, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    recs = [dict(id=p['id'], problem=p['problem'],
                 cot=tok.decode(gen[i], skip_special_tokens=True).split('</think>')[0],
                 n_gen_tok=int((gen[i] != 0).sum()))
            for i, p in enumerate(probs)]
    json.dump(recs, open(fout, 'w'))
    print(f'{tag}: wrote {fout} ({sum(r["n_gen_tok"] for r in recs) // 1000}k tok)', flush=True)
    del m, out, gen, enc
    torch.cuda.empty_cache()

# ================= Part B: gradient + direct check per trace set =================
HOLD = torch.zeros(26_000_000_000, dtype=torch.uint8, device=DEV)
print('CLAIMED 26GB', flush=True)
m0 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                          attn_implementation='eager').to(DEV).eval()
print('loaded W0 fp32', flush=True)

layers0 = m0.model.layers
nl = len(layers0)
tgt = {}
for l in range(nl):
    for name in TYPES:
        mod = layers0[l].self_attn if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else layers0[l].mlp
        w = getattr(mod, name).weight
        w.requires_grad_(True)
        tgt[(l, name)] = dict(w0=w, shape=tuple(w.shape))
keys = list(tgt.keys())
q = len(keys) // 4
groups = [keys[i * q:(i + 1) * q] for i in range(4)]

# Wt weights (fp32 on CPU, read per use)
m1 = AutoModelForCausalLM.from_pretrained(RL, torch_dtype=torch.float32,
                                          attn_implementation='eager').to('cpu').eval()
layers1 = m1.model.layers
wt = {}
for l in range(nl):
    for name in TYPES:
        mod = layers1[l].self_attn if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else layers1[l].mlp
        wt[(l, name)] = getattr(mod, name).weight.detach().float()
del HOLD
torch.cuda.empty_cache()
print('released hold; Wt weights staged on CPU', flush=True)

probs_holder = []
hooks = [m0.model.layers[l].self_attn.register_forward_hook(
    lambda mod, inp, out, h=probs_holder: h.append(out[1])) for l in [32, 33, 34, 35]]

def entropy(attns):
    return sum((-(a.float() * torch.log(a.float().clamp_min(1e-12))).sum(-1)).mean() for a in attns) / len(attns)

def load_seqs(fname):
    recs = json.load(open(fname))
    seqs = []
    for r in recs:
        t = tok(r['problem'] + '\n\n' + r['cot']).input_ids
        if len(t) >= 512:
            seqs.append(t[:512])
        if len(seqs) >= 10:
            break
    return seqs

# null model for the direct fixed-input check (loaded per set to limit memory)
def null_model():
    return AutoModelForCausalLM.from_pretrained(MODELS['null'], torch_dtype=torch.float32,
                                                attn_implementation='eager').to(DEV).eval()

for tag in ['base', 'sub', 'null', 'full']:
    fname = f'/tmp/own_cot_{tag}.json'
    seqs = load_seqs(fname)
    print(f'\n=== trace set: {tag} ({len(seqs)} seqs) ===', flush=True)

    G = {(l, name): torch.zeros_like(tgt[(l, name)]['w0']) for (l, name) in tgt}
    for si in range(len(seqs)):
        ids = torch.tensor([seqs[si]], device=DEV)
        for gi, gkeys in enumerate(groups):
            probs_holder.clear()
            m0(ids)
            H = entropy(list(probs_holder))
            gs = torch.autograd.grad(H, [tgt[k]['w0'] for k in gkeys], retain_graph=False,
                                     allow_unused=True)
            for k, g in zip(gkeys, gs):
                if g is not None:
                    G[k] += g.detach() / len(seqs)
            del gs, H
            torch.cuda.empty_cache()
        del ids
        if (si + 1) % 5 == 0:
            print(f'  grad {si + 1}/{len(seqs)} done', flush=True)

    results = []
    for (l, name), g in G.items():
        if g.norm() < 1e-30:
            continue
        w0 = tgt[(l, name)]['w0'].detach().float()
        dW = wt[(l, name)].to(DEV) - w0
        m, n = w0.shape
        U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
        if m > n:
            P_dW = U @ (U.T @ dW); P_g = U @ (U.T @ g)
        else:
            P_dW = dW @ Vh.T @ Vh; P_g = g @ Vh.T @ Vh
        dWp = (dW - P_dW).flatten(); Gp = (g - P_g).flatten()
        ip = float(torch.dot(dWp, Gp))
        cos = float(ip / (float(dWp.norm()) * float(Gp.norm()) + 1e-30))
        results.append(dict(layer=l, type=name, shape=[m, n], ip=ip, cos=cos,
                            gpar_frac=float(P_g.norm() / (g.norm() + 1e-30))))

    del G
    torch.cuda.empty_cache()

    # direct fixed-input entropy difference H(null) - H(base)
    mnull = null_model()
    dH = 0.0
    p0_holder, p1_holder = [], []
    h0 = [m0.model.layers[l].self_attn.register_forward_hook(
        lambda mod, inp, out, h=p0_holder: h.append(out[1])) for l in [32, 33, 34, 35]]
    h1 = [mnull.model.layers[l].self_attn.register_forward_hook(
        lambda mod, inp, out, h=p1_holder: h.append(out[1])) for l in [32, 33, 34, 35]]
    with torch.no_grad():
        for si in range(len(seqs)):
            ids = torch.tensor([seqs[si]], device=DEV)
            p0_holder.clear(); p1_holder.clear()
            m0(ids); mnull(ids)
            dH += (entropy(list(p1_holder)) - entropy(list(p0_holder))).item() / len(seqs)
    for h in h0 + h1:
        h.remove()
    del mnull
    torch.cuda.empty_cache()

    attn = [r for r in results if r['type'] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    mlp = [r for r in results if r['type'] in ('gate_proj', 'up_proj', 'down_proj')]
    def summ(rs, tg):
        cs = [r['cos'] for r in rs]
        print(f'  {tg:8s} n={len(rs):3d} sum_ip={sum(r["ip"] for r in rs):+.4f} '
              f'mean_cos={sum(cs)/len(cs):+.5f} neg={sum(1 for c in cs if c < 0)}/{len(rs)} '
              f'|Gp|/|G|={sum(r["gpar_frac"] for r in rs)/len(rs):.3f}', flush=True)
    summ(results, 'ALL')
    summ(attn, 'ATTN')
    summ(mlp, 'MLP')
    print(f'  direct fixed-input dH = H(null)-H(base) = {dH:+.5f} nats', flush=True)
    json.dump(dict(trace_set=tag, results=results, fixed_input_dH=dH),
              open(f'{OUTDIR}/own_cot_results_{tag}.json', 'w'))

for h in hooks:
    h.remove()
print('DONE', flush=True)
