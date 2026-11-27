#!/usr/bin/env python3
"""<DeltaW_perp, G_perp> for the 7 math datasets (Qwen3-4B 2507 pair), one GPU per dataset.

  G_ds = grad of L32-35 mean attention entropy H, at W0 (Instruct-2507), fp32 autograd,
         measured on that dataset's chain-of-thought.
  <DeltaW_perp, G_perp> per (layer, weight-type), projection side by shape (tall m>n ->
  left/U U^T, wide m<n -> row/Vh^T Vh), matching grad_cos.py and build_perponly.py.

The dataset files (sar/data/*.jsonl) hold {id, problem, answer} and NO chain-of-thought, so
phase A generates one with Qwen3-4B-Thinking-2507 exactly as gen_cot.py does (bf16, chat
template, temp 0.6, top_p 0.95, max_new 4608, keep the </think> prefix), and phase B runs
the gradient on the first 512 tokens of problem + "\n\n" + cot. Phase A's model is freed
before phase B allocates the fp32 copy, so peak is ~30 GB either way.

Sharding: run one process per dataset. The launcher sets CUDA_VISIBLE_DEVICES, so this
script always talks to cuda:0 -- do not pass a device index here.

  python grad_cos_7datasets.py --datasets aime24            # one dataset, one GPU
  python grad_cos_7datasets.py                              # all 7, sequentially
  python grad_cos_7datasets.py --report                     # print the combined table

Output is one JSON per dataset under {OUTDIR}/grad_cos_7datasets/, written as soon as that
dataset finishes, so concurrent processes never touch the same file and a crash keeps the
rest. Re-running skips datasets already present. See run_7datasets.sh for the launcher.
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
def arg(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default

DATA_DIR = arg('--data', os.path.join(ROOT, 'data'))
OUTDIR = arg('--out', ROOT)
WORK = arg('--work', '/tmp/cot7d')
NSEQ = int(arg('--nseq', 10))
SEQ = 512
POOL = int(arg('--pool', 12))          # problems to generate per dataset (>= NSEQ)
# gen_cot.py used 4608, but the measurement window is only the first SEQ=512 tokens of
# (problem + cot), so anything past that is discarded. Autoregressive sampling is
# prefix-consistent, so stopping at MAXNEW still yields the identical 512-token window
# and cuts generation ~7x. Keep MAXNEW comfortably above SEQ.
MAXNEW = int(arg('--maxnew', 640))
DATASETS_ALL = ['aime24', 'aime25', 'amc23', 'cmimc25', 'hmmt25', 'math500', 'gsm8k']
os.environ['HF_HUB_OFFLINE'] = '1'
DEV = 'cuda:0'                          # the launcher pins the real card via CUDA_VISIBLE_DEVICES

env = dict(l.strip().split('=', 1) for l in open(f'{OUTDIR}/paths2507.env')
           if '=' in l and not l.startswith('#'))
BASE, RL = env['BASE2'], env['RL2']
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
LAYERS = [32, 33, 34, 35]               # the paper's measurement window
DEST = f'{OUTDIR}/grad_cos_7datasets'

def rows_of(ds):
    p = f'{DEST}/{ds}.json'
    return json.load(open(p))['rows'] if os.path.exists(p) else None

# ---------------------------------------------------------------- report mode
if '--report' in argv:
    print(f'{"dataset":<9} {"N":>4} {"ALL":>12} {"L32-35":>12} {"ATTN":>12} {"MLP":>12} '
          f'{"mean_cos":>10} {"neg":>7} {"|Gp|/|G|":>9}')
    print('-' * 92)
    pooled = []
    print()
    print(f'{"dataset":<9} {"catcos ALL":>11} {"1/sqrt(D)":>10} {"ratio":>7} '
          f'{"catcos L32-35":>13} {"1/sqrt(Dl)":>10} {"ratio":>7}')
    print('-' * 74)
    for ds in DATASETS_ALL:
        rs = rows_of(ds)
        if not rs:
            continue
        pooled += rs
        f_ = f'{DEST}/{ds}.json'
        if os.path.exists(f_):
            d = json.load(open(f_))
            ca, cl = d.get('catcos_all'), d.get('catcos_late')
            Da, Dl = d.get('dim_total'), d.get('dim_late')
            if ca is not None:
                ra = math.sqrt(Da)
                rl = math.sqrt(Dl)
                print(f'{ds:<9} {ca:>+11.3e} {1/ra:>10.2e} {abs(ca)*ra:>7.2f} '
                      f'{cl:>+13.3e} {1/rl:>10.2e} {abs(cl)*rl:>7.2f}')
    print('  ratio = |cos| / (1/sqrt(D)): how many times larger than two random vectors.')
    print('  A ratio near 1 means no detectable alignment; >>1 means real.\n', flush=True)
    for ds in DATASETS_ALL:
        rs = rows_of(ds)
        if not rs:
            print(f'{ds:<9}  (not computed yet)')
            continue
        A = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        cs = [r['cos'] for r in rs]
        print(f'{ds:<9} {len(rs):>4} {sum(r["ip"] for r in rs):>+12.5f} '
              f'{sum(r["ip"] for r in rs if r["layer"] in LAYERS):>+12.5f} '
              f'{sum(r["ip"] for r in rs if r["type"] in A):>+12.5f} '
              f'{sum(r["ip"] for r in rs if r["type"] not in A):>+12.5f} '
              f'{sum(cs)/len(cs):>+10.2e} {sum(1 for c in cs if c < 0):>3}/{len(cs):<3} '
              f'{sum(r["gpar_frac"] for r in rs)/len(rs):>9.3f}')
    if pooled:
        A = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        cs = [r['cos'] for r in pooled]
        ang = [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cs]
        print('-' * 92)
        print(f'{"POOLED":<9} {len(pooled):>4} {sum(r["ip"] for r in pooled):>+12.5f} '
              f'{sum(r["ip"] for r in pooled if r["layer"] in LAYERS):>+12.5f} '
              f'{sum(r["ip"] for r in pooled if r["type"] in A):>+12.5f} '
              f'{sum(r["ip"] for r in pooled if r["type"] not in A):>+12.5f} '
              f'{sum(cs)/len(cs):>+10.2e} {sum(1 for c in cs if c < 0):>3}/{len(cs):<3} '
              f'{sum(r["gpar_frac"] for r in pooled)/len(pooled):>9.3f}')
        print(f'  angle range {min(ang):.1f}-{max(ang):.1f} deg')
    os.sys.exit(0)

DATASETS = (arg('--datasets') or ','.join(DATASETS_ALL)).split(',')
os.makedirs(DEST, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()

# ---------------------------------------------------------------- phase A: CoT
def gen_cot(ds):
    """Generate this dataset's CoT with Thinking-2507, gen_cot.py's exact settings."""
    out = f'{WORK}/{ds}.json'
    if os.path.exists(out):
        recs = json.load(open(out))
        # guard on POOL, not NSEQ: some of these traces are too short to reach SEQ and
        # get dropped later, so a cache built with a smaller --pool must be regenerated
        # (gsm8k: 12 traces yielded only 9 long enough)
        if len(recs) >= POOL:
            print(f'[{ds}] CoT cached ({len(recs)} traces)', flush=True)
            return recs
    probs = [json.loads(l) for l in open(f'{DATA_DIR}/{ds}.jsonl') if l.strip()][:POOL]
    print(f'[{ds}] generating CoT for {len(probs)} problems', flush=True)
    tok = AutoTokenizer.from_pretrained(RL, padding_side='left')
    # transformers 4.55 wants torch_dtype; the newer `dtype=` alias arrived in 4.56
    m = AutoModelForCausalLM.from_pretrained(RL, torch_dtype=torch.bfloat16).to(DEV).eval()
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p['problem']}],
                                     tokenize=False, add_generation_prompt=True) for p in probs]
    enc = tok(texts, return_tensors='pt', padding=True).to(DEV)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=0.6,
                       top_p=0.95, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = g[:, enc.input_ids.shape[1]:]
    recs = []
    for i, p in enumerate(probs):
        cot = tok.decode(gen[i], skip_special_tokens=True).split('</think>')[0]
        recs.append(dict(id=p['id'], problem=p['problem'], cot=cot))
    del m, g, gen, enc, texts
    torch.cuda.empty_cache()
    json.dump(recs, open(out, 'w'))
    print(f'[{ds}] wrote {out} ({len(recs)} traces, '
          f'{sum(len(r["cot"]) for r in recs)//1000}k chars)', flush=True)
    return recs

# ---------------------------------------------------------------- phase B
Q_FIELDS = ['problem', 'question']
C_FIELDS = ['cot', 'response', 'solution', 'reasoning']

def make_seqs(ds, recs, tok):
    seqs = []
    for r in recs:
        q = next((r[f] for f in Q_FIELDS if isinstance(r.get(f), str)), '')
        c = next((r[f] for f in C_FIELDS if isinstance(r.get(f), str)), r.get('text', ''))
        t = tok((q + '\n\n' + c) if q else c).input_ids
        if len(t) >= SEQ:
            seqs.append(t[:SEQ])
        if len(seqs) >= NSEQ:
            break
    return seqs

def run(ds):
    if rows_of(ds):
        print(f'[{ds}] already computed, skip', flush=True)
        return
    recs = gen_cot(ds)
    tok = AutoTokenizer.from_pretrained(RL)
    seqs = make_seqs(ds, recs, tok)
    if len(seqs) < NSEQ:
        raise SystemExit(f'[{ds}] FATAL: only {len(seqs)} traces reach {SEQ} tokens; '
                         f'raise --pool above {POOL}')
    print(f'[{ds}] {len(seqs)} seqs x {SEQ} tok; loading fp32 pair', flush=True)

    m0 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                              attn_implementation='eager').to(DEV).eval()
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
            tgt[(l, name)] = dict(w0=w0, w1=w1.detach().float(), shape=tuple(w0.shape))
    del m1
    torch.cuda.empty_cache()
    print(f'[{ds}] models ready ({time.time()-t0:.0f}s elapsed)', flush=True)

    def entropy(attns):
        return sum((-(a.float() * torch.log(a.float().clamp_min(1e-12))).sum(-1)).mean()
                   for a in attns) / len(attns)

    holder = []
    hooks = [layers0[l].self_attn.register_forward_hook(
        lambda mod, i, o, h=holder: h.append(o[1])) for l in LAYERS]
    keys = list(tgt.keys()); q = len(keys) // 4
    groups = [keys[i*q:(i+1)*q] for i in range(4)]

    G = {k: torch.zeros_like(tgt[k]['w0']) for k in tgt}
    for si, ids in enumerate(seqs):
        ids = torch.tensor([ids], device=DEV)
        for gkeys in groups:
            holder.clear()
            m0(ids)
            H = entropy(list(holder))
            gs = torch.autograd.grad(H, [tgt[k]['w0'] for k in gkeys],
                                     retain_graph=False, allow_unused=True)
            for k, g in zip(gkeys, gs):
                if g is not None:
                    G[k] += g.detach() / len(seqs)
            del gs, H
            torch.cuda.empty_cache()
        del ids
        if (si + 1) % 5 == 0:
            print(f'[{ds}] grad {si+1}/{len(seqs)} ({time.time()-t0:.0f}s)', flush=True)
    for h in hooks:
        h.remove()

    rows, chk = [], 0.0
    for (l, name), g in G.items():
        if g.norm() < 1e-30:            # no gradient path (L35 o/MLP never feeds attention)
            continue
        w0 = tgt[(l, name)]['w0'].detach().float()
        dW = tgt[(l, name)]['w1'].to(DEV) - w0
        m, n = w0.shape
        U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
        if m > n:                       # tall -> left-null (method convention)
            P_dW = U @ (U.T @ dW); P_g = U @ (U.T @ g)
        else:                           # wide -> right/row-space null
            P_dW = dW @ Vh.T @ Vh; P_g = g @ Vh.T @ Vh
        dWp = (dW - P_dW).flatten(); Gp = (g - P_g).flatten()
        ip = float(torch.dot(dWp, Gp))
        # dWp lies in the complement, so <dWp,Gp> must equal <dWp,g>; catches a
        # projection-convention slip on the g side. Normalise by the term scale
        # |dWp|*|g|, NOT by ip: the null components are near-orthogonal, so ip sits
        # ~1e4 below that scale and dividing by it inflates fp32 roundoff into a
        # false alarm. Measured fp32 roundoff on this identity is ~1e-9.
        chk = max(chk, abs(ip - float(torch.dot(dWp, g.flatten()))) /
                      (float(dWp.norm()) * float(g.norm()) + 1e-30))
        rows.append(dict(layer=l, type=name, shape=[m, n], ip=ip,
                         cos=float(ip / (float(dWp.norm()) * float(Gp.norm()) + 1e-30)),
                         dwp_norm=float(dWp.norm()), gp_norm=float(Gp.norm()),
                         gpar_frac=float(P_g.norm() / (g.norm() + 1e-30))))
        del P_dW, P_g, dWp, Gp, dW, U, Vh
    print(f'[{ds}] projection check: max dev from <dWp,g> = {chk:.2e} '
          f'({"OK" if chk < 1e-6 else "SUSPECT"})', flush=True)

    late = [r for r in rows if r['layer'] in LAYERS]
    # concatenated cosine: stack every DeltaW_perp into one vector and every G_perp into
    # another, then take the angle between them. This is what the per-row `cos` values
    # cannot give, because it weights each matrix by |dWp|*|Gp| rather than counting it
    # equally. Random baseline for two unrelated vectors in D dims is 1/sqrt(D).
    def catcos(rs):
        s = sum(r['ip'] for r in rs)
        a = math.sqrt(sum(r['dwp_norm'] ** 2 for r in rs))
        b = math.sqrt(sum(r['gp_norm'] ** 2 for r in rs))
        return s / (a * b + 1e-30)
    D = sum(r['shape'][0] * r['shape'][1] for r in rows)
    tmp = f'{DEST}/{ds}.json.tmp'
    json.dump(dict(dataset=ds, nseq=len(seqs), seq=SEQ, pool=POOL, rows=rows,
                   sum_ip_all=sum(r['ip'] for r in rows),
                   sum_ip_late=sum(r['ip'] for r in late),
                   catcos_all=catcos(rows), catcos_late=catcos(late),
                   dim_total=D, dim_late=sum(r['shape'][0] * r['shape'][1] for r in late),
                   projection_check=chk, elapsed_s=time.time() - t0), open(tmp, 'w'))
    os.replace(tmp, f'{DEST}/{ds}.json')      # atomic: safe under concurrent launches
    print(f'[{ds}] DONE sum<dWp,Gp> ALL={sum(r["ip"] for r in rows):+.5f} '
          f'L32-35={sum(r["ip"] for r in late):+.5f}  ({time.time()-t0:.0f}s)', flush=True)

for ds in DATASETS:
    run(ds)
print(f'ALL REQUESTED DONE ({time.time()-t0:.0f}s)', flush=True)
