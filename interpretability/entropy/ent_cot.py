"""Attention entropy vs pretraining step, measured on real CoT (paper's protocol).

Input = Qwen3-4B-Thinking-2507's own chain-of-thought on 10 AIME problems, one 1024-token
window per problem (the paper uses 10 problems x ~980 CoT positions).
Because attention is causal, H_t for t < 512 is identical to what a 512-long run would give,
so the same forward yields a number directly comparable to the earlier C4 / AIME-problem runs.
Computes in one checkpoint load: (1) per-layer H, (2) local entropy gradient split by the
head's row space, (3) the rho baseline E||P h||^2 / E||h||^2.
"""
import os, json, glob, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


HUB = f"{HF_CACHE}/hub/models--EleutherAI--pythia-1.4b"
STEPS = [0, 1000, 3000, 8000, 13000, 20000, 32000, 50000, 80000, 110000, 143000]
DEV, SEQ, NSEQ, NB = "cuda:0", 1024, 10, 8
os.environ["HF_HUB_OFFLINE"] = "1"
def snap(s): return f"{HUB}/snapshots/{open(f'{HUB}/refs/step{s}').read().strip()}"

tok = AutoTokenizer.from_pretrained(snap(143000))
recs = json.load(open("/tmp/aime_cot.json"))
seqs, short = [], []
for r in recs:
    t = tok(r["problem"] + "\n\n" + r["cot"]).input_ids
    (seqs if len(t) >= SEQ else short).append(t[:SEQ])
print(f"CoT: {len(recs)} traces, {len(seqs)} reach {SEQ} tok "
      f"(median trace {sorted(len(tok(r['cot']).input_ids) for r in recs)[len(recs)//2]} Pythia tok)", flush=True)
assert len(seqs) >= NSEQ, f"only {len(seqs)} long enough"
ids = torch.tensor(seqs[:NSEQ]).to(DEV)
print("input batch:", tuple(ids.shape), flush=True)

def ent(attns, upto=None):
    out = []
    for a in attns:
        p = a.float()
        H = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)          # (b, nh, q)
        out.append(H[:, :, 1:upto].mean())
    return torch.stack(out)

res = {}
for step in STEPS:
    m = AutoModelForCausalLM.from_pretrained(snap(step), dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
    c = m.config
    nh, hs, d, nl = (c.num_attention_heads, c.hidden_size // c.num_attention_heads,
                     c.hidden_size, c.num_hidden_layers)
    layers = m.gpt_neox.layers

    # ---- (1) entropy (full 1024, and the t<512 prefix) + (3) rho, one no-grad pass ----
    S = [torch.zeros(d, d, device=DEV) for _ in range(nl)]
    hooks = [layers[l].attention.query_key_value.register_forward_hook(
        (lambda l: (lambda mod, i, o: S[l].add_((lambda h: h.T @ h)(i[0].detach().float().reshape(-1, d)))))(l))
        for l in range(nl)]
    A = torch.zeros(nl, device=DEV); A512 = torch.zeros(nl, device=DEV)
    with torch.no_grad():
        for i in range(0, NSEQ, 2):
            o = m(ids[i:i+2], output_attentions=True); w = ids[i:i+2].shape[0] / NSEQ
            A += ent(o.attentions) * w; A512 += ent(o.attentions, 512) * w
            del o
    for h in hooks: h.remove()

    # ---- (2) local entropy gradient d H_l / d W_qkv^(l) ----
    W = [l.attention.query_key_value.weight for l in layers]
    for p_ in m.parameters(): p_.requires_grad_(False)
    for w_ in W: w_.requires_grad_(True)
    G = [torch.zeros_like(w_) for w_ in W]
    for i in range(0, NB, 1):
        o = m(ids[i:i+1], output_attentions=True); Hs = ent(o.attentions)
        for l in range(nl):
            G[l] += torch.autograd.grad(Hs[l], W[l], retain_graph=(l < nl - 1))[0] / NB
        del o, Hs; torch.cuda.empty_cache()

    ratio, rho, vchk = [], [], []
    for l in range(nl):
        Gf = G[l].detach().float().view(nh, 3, hs, d)
        Wf = W[l].detach().float().view(nh, 3, hs, d)
        vchk.append(float(Gf[:, 2].norm() / Gf.norm()))
        tot = float(torch.diagonal(S[l]).sum()); rr, pp = [], []
        for h in range(nh):
            V = torch.linalg.qr(Wf[h, 0].T, mode="reduced").Q
            rr.append(float((Gf[h, 0] @ V).norm() / Gf[h, 0].norm().clamp_min(1e-20)))
            pp.append(float(torch.einsum('ij,jk,ki->', V.T, S[l], V) / tot))
        ratio.append(sum(rr) / nh); rho.append(sum(pp) / nh)

    res[step] = dict(H=A.cpu().tolist(), H512=A512.cpu().tolist(), ratio=ratio, rho=rho,
                     vfrac=vchk, nl=nl, nh=nh, hs=hs, d=d)
    nr = (sum(ratio) / nl) / math.sqrt(sum(rho) / nl)
    lnr = (sum(ratio[nl-6:]) / 6) / math.sqrt(sum(rho[nl-6:]) / 6)
    print(f"step {step:>6}  H={sum(A.tolist())/nl:.4f}  H(t<512)={sum(A512.tolist())/nl:.4f}  "
          f"H[L23]={A[-1]:.3f}  |GP|/|G|={sum(ratio)/nl:.4f}  sqrt(rho)={math.sqrt(sum(rho)/nl):.4f}  "
          f"norm={nr:.3f}  late-norm={lnr:.3f}  vfrac={max(vchk):.1e}", flush=True)
    json.dump(res, open("/tmp/ent_cot.json", "w"))
    del m, W, G, S; torch.cuda.empty_cache()
print("DONE -> /tmp/ent_cot.json", flush=True)
