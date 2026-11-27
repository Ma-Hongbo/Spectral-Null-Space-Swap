#!/usr/bin/env python3
"""Multi-dataset functional-norm shares + random-subspace control.
5 input sets: aime (10 probs), math500 (20), gsm8k (20), amc23 (20), wiki (10 passages).
Models: base=W0, sub=W0+dW_par, perp=W0+dW_perp, full=W0+dW, rand=W0+P_R(dW).
Per set: f~_* = |Dh_*|^2 / |Dh_full|^2 (per-token mean, layer-unweighted global mean),
and efficiency = f~ / energy-share (energy shares fixed from weight control:
sub 0.381, perp 0.619, rand 0.345)."""
import json, os, math
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
MODELS = dict(sub=os.path.join(ROOT, "models/subonly_2507"),
              perp=os.path.join(ROOT, "models/perponly_2507"),
              full=RL,
              rand=os.path.join(ROOT, "models/randpar_2507"))
ENERGY = dict(sub=0.381, perp=0.619, rand=0.345)
DEV = 'cuda:0'
OUT = os.path.join(ROOT, "functional_multi_control.json")

tok = AutoTokenizer.from_pretrained(RL)

# ---- build input sets ----
sets = {}
probs_aime = []
for f in ('/tmp/aime24.jsonl', '/tmp/aime25.jsonl'):
    probs_aime += [json.loads(l) for l in open(f)]
sets['aime'] = [tok.apply_chat_template([{'role': 'user', 'content': p['problem']}],
                                        tokenize=False, add_generation_prompt=True)
                for p in probs_aime[:10]]
for name, fn, n in [('math500', 'data/math500.jsonl', 20), ('gsm8k', 'data/gsm8k.jsonl', 20),
                    ('amc23', 'data/amc23.jsonl', 20)]:
    probs = [json.loads(l) for l in open(os.path.join(ROOT, fn))]
    sets[name] = [tok.apply_chat_template([{'role': 'user', 'content': p['problem']}],
                                          tokenize=False, add_generation_prompt=True)
                  for p in probs[:n]]
# non-math contrast sets: SimpleQA questions (factual QA prompts) + Thinking's own CoT traces (reasoning prose)
sq = [json.loads(l) for l in open(os.path.join(ROOT, 'data/simpleqa.jsonl'))][:20]
sets['simpleqa'] = [tok.apply_chat_template([{'role': 'user', 'content': q['question']}],
                                            tokenize=False, add_generation_prompt=True) for q in sq]
cot_recs = json.load(open('/tmp/own_cot_full.json'))
sets['cot'] = [tok.decode(tok(r['problem'] + '\n\n' + r['cot'])['input_ids'][:512]) for r in cot_recs[:10]]

encs = {}
for name, texts in sets.items():
    e = tok(texts, return_tensors='pt', padding=True)
    encs[name] = dict(ids=e['input_ids'].to(DEV), mask=e['attention_mask'].to(DEV),
                      N=int(e['attention_mask'].sum()))
    print(f'{name:8s}: {len(texts)} texts, {encs[name]["N"]} tokens', flush=True)

def load(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                attn_implementation='eager').to(DEV).eval()

def forward_hidden(m, enc):
    with torch.no_grad():
        o = m(enc['ids'], attention_mask=enc['mask'], output_hidden_states=True)
    return [h.detach().float().cpu() for h in o.hidden_states]

print('forward base on all sets...', flush=True)
m0 = load(BASE)
base_hs = {name: forward_hidden(m0, encs[name]) for name in sets}
del m0
torch.cuda.empty_cache()

results = {name: {t: [0.0] * 37 for t in MODELS} for name in sets}
dh_cache = {name: {} for name in sets}

for tag, path in MODELS.items():
    print(f'forward {tag}...', flush=True)
    m = load(path)
    for name in sets:
        enc = encs[name]
        with torch.no_grad():
            o = m(enc['ids'], attention_mask=enc['mask'], output_hidden_states=True)
        hs = o.hidden_states
        del o
        dhs = []
        for l in range(37):
            dh = hs[l].float() - base_hs[name][l].to(DEV)
            n2 = (dh * dh).sum(-1) * enc['mask'].float().to(DEV)
            results[name][tag][l] = float(n2.sum() / enc['N'])
            dhs.append(dh.detach().cpu())
        del hs
        torch.cuda.empty_cache()
        dh_cache[name][tag] = dhs
    del m
    torch.cuda.empty_cache()

def share(name, tag, l):
    return results[name][tag][l] / max(results[name]['full'][l], 1e-30)

out = {}
print('\nper-set global means (layer-unweighted):')
print(f'{"set":8s} {"tokens":>7s} {"f~_sub":>7s} {"f~_perp":>7s} {"f~_rand":>7s} {"inter":>7s} {"eff_sub":>7s} {"eff_perp":>7s} {"eff_rand":>7s}')
for name in sets:
    mean = lambda t: sum(share(name, t, l) for l in range(37)) / 37
    # interaction: |Dh_full - Dh_sub - Dh_perp|^2 / |Dh_full|^2 (per layer, then mean)
    inter = 0.0
    for l in range(37):
        r = dh_cache[name]['full'][l].to(DEV) - dh_cache[name]['sub'][l].to(DEV) - dh_cache[name]['perp'][l].to(DEV)
        n2 = (r * r).sum(-1) * encs[name]['mask'].float().to(DEV)
        inter += (n2.sum() / encs[name]['N']) / max(results[name]['full'][l], 1e-30)
    inter /= 37
    row = dict(f_sub=mean('sub'), f_perp=mean('perp'), f_rand=mean('rand'), inter=float(inter),
               eff_sub=mean('sub') / ENERGY['sub'], eff_perp=mean('perp') / ENERGY['perp'],
               eff_rand=mean('rand') / ENERGY['rand'])
    out[name] = row
    print(f'{name:8s} {encs[name]["N"]:7d} {row["f_sub"]*100:6.1f}% {row["f_perp"]*100:6.1f}% '
          f'{row["f_rand"]*100:6.1f}% {row["inter"]*100:6.1f}% {row["eff_sub"]:7.2f} {row["eff_perp"]:7.2f} {row["eff_rand"]:7.2f}')
json.dump(out, open(OUT, 'w'))
print('DONE', OUT, flush=True)
