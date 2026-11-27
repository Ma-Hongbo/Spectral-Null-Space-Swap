#!/usr/bin/env python3
"""Functional-norm shares + random-subspace control (README protocol).
Input: AIME24+25 first 10 problems, chat template, no generation.
Models: base=W0, sub=W0+dW_par, perp=W0+dW_perp, full=W0+dW, rand=W0+P_R(dW)
(random same-dim as S). Per layer l: f~_* = |Dh_*^(l)|^2 / |Dh_full^(l)|^2
(per-token mean squared norm); interaction = |Dh_full - Dh_sub - Dh_perp|^2/|Dh_full|^2.
Hidden states streamed to CPU per layer."""
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
MODELS = dict(base=BASE,
              sub=os.path.join(ROOT, "models/subonly_2507"),
              perp=os.path.join(ROOT, "models/perponly_2507"),
              full=RL,
              rand=os.path.join(ROOT, "models/randpar_2507"))
DEV = 'cuda:0'
OUT = os.path.join(ROOT, 'functional_rand_control.json')

tok = AutoTokenizer.from_pretrained(RL)
probs = []
for f in ('/tmp/aime24.jsonl', '/tmp/aime25.jsonl'):
    probs += [json.loads(l) for l in open(f)]
probs = probs[:10]
texts = [tok.apply_chat_template([{'role': 'user', 'content': p['problem']}],
                                 tokenize=False, add_generation_prompt=True) for p in probs]
enc = tok(texts, return_tensors='pt', padding=True)
ids, mask = enc['input_ids'].to(DEV), enc['attention_mask'].to(DEV)
N = int(mask.sum())
print(f'{len(probs)} problems, {N} tokens', flush=True)

def load(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                attn_implementation='eager').to(DEV).eval()

print('forward base...', flush=True)
m0 = load(BASE)
with torch.no_grad():
    o0 = m0(ids, attention_mask=mask, output_hidden_states=True)
base_hs = [h.detach().float().cpu() for h in o0.hidden_states]
del o0, m0
torch.cuda.empty_cache()
print(f'cached {len(base_hs)} layers x {base_hs[0].shape}', flush=True)

results = {t: [0.0] * len(base_hs) for t in ['sub', 'perp', 'full', 'rand']}
inter = [0.0] * len(base_hs)
dh_cache = {'sub': None, 'perp': None, 'full': None}

for tag, path in MODELS.items():
    if tag == 'base':
        continue
    print(f'forward {tag}...', flush=True)
    m = load(path)
    with torch.no_grad():
        o = m(ids, attention_mask=mask, output_hidden_states=True)
    hs = o.hidden_states
    del o, m
    torch.cuda.empty_cache()
    # per layer: stream to CPU and accumulate
    dhs = []
    for l in range(len(base_hs)):
        dh = hs[l].float() - base_hs[l].to(DEV)        # (B, T, d), both on GPU
        n2 = (dh * dh).sum(-1) * mask.float().to(DEV)
        results[tag][l] = float(n2.sum() / N)
        dhs.append(dh.detach().cpu())
    del hs
    torch.cuda.empty_cache()
    dh_cache[tag] = dhs
    print(f'  {tag}: mean f~ = {sum(results[tag]) / len(base_hs):.4f}', flush=True)

# interaction: |Dh_full - Dh_sub - Dh_perp|^2 / |Dh_full|^2
for l in range(len(base_hs)):
    r = (dh_cache['full'][l] - dh_cache['sub'][l] - dh_cache['perp'][l]).float().to(DEV)
    inter[l] = float(((r * r).sum(-1) * mask.float().to(DEV)).sum() / N)

def share(tag, l):
    return results[tag][l] / max(results['full'][l], 1e-30)

print('\nper-layer shares f~ (vs full):')
print(f'{"layer":>6} {"f~_sub":>8} {"f~_perp":>8} {"f~_rand":>8} {"inter":>8}')
for l in range(len(base_hs)):
    print(f'{l:6d} {share("sub", l)*100:7.1f}% {share("perp", l)*100:7.1f}% '
          f'{share("rand", l)*100:7.1f}% {inter[l]/max(results["full"][l],1e-30)*100:7.1f}%')
mean = lambda t: sum(share(t, l) for l in range(len(base_hs))) / len(base_hs)
mi = sum(inter[l] / max(results['full'][l], 1e-30) for l in range(len(base_hs))) / len(base_hs)
print(f'\nGLOBAL means: f~_sub = {mean("sub")*100:.1f}%   f~_perp = {mean("perp")*100:.1f}%   '
      f'f~_rand = {mean("rand")*100:.1f}%   interaction = {mi*100:.1f}%')
json.dump(dict(n_tokens=N, results=results, interaction=inter), open(OUT, 'w'))
print('DONE', OUT, flush=True)
