"""Generate CoT on AIME with Qwen3-4B-Thinking-2507 (the paper's Full model).

The traces are used only as a FIXED input text for the Pythia entropy sweep, matching the
paper's protocol (10 problems x ~980 CoT positions) instead of raw problem statements.
"""
import os, json, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

os.environ["HF_HUB_OFFLINE"] = "1"

M = sorted(glob.glob(f"{HF_CACHE}/hub/"
                     "models--Qwen--Qwen3-4B-Thinking-2507/snapshots/*"))[-1]
NPROB, MAXNEW = 10, 4608
probs = []
for f in ("/tmp/aime24.jsonl", "/tmp/aime25.jsonl"):
    probs += [json.loads(l) for l in open(f)]
probs = probs[:NPROB]
print(f"model {M}\n{len(probs)} problems", flush=True)

tok = AutoTokenizer.from_pretrained(M, padding_side="left")
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda:0").eval()

texts = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}],
                                 tokenize=False, add_generation_prompt=True) for p in probs]
enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
print("prompt batch:", tuple(enc.input_ids.shape), flush=True)

with torch.no_grad():
    out = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=0.6,
                     top_p=0.95, pad_token_id=tok.pad_token_id or tok.eos_token_id)
gen = out[:, enc.input_ids.shape[1]:]

recs = []
for i, p in enumerate(probs):
    cot = tok.decode(gen[i], skip_special_tokens=True)
    # keep the thinking segment only, if the model closed it
    cot = cot.split("</think>")[0]
    recs.append(dict(id=p["id"], problem=p["problem"], answer=p["answer"],
                     cot=cot, n_gen_tok=int((gen[i] != (tok.pad_token_id or 0)).sum())))
    print(f"  {p['id']}: {len(cot)} chars, {recs[-1]['n_gen_tok']} gen tokens", flush=True)

json.dump(recs, open("/tmp/aime_cot.json", "w"))
print("WROTE /tmp/aime_cot.json  total chars:", sum(len(r["cot"]) for r in recs), flush=True)
