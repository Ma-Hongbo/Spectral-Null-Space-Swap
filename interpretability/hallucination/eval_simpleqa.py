#!/usr/bin/env python3
"""SimpleQA evaluation with vLLM.
Short factual QA — model generates answer, graded by string matching.
For paper-quality results, add LLM judge grading afterward.
"""
import argparse, json, os, re, time, unicodedata
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = (
    "Answer the following question with a brief, factual response. "
    "Be concise and precise.\n\n"
    "Question: {question}\n\n"
    "Put your final answer in \\boxed{{}}."
)


def extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def normalize_answer(s):
    """Normalize for comparison: lowercase, strip punct/articles/whitespace."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def grade(pred_raw, gold_raw):
    """Grade with multiple strategies. Returns (grade, method)."""
    if pred_raw is None:
        return "NOT_ATTEMPTED", "no_parse"
    pred = normalize_answer(pred_raw)
    gold = normalize_answer(gold_raw)
    if not pred:
        return "NOT_ATTEMPTED", "empty"
    if pred == gold:
        return "CORRECT", "exact"
    if gold in pred or pred in gold:
        return "CORRECT", "contains"
    # Check if gold tokens are all in pred
    gold_tokens = set(gold.split())
    pred_tokens = set(pred.split())
    if gold_tokens and gold_tokens.issubset(pred_tokens):
        return "CORRECT", "token_subset"
    return "INCORRECT", "mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-frac", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    problems = []
    for line in open(args.data):
        problems.append(json.loads(line))
    if args.limit:
        problems = problems[:args.limit]
    print(f"Loaded {len(problems)} SimpleQA questions", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in problems:
        content = PROMPT_TEMPLATE.format(question=r["question"])
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True))

    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_frac, max_model_len=args.max_model_len,
              seed=args.seed, trust_remote_code=True, swap_space=8,
              kv_cache_dtype=args.kv_cache_dtype,
              disable_custom_all_reduce=args.tp > 1)

    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    records = []
    counts = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
    for r, o in zip(problems, outs):
        text = o.outputs[0].text
        pred_raw = extract_boxed(text)
        g, method = grade(pred_raw, r["answer"])
        counts[g] += 1
        records.append({
            "id": r["id"],
            "topic": r.get("topic", ""),
            "gold": r["answer"],
            "pred": pred_raw,
            "grade": g,
            "method": method,
            "len_tokens": len(o.outputs[0].token_ids),
            "truncated": o.outputs[0].finish_reason == "length",
        })

    n = len(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result = {
        "model": args.model,
        "benchmark": "simpleqa",
        "config": vars(args),
        "elapsed_s": elapsed,
        "accuracy": counts["CORRECT"] / n if n else 0,
        "not_attempted": counts["NOT_ATTEMPTED"] / n if n else 0,
        "n_total": n,
        "counts": counts,
        "records": records,
    }
    json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"Accuracy: {counts['CORRECT']}/{n} = {result['accuracy']:.4f}")
    print(f"Not attempted: {counts['NOT_ATTEMPTED']}/{n}")
    print(f"Incorrect: {counts['INCORRECT']}/{n}")
    print(f"Elapsed: {elapsed:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
