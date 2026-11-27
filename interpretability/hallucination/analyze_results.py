#!/usr/bin/env python3
"""Analyze hallucination evaluation results.
Reads output JSON files from eval_truthfulqa.py and eval_simpleqa.py.
Generates summary tables and per-category breakdowns.

Usage:
  python analyze_results.py --dir /path/to/hallucination_eval/
  python analyze_results.py --dir .   # if results are in current dir
"""
import argparse, json, os, sys
from collections import defaultdict

MODELS = ["base", "subonly", "perponly", "full"]
MODEL_DISPLAY = {
    "base": "Base (W₀)",
    "subonly": "Sub (W₀+ΔW∥)",
    "perponly": "Perp (W₀+ΔW⊥)",
    "full": "Full (W₀+ΔW)",
}


def load_result(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None


def analyze_truthfulqa(results_dir):
    print("\n" + "=" * 70)
    print("TruthfulQA MC1 Results")
    print("=" * 70)

    data = {}
    for m in MODELS:
        r = load_result(os.path.join(results_dir, f"truthfulqa_{m}.json"))
        if r:
            data[m] = r

    if not data:
        print("  No TruthfulQA results found.")
        return

    # Overall accuracy
    print(f"\n{'Model':<25s} {'MC1 Acc':>10s} {'Correct':>10s} {'Total':>8s} "
          f"{'Parsed':>8s} {'Avg Len':>10s}")
    print("-" * 75)
    for m in MODELS:
        if m not in data:
            continue
        r = data[m]
        avg_len = sum(rec["len_tokens"] for rec in r["records"]) / len(r["records"])
        print(f"{MODEL_DISPLAY[m]:<25s} {r['mc1_accuracy']:>10.4f} "
              f"{r['n_correct']:>10d} {r['n_total']:>8d} "
              f"{r['n_parsed']:>8d} {avg_len:>10.0f}")

    # Per-category breakdown (if categories exist)
    cats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for m in MODELS:
        if m not in data:
            continue
        for rec in data[m]["records"]:
            cat = rec.get("category", "unknown") or "unknown"
            cats[cat][m]["total"] += 1
            if rec["correct"]:
                cats[cat][m]["correct"] += 1

    if len(cats) > 1:
        print(f"\nPer-category MC1 accuracy:")
        print(f"{'Category':<30s}", end="")
        for m in MODELS:
            if m in data:
                print(f" {m:>10s}", end="")
        print()
        print("-" * (30 + 11 * len(data)))
        for cat in sorted(cats.keys()):
            print(f"{cat[:29]:<30s}", end="")
            for m in MODELS:
                if m in data:
                    c = cats[cat][m]
                    if c["total"] > 0:
                        acc = c["correct"] / c["total"]
                        print(f" {acc:>10.3f}", end="")
                    else:
                        print(f" {'—':>10s}", end="")
            print()

    return data


def analyze_simpleqa(results_dir):
    print("\n" + "=" * 70)
    print("SimpleQA Results")
    print("=" * 70)

    data = {}
    for m in MODELS:
        r = load_result(os.path.join(results_dir, f"simpleqa_{m}.json"))
        if r:
            data[m] = r

    if not data:
        print("  No SimpleQA results found.")
        return

    print(f"\n{'Model':<25s} {'Accuracy':>10s} {'Correct':>9s} {'Incorrect':>10s} "
          f"{'Not Att.':>9s} {'Total':>7s} {'Avg Len':>9s}")
    print("-" * 85)
    for m in MODELS:
        if m not in data:
            continue
        r = data[m]
        c = r["counts"]
        avg_len = sum(rec["len_tokens"] for rec in r["records"]) / len(r["records"])
        print(f"{MODEL_DISPLAY[m]:<25s} {r['accuracy']:>10.4f} "
              f"{c['CORRECT']:>9d} {c['INCORRECT']:>10d} "
              f"{c['NOT_ATTEMPTED']:>9d} {r['n_total']:>7d} {avg_len:>9.0f}")

    # Hallucination rate = INCORRECT / (CORRECT + INCORRECT)
    # (excluding NOT_ATTEMPTED)
    print(f"\nHallucination rate (incorrect / attempted):")
    for m in MODELS:
        if m not in data:
            continue
        c = data[m]["counts"]
        attempted = c["CORRECT"] + c["INCORRECT"]
        if attempted > 0:
            hall_rate = c["INCORRECT"] / attempted
            print(f"  {MODEL_DISPLAY[m]:<25s}: {hall_rate:.4f} "
                  f"({c['INCORRECT']}/{attempted})")

    return data


def print_latex_table(tqa_data, sqa_data):
    """Print LaTeX-ready comparison table."""
    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Hallucination evaluation results.}")
    print(r"\label{tab:hallucination}")
    print(r"\begin{tabular}{l cc cc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{TruthfulQA MC1} "
          r"& \multicolumn{2}{c}{SimpleQA} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r"Model & Acc & CoT Len & Acc & Hall. Rate \\")
    print(r"\midrule")

    for m in MODELS:
        name_tex = {
            "base": r"Base ($W_0$)",
            "subonly": r"Sub ($W_0\!+\!\Delta W_\parallel$)",
            "perponly": r"Perp ($W_0\!+\!\Delta W_\perp$)",
            "full": r"Full ($W_0\!+\!\Delta W$)",
        }[m]
        parts = [name_tex]

        if tqa_data and m in tqa_data:
            r = tqa_data[m]
            avg_len = sum(rec["len_tokens"] for rec in r["records"]) / len(r["records"])
            parts.append(f"& {r['mc1_accuracy']:.3f} & {avg_len:.0f}")
        else:
            parts.append("& — & —")

        if sqa_data and m in sqa_data:
            r = sqa_data[m]
            c = r["counts"]
            attempted = c["CORRECT"] + c["INCORRECT"]
            hall = c["INCORRECT"] / attempted if attempted else 0
            parts.append(f"& {r['accuracy']:.3f} & {hall:.3f}")
        else:
            parts.append("& — & —")

        parts.append(r"\\")
        print(" ".join(parts))

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with result JSON files")
    args = ap.parse_args()

    tqa = analyze_truthfulqa(args.dir)
    sqa = analyze_simpleqa(args.dir)

    if tqa or sqa:
        print_latex_table(tqa, sqa)


if __name__ == "__main__":
    main()
