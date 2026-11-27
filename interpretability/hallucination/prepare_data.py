#!/usr/bin/env python3
"""Download TruthfulQA and SimpleQA, convert to JSONL for vLLM evaluation."""
import json, os, random

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)


def prepare_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    records = []
    for i, row in enumerate(ds):
        q = row["question"]
        cat = row.get("category", "")

        # MC1: exactly one correct answer
        mc1 = row["mc1_targets"]
        choices_mc1 = mc1["choices"]
        labels_mc1 = mc1["labels"]

        # Shuffle choices, track correct index
        order = list(range(len(choices_mc1)))
        random.shuffle(order)
        shuffled = [choices_mc1[j] for j in order]
        correct_idx = order.index(labels_mc1.index(1))
        correct_letter = chr(ord("A") + correct_idx)

        # MC2: multiple correct answers possible
        mc2 = row["mc2_targets"]
        choices_mc2 = mc2["choices"]
        labels_mc2 = mc2["labels"]
        order2 = list(range(len(choices_mc2)))
        random.shuffle(order2)
        shuffled2 = [choices_mc2[j] for j in order2]
        correct_set2 = [chr(ord("A") + order2.index(j))
                        for j, lbl in enumerate(labels_mc2) if lbl == 1]

        records.append({
            "id": f"tqa-{i}",
            "question": q,
            "category": cat,
            "mc1_choices": shuffled,
            "mc1_correct": correct_letter,
            "mc2_choices": shuffled2,
            "mc2_correct": sorted(correct_set2),
        })

    path = os.path.join(OUT, "truthfulqa_mc.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TruthfulQA: {len(records)} questions -> {path}")
    return records


def prepare_simpleqa():
    """SimpleQA from OpenAI — short factual questions with definite answers."""
    import urllib.request
    url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    csv_path = os.path.join(OUT, "simpleqa_raw.csv")
    if not os.path.exists(csv_path):
        print(f"Downloading SimpleQA from {url} ...")
        urllib.request.urlretrieve(url, csv_path)
    import csv
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            records.append({
                "id": f"sqa-{i}",
                "topic": row.get("metadata", ""),
                "question": row["problem"],
                "answer": row["answer"],
            })
    path = os.path.join(OUT, "simpleqa.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SimpleQA: {len(records)} questions -> {path}")
    return records


if __name__ == "__main__":
    prepare_truthfulqa()
    prepare_simpleqa()
    print("Done.")
