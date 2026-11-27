"""Grade Omni audio generations by their final multiple-choice option.

The generation JSONL keeps the complete model response.  This grader only
extracts the final option for accuracy, so reasoning and token usage remain
available for later analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


OPTION_RE = re.compile(r"(?<![A-Za-z])\(?([A-Ja-j])\)?(?:[.)：:]|\b)")
FINAL_RE = re.compile(
    r"(?:final\s+answer|final|answer|choice|option)\s*(?:is|:|=|-)?\s*\(?([A-Ja-j])\)?\b",
    flags=re.IGNORECASE,
)


def _extract_option(response: str, choices: list[str]) -> str | None:
    text = str(response or "").strip()
    if not text:
        return None
    tail = text[-1200:]
    matches = list(FINAL_RE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper()
    # Most Qwen responses finish with a standalone letter after </think>.
    options = list(OPTION_RE.finditer(tail))
    if options:
        return options[-1].group(1).upper()
    lowered = tail.lower()
    found = [
        chr(ord("A") + index)
        for index, choice in enumerate(choices)
        if str(choice).strip().lower() and str(choice).strip().lower() in lowered
    ]
    return found[-1] if len(found) == 1 else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [int(row.get("token_length", 0) or 0) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
        "token_usage": {
            "sum": sum(tokens),
            "mean": statistics.mean(tokens) if tokens else 0.0,
            "median": statistics.median(tokens) if tokens else 0.0,
            "min": min(tokens) if tokens else 0,
            "max": max(tokens) if tokens else 0,
        },
        "truncated": sum(bool(row.get("truncated")) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-details", action="store_true")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.parent / "grading_results.json"
    )
    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"no generation rows found in {input_path}")

    details: list[dict[str, Any]] = []
    for row in rows:
        choices = [str(choice) for choice in row.get("choices", [])]
        predicted = _extract_option(str(row.get("response", "")), choices)
        answer_index = row.get("answer_index")
        gold = (
            chr(ord("A") + int(answer_index)).upper()
            if answer_index is not None
            else str(row.get("answer_letter", "")).upper() or None
        )
        detail = {
            "example_id": row.get("example_id"),
            "seed": row.get("seed"),
            "predicted_option": predicted,
            "gold_option": gold,
            "correct": bool(predicted and gold and predicted == gold),
            "token_length": int(row.get("token_length", 0) or 0),
            "finish_reason": row.get("finish_reason"),
            "truncated": bool(row.get("truncated")),
        }
        details.append(detail)

    by_seed: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_seed[str(detail.get("seed"))].append(detail)
    overall = _stats(details)
    payload = {
        "schema_version": 1,
        "input": str(input_path),
        "benchmark": rows[0].get("benchmark"),
        "grading": {
            "method": "final_option_letter",
            "preserves_full_response": True,
        },
        "files": [
            {
                "file": str(input_path),
                "total_rollouts": len(rows),
                **overall,
                "per_seed": {
                    seed: _stats(seed_rows)
                    for seed, seed_rows in sorted(by_seed.items())
                },
            }
        ],
        "overall": overall,
        "updated_at": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_details:
        details_path = output_path.with_name("grading_details.jsonl")
        with details_path.open("w", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(
        f"[omni-audio-grade] input={input_path} total={len(rows)} "
        f"correct={overall['correct']} accuracy={overall['accuracy']:.4f} "
        f"mean_tokens={overall['token_usage']['mean']:.1f}",
        flush=True,
    )
    for seed, seed_rows in sorted(by_seed.items()):
        metric = _stats(seed_rows)
        print(
            f"[omni-audio-grade] seed={seed} total={metric['total']} "
            f"correct={metric['correct']} accuracy={metric['accuracy']:.4f} "
            f"mean_tokens={metric['token_usage']['mean']:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
