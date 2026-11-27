"""Prepare local, reproducible audio JSONL inputs for the Omni benchmarks.

The benchmark archives contain audio bytes in parquet/tar files.  This script
normalizes them to one JSON object per item and writes local audio files so
the vLLM runner can decode/resample them without depending on the datasets
library inside the inference environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/home/hongbo/sar-reproduction")
DEFAULT_AUDIO_ROOT = Path("/data/hongbo/audio_benchmarks")
DEFAULT_PREPARED = DEFAULT_AUDIO_ROOT / "prepared"
MMAR_REPO = Path("/tmp/sar-audio-eval-20260826/MMAR")


def _clean_choice(value: Any) -> str:
    text = str(value).strip()
    return re.sub(r"^\s*\(?[A-Ja-j]\)?[.)：:\-]\s*", "", text).strip()


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(text).lower())).strip()


def _answer_index(answer: Any, choices: list[str]) -> int | None:
    raw = str(answer).strip()
    match = re.match(r"^\s*\(?([A-Ja-j])\)?(?:[.)：:\-]|\s|$)", raw)
    if match:
        index = ord(match.group(1).upper()) - ord("A")
        if 0 <= index < len(choices):
            return index
    norm = _normalize(raw)
    exact = [index for index, choice in enumerate(choices) if _normalize(choice) == norm]
    if len(exact) == 1:
        return exact[0]
    partial = [
        index
        for index, choice in enumerate(choices)
        if norm and (_normalize(choice) in norm or norm in _normalize(choice))
    ]
    return partial[0] if len(partial) == 1 else None


def _safe_relative(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative.removeprefix("./")).resolve()
    path.relative_to(root)
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], metadata: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    manifest = path.with_name("dataset_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": metadata,
                "num_examples": count,
                "jsonl": str(path),
                "audio_root": str((path.parent / "audio").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return count


def prepare_mmau(source: Path, output_dir: Path) -> int:
    import pyarrow.parquet as pq

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)

    def rows() -> Iterable[dict[str, Any]]:
        index = 0
        for batch in parquet.iter_batches(batch_size=32):
            for item in batch.to_pylist():
                context = item["context"] or {}
                source_id = json.loads(item.get("other_attributes") or "{}").get("id")
                source_id = str(source_id or context.get("path") or index)
                audio_bytes = context.get("bytes")
                if not audio_bytes:
                    raise ValueError(f"MMAU row {source_id} has no embedded audio bytes")
                audio_path = audio_dir / f"{source_id}.wav"
                if not audio_path.exists():
                    audio_path.write_bytes(audio_bytes)
                choices = [_clean_choice(choice) for choice in (item.get("choices") or [])]
                answer = str(item.get("answer", "")).strip()
                attrs = json.loads(item.get("other_attributes") or "{}")
                yield {
                    "example_id": source_id,
                    "source_id": source_id,
                    "audio_path": str(audio_path.resolve()),
                    "question": str(item.get("instruction", "")).strip(),
                    "choices": choices,
                    "answer": answer,
                    "answer_index": _answer_index(answer, choices),
                    "answer_letter": (
                        chr(ord("A") + _answer_index(answer, choices))
                        if _answer_index(answer, choices) is not None
                        else None
                    ),
                    "dataset": attrs,
                    "dataset_split": "test_mini",
                }
                index += 1

    return _write_jsonl(
        output_dir / "samples.jsonl",
        rows(),
        {
            "name": "MMAU",
            "repository": "gamma-lab-umd/MMAU-test-mini",
            "requested_split": "test_mini",
            "effective_split": "test_mini",
            "official_size": 1000,
            "source_file": str(source),
        },
    )


def extract_mmar(archive: Path, target: Path) -> Path:
    audio_root = target / "extracted"
    marker = audio_root / ".complete"
    if marker.exists():
        return audio_root
    audio_root.mkdir(parents=True, exist_ok=True)
    root = audio_root.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            candidate = (root / member.name).resolve()
            candidate.relative_to(root)
        handle.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")
    return audio_root


def prepare_mmar(metadata_path: Path, archive: Path, output_dir: Path) -> int:
    audio_root = extract_mmar(archive, output_dir.parent)
    rows: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            choices = [_clean_choice(choice) for choice in item.get("choices", [])]
            answer = str(item.get("answer", "")).strip()
            relative = str(item.get("audio_path", ""))
            audio_path = _safe_relative(audio_root, relative)
            if not audio_path.is_file():
                raise FileNotFoundError(f"MMAR audio missing at line {line_number}: {audio_path}")
            index = _answer_index(answer, choices)
            if index is None:
                raise ValueError(f"MMAR answer does not match choices at line {line_number}")
            rows.append(
                {
                    "example_id": str(item["id"]),
                    "source_id": str(item["id"]),
                    "audio_path": str(audio_path),
                    "question": str(item.get("question", "")).strip(),
                    "choices": choices,
                    "answer": answer,
                    "answer_index": index,
                    "answer_letter": chr(ord("A") + index),
                    "dataset": {
                        key: item.get(key)
                        for key in (
                            "modality", "category", "sub-category", "language",
                            "source", "url", "timestamp", "thinking", "cue", "rubric",
                        )
                    },
                    "dataset_split": "test",
                }
            )
    return _write_jsonl(
        output_dir / "samples.jsonl",
        rows,
        {
            "name": "MMAR",
            "repository": "BoJack/MMAR",
            "requested_split": "test",
            "effective_split": "test",
            "official_size": 1000,
            "metadata_file": str(metadata_path),
            "audio_archive": str(archive),
        },
    )


def prepare_mmsu(parquet_paths: list[Path], output_dir: Path, sample_seed: int, sample_size: int) -> int:
    import pyarrow.parquet as pq

    total = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    if sample_size > total:
        raise ValueError(f"requested {sample_size} MMSU rows, only {total} available")
    selected = set(random.Random(sample_seed).sample(range(total), sample_size))
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=32):
            for item in batch.to_pylist():
                row_index = offset
                offset += 1
                if row_index not in selected:
                    continue
                audio = item.get("audio") or {}
                audio_bytes = audio.get("bytes")
                if not audio_bytes:
                    raise ValueError(f"MMSU row {row_index} has no embedded audio bytes")
                source_id = str(item["id"])
                suffix = Path(str(audio.get("path") or "")).suffix or ".wav"
                audio_path = audio_dir / f"{source_id}{suffix}"
                if not audio_path.exists():
                    audio_path.write_bytes(audio_bytes)
                choices = [str(item.get(key, "")).strip() for key in ("choice_a", "choice_b", "choice_c", "choice_d")]
                answer = str(item.get("answer_gt", "")).strip()
                answer_index = _answer_index(answer, choices)
                if answer_index is None:
                    raise ValueError(f"MMSU answer does not match choices: {source_id}")
                rows.append(
                    {
                        "example_id": source_id,
                        "source_id": source_id,
                        "audio_path": str(audio_path.resolve()),
                        "question": str(item.get("question", "")).strip(),
                        "choices": choices,
                        "answer": answer,
                        "answer_index": answer_index,
                        "answer_letter": chr(ord("A") + answer_index),
                        "dataset": {
                            key: item.get(key)
                            for key in (
                                "task_name", "category", "sub-category",
                                "sub-sub-category", "linguistics_sub_discipline",
                            )
                        },
                        "dataset_split": "train",
                    }
                )
    rows.sort(key=lambda row: row["example_id"])
    if len(rows) != sample_size:
        raise RuntimeError(f"MMSU selected {len(rows)} rows, expected {sample_size}")
    return _write_jsonl(
        output_dir / "samples.jsonl",
        rows,
        {
            "name": "MMSU",
            "repository": "ddwang2000/MMSU",
            "requested_split": "test_mini",
            "effective_split": "custom_test_mini_sample1000",
            "official_size": total,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "sampling_policy": f"random.Random({sample_seed}).sample(range({total}), {sample_size})",
            "source_files": [str(path) for path in parquet_paths],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--sample-seed", type=int, default=202)
    parser.add_argument("--mmsu-sample-size", type=int, default=1000)
    parser.add_argument("--only", choices=("MMAU", "MMAR", "MMSU", "all"), default="all")
    args = parser.parse_args()
    audio_root = args.audio_root.resolve()
    prepared_root = args.prepared_root.resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)

    if args.only in ("MMAU", "all"):
        count = prepare_mmau(
            audio_root / "MMAU-test-mini" / "test_mini.parquet",
            prepared_root / "MMAU",
        )
        print(f"MMAU prepared rows={count}", flush=True)
    if args.only in ("MMAR", "all"):
        count = prepare_mmar(
            MMAR_REPO / "MMAR-meta.jsonl",
            audio_root / "MMAR" / "mmar-audio.tar.gz",
            prepared_root / "MMAR",
        )
        print(f"MMAR prepared rows={count}", flush=True)
    if args.only in ("MMSU", "all"):
        count = prepare_mmsu(
            [
                audio_root / "MMSU" / "data" / "train-00000-of-00003.parquet",
                audio_root / "MMSU" / "data" / "train-00001-of-00003.parquet",
                audio_root / "MMSU" / "data" / "train-00002-of-00003.parquet",
            ],
            prepared_root / "MMSU",
            args.sample_seed,
            args.mmsu_sample_size,
        )
        print(f"MMSU prepared rows={count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
