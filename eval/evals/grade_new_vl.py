"""Grade the newly added MathVista-testmini and ARKitScenes runs.

This file intentionally keeps grading independent of the generation runner so
it can be run while other vLLM jobs are using the GPUs.  MathVista follows the
public testmini answer types.  ARKitScenes follows the Qwen3-VL/Omni3D
category-conditioned 9-DoF protocol and reports mAP@0.15.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError:  # pragma: no cover - the evaluation environment includes scipy
    ConvexHull = None  # type: ignore[assignment,misc]


SEEDS = (101, 202, 303, 404)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
BOXED_RE = re.compile(r"\\boxed\s*\{(.*)\}", flags=re.DOTALL)
CHOICE_RE = re.compile(r"(?i)(?:answer|option|choice|correct answer)\s*(?:is|:|=)?\s*[(\[]?\s*([A-Z])\b")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "generation_manifest.json"
    return _json(path) if path.is_file() else {}


def _generation_file(eval_dir: Path, manifest: dict[str, Any]) -> Path:
    output = manifest.get("output_file")
    if isinstance(output, str) and Path(output).is_file():
        return Path(output)
    candidates = sorted(
        path
        for path in eval_dir.glob("*.jsonl")
        if not path.name.endswith("grading_details.jsonl")
    )
    if not candidates:
        raise FileNotFoundError(f"No generation JSONL found under {eval_dir}")
    return candidates[-1]


def _data_file(manifest: dict[str, Any], eval_dir: Path) -> Path:
    value = manifest.get("data_path")
    if isinstance(value, str) and Path(value).is_file():
        return Path(value)
    value = manifest.get("dataset_path")
    if isinstance(value, str) and Path(value).is_file():
        return Path(value)
    raise FileNotFoundError(f"Generation manifest has no readable data_path: {eval_dir}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _norm_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("\\\\", "\\")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[\s.,;:!?]+$", "", text)
    return text.strip()


def _clean_response(response: Any) -> str:
    text = str(response if response is not None else "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json|text|答案)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    return text


def _boxed_or_final(response: Any) -> str:
    text = _clean_response(response)
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    answer_matches = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer|答案)\s*(?:is|:|=)?\s*(.+?)(?=\n|$)",
            text,
        )
    )
    if answer_matches:
        return answer_matches[-1].group(1).strip().strip("`* ")
    hashes = re.findall(r"(?im)^\s*####\s*(.+?)\s*$", text)
    if hashes:
        return hashes[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _choice_index(value: Any, choices: Any) -> int | None:
    if not isinstance(choices, list) or not choices:
        return None
    gold = _norm_text(value)
    for index, choice in enumerate(choices):
        if gold == _norm_text(choice):
            return index
    if len(gold) == 1 and "a" <= gold <= "z":
        index = ord(gold) - ord("a")
        return index if index < len(choices) else None
    return None


def _candidate_choice(response: Any, choices: list[Any]) -> int | None:
    text = _clean_response(response)
    final = _boxed_or_final(text)
    candidates = [final, text]
    for candidate in candidates:
        candidate_norm = _norm_text(candidate).strip("()[]{} ")
        if len(candidate_norm) == 1 and candidate_norm.isalpha():
            index = ord(candidate_norm.upper()) - ord("A")
            if 0 <= index < len(choices):
                return index
        for index, choice in enumerate(choices):
            if _norm_text(choice) == candidate_norm:
                return index
    matches = CHOICE_RE.findall(text)
    if matches:
        index = ord(matches[-1].upper()) - ord("A")
        if 0 <= index < len(choices):
            return index
    # A final parenthesized/standalone option is safer than taking the first
    # letter in a long chain-of-thought answer.
    matches = re.findall(r"(?<![A-Za-z])\(([A-Z])\)(?![A-Za-z])|(?<![A-Za-z])([A-Z])(?![A-Za-z])", text)
    for first, second in reversed(matches):
        letter = first or second
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return index
    return None


def _last_number(value: Any) -> float | None:
    matches = NUMBER_RE.findall(_boxed_or_final(value))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _parse_list(value: Any) -> Any:
    text = _boxed_or_final(value).strip()
    for candidate in (text, text.rstrip(".")):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                pass
    return None


def _mathvista_correct(row: dict[str, Any], sample: dict[str, Any]) -> tuple[bool, str]:
    response = row.get("response")
    if not isinstance(response, str) or not response.strip() or row.get("truncated"):
        return False, "invalid_or_truncated"
    answer_type = str(sample.get("answer_type") or "").lower()
    question_type = str(sample.get("question_type") or "").lower()
    gold = sample.get("answer")
    choices = sample.get("choices")
    if "choice" in answer_type or "choice" in question_type or isinstance(choices, list):
        if not isinstance(choices, list) or not choices:
            return False, "missing_choices"
        predicted = _candidate_choice(response, choices)
        expected = _choice_index(gold, choices)
        return predicted is not None and predicted == expected, "choice"
    if "integer" in answer_type:
        predicted = _last_number(response)
        try:
            expected = int(float(str(gold).replace(",", "")))
        except (TypeError, ValueError):
            return False, "bad_gold"
        return predicted is not None and int(round(predicted)) == expected, "integer"
    if "float" in answer_type or "number" in answer_type:
        predicted = _last_number(response)
        try:
            expected = float(str(gold).replace(",", "").replace("%", ""))
        except (TypeError, ValueError):
            return False, "bad_gold"
        precision = sample.get("precision")
        try:
            digits = int(precision) if precision is not None else 2
        except (TypeError, ValueError):
            digits = 2
        return predicted is not None and round(predicted, digits) == round(expected, digits), "float"
    if "list" in answer_type:
        predicted = _parse_list(response)
        expected = gold
        if predicted is not None:
            return _norm_text(predicted) == _norm_text(expected), "list"
    return _norm_text(_boxed_or_final(response)) == _norm_text(gold), "text"


def _token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["token_length"]) for row in rows if row.get("token_length") is not None]
    truncated = sum(1 for row in rows if row.get("truncated") or row.get("finish_reason") == "length")
    return {
        "num_rollouts": len(rows),
        "avg_token_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "total_tokens": sum(lengths),
        "truncated_rollouts": truncated,
    }


def grade_mathvista(eval_dir: Path) -> dict[str, Any]:
    manifest = _manifest(eval_dir)
    data_path = _data_file(manifest, eval_dir)
    samples = {str(row["example_id"]): row for row in _read_jsonl(data_path)}
    generation_path = _generation_file(eval_dir, manifest)
    rows = list(_read_jsonl(generation_path))
    scores: list[dict[str, Any]] = []
    by_seed: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        sample = samples.get(str(row.get("example_id")), {})
        correct, mode = _mathvista_correct(row, sample)
        seed = int(row.get("seed", -1))
        by_seed[seed].append(int(correct))
        scores.append({
            "example_id": row.get("example_id"),
            "seed": seed,
            "correct": bool(correct),
            "answer_mode": mode,
            "token_length": row.get("token_length"),
            "truncated": bool(row.get("truncated") or row.get("finish_reason") == "length"),
        })
    expected = int(manifest.get("num_examples") or len(samples))
    expected_seeds = [int(seed) for seed in manifest.get("seeds", SEEDS)]
    expected_total = expected * len(expected_seeds)
    accuracy = sum(item["correct"] for item in scores) / len(scores) if scores else 0.0
    seed_scores = {
        str(seed): (sum(values) / len(values) if values else 0.0)
        for seed, values in sorted(by_seed.items())
    }
    stats = _token_stats(rows)
    result = {
        "task_name": "MathVista-testmini",
        "evaluation_mode": "official testmini answer-type normalization",
        "generation_file": str(generation_path),
        "num_problems": expected,
        "expected_rollouts_per_problem": len(expected_seeds),
        "total_rollouts": len(rows),
        "expected_total_rollouts": expected_total,
        "missing_rollouts": max(0, expected_total - len(rows)),
        "accuracy": accuracy,
        "mean_score": accuracy,
        "pass1_by_seed": seed_scores,
        "pass1_mean": (sum(seed_scores.values()) / len(seed_scores)) if seed_scores else 0.0,
        **stats,
    }
    (eval_dir / "grading_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (eval_dir / f"{generation_path.stem}.grading_details.jsonl").open("w", encoding="utf-8") as handle:
        for item in scores:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return result


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ ry @ rx


def _box_corners(box: Any) -> np.ndarray | None:
    if not isinstance(box, dict):
        return None
    values = box.get("bbox_3d")
    if not isinstance(values, list) or len(values) != 9:
        return None
    try:
        center = np.asarray([float(value) for value in values[:3]], dtype=float)
        size = np.asarray([float(value) for value in values[3:6]], dtype=float)
        roll, pitch, yaw = (float(value) for value in values[6:9])
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(center)) or np.any(~np.isfinite(size)) or np.any(size <= 0):
        return None
    half = size / 2.0
    local = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=float) * half
    return local @ _rpy_matrix(roll, pitch, yaw).T + center


def _ground_truth_corners(box: dict[str, Any]) -> np.ndarray | None:
    corners = box.get("corners")
    if isinstance(corners, list) and len(corners) == 8:
        try:
            value = np.asarray(corners, dtype=float)
            if value.shape == (8, 3) and np.all(np.isfinite(value)):
                return value
        except (TypeError, ValueError):
            pass
    return _box_corners(box)


def _parse_boxes(response: Any, requested_class: str) -> list[dict[str, Any]]:
    text = _clean_response(response)
    if not text:
        return []
    candidates: list[Any] = []
    for start, end in ((text.find("["), text.rfind("]")), (text.find("{"), text.rfind("}"))):
        if start >= 0 and end > start:
            fragment = text[start : end + 1]
            try:
                candidates.append(json.loads(fragment))
            except json.JSONDecodeError:
                try:
                    candidates.append(ast.literal_eval(fragment))
                except (SyntaxError, ValueError):
                    pass
    raw: Any = candidates[0] if candidates else []
    if isinstance(raw, dict):
        for key in ("boxes", "bboxes", "predictions", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw] if "bbox_3d" in raw else []
    if not isinstance(raw, list):
        return []
    requested = _norm_text(requested_class)
    result = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("bbox_3d"), list):
            continue
        label = item.get("label", item.get("category", item.get("class_name")))
        if label is not None and _norm_text(label) not in {requested, ""}:
            continue
        if _box_corners(item) is not None:
            result.append(item)
    return result


def _planes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = corners.mean(axis=0)
    # The six local axes are recovered from the three edge directions at the
    # first corner.  ConvexHull equations are more robust for GT corners, so
    # use them and deduplicate the two triangles per face.
    if ConvexHull is None:
        raise RuntimeError("scipy is required for ARKitScenes grading")
    hull = ConvexHull(corners)
    normals: list[np.ndarray] = []
    offsets: list[float] = []
    for equation in hull.equations:
        normal = np.asarray(equation[:3], dtype=float)
        offset = float(equation[3])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal /= norm
        offset /= norm
        if not any(np.linalg.norm(normal - old_n) < 1e-6 and abs(offset - old_d) < 1e-6 for old_n, old_d in zip(normals, offsets)):
            normals.append(normal)
            offsets.append(offset)
    if len(normals) < 6:
        raise ValueError("could not recover six box planes")
    # Orient all inequalities as n dot x + d <= 0 and remove numerical drift.
    planes = np.asarray(normals[:6], dtype=float)
    ds = np.asarray(offsets[:6], dtype=float)
    if np.any(planes @ center + ds > 1e-6):
        flip = planes @ center + ds > 0
        planes[flip] *= -1.0
        ds[flip] *= -1.0
    return planes, ds, float(hull.volume)


def _oriented_iou(first: np.ndarray, second: np.ndarray) -> float:
    if ConvexHull is None:
        raise RuntimeError("scipy is required for ARKitScenes grading")
    min_first, max_first = first.min(axis=0), first.max(axis=0)
    min_second, max_second = second.min(axis=0), second.max(axis=0)
    if np.any(np.minimum(max_first, max_second) < np.maximum(min_first, min_second)):
        return 0.0
    planes_a, ds_a, volume_a = _planes(first)
    planes_b, ds_b, volume_b = _planes(second)
    planes = np.concatenate([planes_a, planes_b], axis=0)
    ds = np.concatenate([ds_a, ds_b], axis=0)
    combinations = np.asarray(list(itertools.combinations(range(len(planes)), 3)), dtype=int)
    matrices = planes[combinations]
    determinants = np.linalg.det(matrices)
    valid = np.abs(determinants) > 1e-9
    if not np.any(valid):
        return 0.0
    rhs = -np.broadcast_to(ds[combinations], (len(combinations), 3))
    vertices = np.full((len(combinations), 3), np.nan, dtype=float)
    vertices[valid] = np.linalg.solve(
        matrices[valid], rhs[valid, :, None]
    )[:, :, 0]
    inside = np.all(planes @ vertices.T + ds[:, None] <= 1e-6, axis=0)
    points = vertices[inside & np.all(np.isfinite(vertices), axis=1)]
    if len(points) < 4:
        return 0.0
    points = np.unique(np.round(points, decimals=10), axis=0)
    if len(points) < 4:
        return 0.0
    intersection = float(ConvexHull(points).volume)
    union = volume_a + volume_b - intersection
    return max(0.0, min(1.0, intersection / union)) if union > 0 else 0.0


def _average_precision(matches: list[bool], num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    tp = np.cumsum(np.asarray(matches, dtype=float))
    fp = np.cumsum(1.0 - np.asarray(matches, dtype=float))
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = np.linspace(0.0, 1.0, 101)
    values = [float(envelope[recall >= point].max()) if np.any(recall >= point) else 0.0 for point in samples]
    return float(np.mean(values))


def grade_arkitscenes(eval_dir: Path) -> dict[str, Any]:
    manifest = _manifest(eval_dir)
    data_path = _data_file(manifest, eval_dir)
    data = {str(row["example_id"]): row for row in _read_jsonl(data_path)}
    generation_path = _generation_file(eval_dir, manifest)
    rows = list(_read_jsonl(generation_path))
    per_seed_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    token_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for row in rows:
        sample = data.get(str(row.get("example_id")))
        if sample is None:
            continue
        seed = int(row.get("seed", -1))
        requested = str(sample.get("class_name", ""))
        predicted = _parse_boxes(row.get("response"), requested)
        gt = [box for box in sample.get("ground_truth_boxes", []) if isinstance(box, dict)]
        per_seed_rows[seed].append({"sample": sample, "predicted": predicted, "gt": gt, "row": row})
        token_rows.append(row)
        details.append({
            "example_id": row.get("example_id"),
            "seed": seed,
            "num_predicted": len(predicted),
            "num_ground_truth": len(gt),
            "truncated": bool(row.get("truncated") or row.get("finish_reason") == "length"),
            "token_length": row.get("token_length"),
        })

    category_ap: dict[str, dict[str, float]] = {}
    seed_maps: dict[str, float] = {}
    all_categories = sorted({str(row.get("class_name", "")) for row in data.values()})
    for seed, seed_rows in sorted(per_seed_rows.items()):
        aps: dict[str, float] = {}
        for category in all_categories:
            category_rows = [row for row in seed_rows if str(row["sample"].get("class_name")) == category]
            detections: list[tuple[float, int, np.ndarray]] = []
            gt_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
            for row in category_rows:
                image_id = int(row["sample"]["image_id"])
                for box in row["gt"]:
                    corners = _ground_truth_corners(box)
                    if corners is not None:
                        gt_by_image[image_id].append(corners)
                for order, box in enumerate(row["predicted"]):
                    corners = _box_corners(box)
                    if corners is not None:
                        detections.append((1.0, image_id, corners))
            num_gt = sum(len(value) for value in gt_by_image.values())
            detections.sort(key=lambda item: item[0], reverse=True)
            used: dict[int, set[int]] = defaultdict(set)
            matches: list[bool] = []
            for _, image_id, predicted in detections:
                best_iou, best_index = 0.0, None
                for index, target in enumerate(gt_by_image.get(image_id, [])):
                    if index in used[image_id]:
                        continue
                    iou = _oriented_iou(predicted, target)
                    if iou > best_iou:
                        best_iou, best_index = iou, index
                if best_index is not None and best_iou >= 0.15:
                    used[image_id].add(best_index)
                    matches.append(True)
                else:
                    matches.append(False)
            aps[category] = _average_precision(matches, num_gt)
        category_ap[str(seed)] = aps
        seed_maps[str(seed)] = sum(aps.values()) / len(aps) if aps else 0.0
    result = {
        "task_name": "ARKitScenes",
        "evaluation_mode": "Omni3D ARKitScenes mAP@0.15, confidence=1.0",
        "generation_file": str(generation_path),
        "num_problems": len(data),
        "expected_rollouts_per_problem": len(manifest.get("seeds", SEEDS)),
        "total_rollouts": len(rows),
        "expected_total_rollouts": len(data) * len(manifest.get("seeds", SEEDS)),
        "missing_rollouts": max(0, len(data) * len(manifest.get("seeds", SEEDS)) - len(rows)),
        "metric": "mAP@0.15",
        "accuracy": sum(seed_maps.values()) / len(seed_maps) if seed_maps else 0.0,
        "mean_score": sum(seed_maps.values()) / len(seed_maps) if seed_maps else 0.0,
        "map_by_seed": seed_maps,
        "category_ap_by_seed": category_ap,
        **_token_stats(token_rows),
    }
    (eval_dir / "grading_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (eval_dir / f"{generation_path.stem}.grading_details.jsonl").open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = _manifest(args.eval_dir)
    benchmark = str(manifest.get("benchmark", args.eval_dir.name)).lower()
    if "mathvista" in benchmark:
        result = grade_mathvista(args.eval_dir)
    elif "arkit" in benchmark:
        result = grade_arkitscenes(args.eval_dir)
    else:
        raise SystemExit(f"Unsupported benchmark in manifest: {benchmark}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
