"""Prepare the two newly requested Qwen3-VL benchmark snapshots.

The generation runner consumes JSONL rows with OpenAI-style multimodal
messages.  This adapter keeps the source annotations beside each row and uses
``file://`` image URLs so the image bytes are not duplicated in every JSONL
row.  vLLM is given the corresponding ``--allowed-local-media-path`` at run
time.

MathVista uses the public ``testmini`` split from AI4Math.  ARKitScenes follows
the Qwen3-VL technical-report protocol: the ARKitScenes portion of Omni3D,
category-conditioned 9-DoF 3D-box generation, and mAP@0.15 grading.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import mimetypes
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


def _builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(v) for v in value]
    return value


def _json_answer(value: Any) -> str:
    value = _builtin(value)
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _image_suffix(data: bytes | None, fallback: str = ".jpg") -> str:
    if data:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
    return fallback if fallback.startswith(".") else f".{fallback}"


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _file_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            yield {str(k): _builtin(v) for k, v in row.items()}


def _image_bytes_from_row(row: dict[str, Any]) -> tuple[bytes | None, str | None]:
    for key in ("decoded_image", "image"):
        value = row.get(key)
        if isinstance(value, dict):
            data = value.get("bytes")
            path = value.get("path")
            if isinstance(data, (bytes, bytearray)):
                return bytes(data), str(path) if path else None
            if isinstance(data, str):
                try:
                    return base64.b64decode(data), str(path) if path else None
                except Exception:
                    pass
            if path:
                return None, str(path)
        elif isinstance(value, (bytes, bytearray)):
            return bytes(value), None
        elif isinstance(value, str):
            return None, value
    return None, None


def _extract_image_from_zip(
    image_zip: Path,
    relative_name: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = Path(relative_name).name
    target = output_dir / target_name
    if target.is_file() and target.stat().st_size > 0:
        return target
    with zipfile.ZipFile(image_zip) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if Path(name).name == target_name or name.endswith(relative_name)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Image {relative_name!r} was not found in {image_zip}"
            )
        with archive.open(candidates[0]) as source, target.open("wb") as sink:
            sink.write(source.read())
    return target


def _mathvista_prompt(row: dict[str, Any]) -> str:
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    question = str(row.get("question", "")).strip()
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        return f"{question}\n{options}\nAnswer with the option letter only."
    return question


def prepare_mathvista(args: argparse.Namespace) -> dict[str, Any]:
    parquet = Path(args.mathvista_parquet).expanduser()
    if not parquet.is_file():
        raise FileNotFoundError(f"MathVista parquet not found: {parquet}")
    image_dir = Path(args.mathvista_image_dir).expanduser()
    image_dir.mkdir(parents=True, exist_ok=True)
    image_zip = Path(args.mathvista_images_zip).expanduser() if args.mathvista_images_zip else None

    def rows() -> Iterable[dict[str, Any]]:
        for index, row in enumerate(_read_parquet_rows(parquet)):
            if args.limit and index >= args.limit:
                break
            pid = str(row.get("pid") or row.get("id") or index)
            image_bytes, image_name = _image_bytes_from_row(row)
            if image_bytes:
                image_path = image_dir / f"{pid}{_image_suffix(image_bytes)}"
                if not image_path.is_file() or image_path.stat().st_size != len(image_bytes):
                    image_path.write_bytes(image_bytes)
            elif image_name and image_zip:
                image_path = _extract_image_from_zip(image_zip, image_name, image_dir)
            else:
                raise FileNotFoundError(
                    f"MathVista row {pid} has no embedded image and no image zip was supplied"
                )
            metadata = _builtin(row.get("metadata"))
            yield {
                "example_id": pid,
                "source_id": pid,
                "question": str(row.get("question", "")),
                "query": _mathvista_prompt(row),
                "answer": _json_answer(row.get("answer", "")),
                "choices": _builtin(row.get("choices")),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
                "precision": row.get("precision"),
                "metadata": metadata,
                "image_path": str(image_path.resolve()),
                "messages": _messages(image_path, _mathvista_prompt(row)),
            }

    count = _write_jsonl(Path(args.mathvista_output), rows())
    manifest = {
        "benchmark": "MathVista-testmini",
        "source": "AI4Math/MathVista",
        "split": "testmini",
        "num_examples": count,
        "data_file": str(parquet.resolve()),
        "image_root": str(image_dir.resolve()),
        "answer_protocol": "official MathVista testmini normalization",
    }
    manifest_path = Path(args.mathvista_output).with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _rotation_to_rpy(rotation: Any) -> list[float]:
    """Extract roll, pitch, yaw for R=Rz(roll)Ry(yaw)Rx(pitch)."""
    matrix = [[float(rotation[r][c]) for c in range(3)] for r in range(3)]
    value = max(-1.0, min(1.0, -matrix[2][0]))
    yaw = math.asin(value)
    cos_yaw = math.cos(yaw)
    if abs(cos_yaw) > 1e-7:
        pitch = math.atan2(matrix[2][1], matrix[2][2])
        roll = math.atan2(matrix[1][0], matrix[0][0])
    else:
        pitch = 0.0
        roll = math.atan2(-matrix[0][1], matrix[1][1])
    return [roll, pitch, yaw]


def _annotation_box(annotation: dict[str, Any]) -> dict[str, Any] | None:
    if not annotation.get("valid3D", True):
        return None
    center = annotation.get("center_cam")
    dimensions = annotation.get("dimensions")
    rotation = annotation.get("R_cam")
    if not (isinstance(center, list) and len(center) == 3):
        return None
    if not (isinstance(dimensions, list) and len(dimensions) == 3):
        return None
    if not (isinstance(rotation, list) and len(rotation) == 3):
        return None
    # Omni3D stores dimensions as [width, height, length], while the
    # Qwen3-VL 9-DoF prompt uses the local box axes [x_size, y_size, z_size]
    # corresponding to [length, height, width].  The swap is required to
    # reconstruct the published bbox3D_cam corners exactly.
    qwen_dimensions = [float(dimensions[2]), float(dimensions[1]), float(dimensions[0])]
    result = {
        "bbox_3d": [
            *[float(x) for x in center],
            *qwen_dimensions,
            *_rotation_to_rpy(rotation),
        ],
        "label": str(annotation.get("category_name", "")),
    }
    corners = annotation.get("bbox3D_cam")
    if isinstance(corners, list) and len(corners) == 8:
        result["corners"] = _builtin(corners)
    return result


def _find_image(image_root: Path, file_path: str) -> Path | None:
    direct = image_root / file_path
    if direct.is_file():
        return direct
    # Omni3D's archive has used both ``arkitscenes/...`` and a bare relative
    # path in different releases.  Resolve by basename as a compatibility
    # fallback, but keep it deterministic.
    basename = Path(file_path).name
    matches = sorted(image_root.rglob(basename))
    return matches[0] if matches else None


def _image_index(image_root: Path) -> dict[str, Path]:
    """Build one deterministic lookup instead of rescanning for every image."""
    index: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        index.setdefault(path.name, path)
        try:
            index.setdefault(str(path.relative_to(image_root)), path)
        except ValueError:
            pass
    return index


def prepare_arkitscenes(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = Path(args.omni3d_annotation).expanduser()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Omni3D annotation file not found: {annotation_path}")
    image_root = Path(args.arkit_image_root).expanduser()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in payload.get("categories", [])
    }
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        box = _annotation_box(annotation)
        if box is None:
            continue
        box["label"] = str(annotation.get("category_name") or categories.get(annotation.get("category_id"), ""))
        by_image.setdefault(image_id, []).append(box)

    image_index = _image_index(image_root)

    def rows() -> Iterable[dict[str, Any]]:
        count = 0
        for image in images:
            if args.limit and count >= args.limit:
                break
            try:
                image_id = int(image["id"])
            except (KeyError, TypeError, ValueError):
                continue
            boxes = by_image.get(image_id, [])
            if not boxes:
                continue
            file_path = str(image.get("file_path", ""))
            image_path = image_index.get(file_path)
            if image_path is None:
                image_path = image_index.get(Path(file_path).name)
            if image_path is None:
                image_path = _find_image(image_root, file_path)
            if image_path is None:
                raise FileNotFoundError(
                    f"ARKitScenes image not found: {image.get('file_path')} under {image_root}"
                )
            grouped: dict[str, list[dict[str, Any]]] = {}
            for box in boxes:
                if box["label"]:
                    grouped.setdefault(box["label"], []).append(box)
            for class_name, class_boxes in sorted(grouped.items()):
                prompt = (
                    f"Locate the {class_name} in the provided image and output their positions "
                    "and dimensions using 3D bounding boxes. The results must be in the JSON "
                    'format: [{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, '
                    'z_size, roll, pitch, yaw],"label":"category"}].'
                )
                sample_id = f"{image_id}:{class_name}"
                public_boxes = [
                    {"bbox_3d": box["bbox_3d"], "label": box["label"]}
                    for box in class_boxes
                ]
                yield {
                    "example_id": sample_id,
                    "source_id": str(image.get("file_path", image_id)),
                    "question": prompt,
                    "answer": _json_answer(public_boxes),
                    "class_name": class_name,
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "ground_truth_boxes": class_boxes,
                    "messages": _messages(image_path, prompt),
                }
                count += 1
                if args.limit and count >= args.limit:
                    break

    count = _write_jsonl(Path(args.arkit_output), rows())
    manifest = {
        "benchmark": "ARKitScenes",
        "source": "Omni3D ARKitScenes split",
        "annotation_file": str(annotation_path.resolve()),
        "image_root": str(image_root.resolve()),
        "num_examples": count,
        "metric": "mAP@0.15",
        "confidence": 1.0,
        "prompt_protocol": "Qwen3-VL technical report Appendix B.5",
        "box_format": "[x_center,y_center,z_center,x_size,y_size,z_size,roll,pitch,yaw]",
    }
    manifest_path = Path(args.arkit_output).with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathvista-parquet")
    parser.add_argument("--mathvista-output")
    parser.add_argument("--mathvista-image-dir")
    parser.add_argument("--mathvista-images-zip")
    parser.add_argument("--omni3d-annotation")
    parser.add_argument("--arkit-image-root")
    parser.add_argument("--arkit-output")
    parser.add_argument("--only-mathvista", action="store_true")
    parser.add_argument("--only-arkitscenes", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.only_arkitscenes:
        required = {
            "--mathvista-parquet": args.mathvista_parquet,
            "--mathvista-output": args.mathvista_output,
            "--mathvista-image-dir": args.mathvista_image_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing MathVista arguments: {', '.join(missing)}")
        mathvista = prepare_mathvista(args)
        print(json.dumps(mathvista, ensure_ascii=False, indent=2))
    if not args.only_mathvista:
        required = {
            "--omni3d-annotation": args.omni3d_annotation,
            "--arkit-image-root": args.arkit_image_root,
            "--arkit-output": args.arkit_output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing ARKitScenes arguments: {', '.join(missing)}")
        arkit = prepare_arkitscenes(args)
        print(json.dumps(arkit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
