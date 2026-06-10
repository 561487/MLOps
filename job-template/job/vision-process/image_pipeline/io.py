import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageItem:
    path: Path
    relative_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_path() -> Path:
    creator = os.getenv("KFJ_CREATOR", "admin")
    pipeline_name = os.getenv("KFJ_PIPELINE_NAME", "vision-process")
    task_name = os.getenv("KFJ_TASK_NAME", "output")
    return Path("/mnt") / creator / "pipeline" / pipeline_name / task_name


def iter_image_files(input_path: str, recursive: bool = True) -> List[ImageItem]:
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")

    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            return []
        return [ImageItem(root, Path(root.name))]

    pattern = "**/*" if recursive else "*"
    items: List[ImageItem] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append(ImageItem(path, path.relative_to(root)))
    return items


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"failed to encode image as {suffix}: {path}")
    encoded.tofile(str(path))


def save_array(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(str(path), array)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


def output_image_path(output_root: Path, relative_path: Path, suffix: Optional[str] = None) -> Path:
    relative = relative_path
    if suffix:
        relative = relative.with_suffix(suffix)
    return output_root / "images" / relative


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_number_list(value: Any, item_type: type = float) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            value = json.loads(value)
        else:
            value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, Iterable):
        raise ValueError(f"expected list value, got {value!r}")
    return [item_type(item) for item in value]


def build_summary(
    deal_type: str,
    input_path: str,
    output_path: Path,
    total: int,
    success: int,
    failed_items: Sequence[Dict[str, Any]],
    report_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "deal_type": deal_type,
        "input_path": input_path,
        "output_path": str(output_path),
        "images_path": str(output_path / "images"),
        "total": total,
        "success": success,
        "failed": len(failed_items),
        "failed_items": list(failed_items),
    }
    if report_path:
        summary["report_path"] = str(report_path)
    if extra:
        summary.update(extra)
    return summary

