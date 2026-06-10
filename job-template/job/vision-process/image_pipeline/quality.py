from pathlib import Path
from typing import Any, Dict, List

import cv2

from .io import (
    build_summary,
    copy_file,
    default_output_path,
    ensure_dir,
    iter_image_files,
    normalize_bool,
    output_image_path,
    read_image,
    write_csv,
    write_json,
)


def assess_dataset(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    output_root = Path(output_path) if output_path else default_output_path()
    ensure_dir(output_root)

    recursive = normalize_bool(config.get("recursive"), True)
    fail_fast = normalize_bool(config.get("fail_fast"), False)
    copy_valid = normalize_bool(config.get("copy_valid"), False)
    min_width = int(config.get("min_width", 32))
    min_height = int(config.get("min_height", 32))
    min_file_size = int(config.get("min_file_size", 1024))
    blur_threshold = float(config.get("blur_threshold", 100))
    brightness_min = float(config.get("brightness_min", 10))
    brightness_max = float(config.get("brightness_max", 245))

    items = iter_image_files(input_path, recursive=recursive)
    rows: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    valid_count = 0
    corrupt_count = 0

    for item in items:
        row: Dict[str, Any] = {
            "input_path": str(item.path),
            "relative_path": str(item.relative_path),
            "file_size": item.path.stat().st_size,
            "status": "valid",
            "reasons": "",
        }
        reasons: List[str] = []
        try:
            image = read_image(item.path)
            if image is None:
                raise ValueError("image decode failed")

            height, width = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            row.update(
                {
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "brightness": round(brightness, 4),
                    "blur_score": round(blur_score, 4),
                }
            )

            if width < min_width:
                reasons.append("width_too_small")
            if height < min_height:
                reasons.append("height_too_small")
            if row["file_size"] < min_file_size:
                reasons.append("file_too_small")
            if brightness < brightness_min:
                reasons.append("too_dark")
            if brightness > brightness_max:
                reasons.append("too_bright")
            if blur_score < blur_threshold:
                reasons.append("blurred")

            if reasons:
                row["status"] = "invalid"
                row["reasons"] = ",".join(reasons)
                failed_items.append(row.copy())
            else:
                valid_count += 1
                if copy_valid:
                    copy_file(item.path, output_image_path(output_root, item.relative_path))
        except Exception as exc:
            corrupt_count += 1
            row["status"] = "corrupt"
            row["reasons"] = str(exc)
            failed_items.append(row.copy())
            if fail_fast:
                rows.append(row)
                break
        rows.append(row)

    report_json = output_root / "report.json"
    report_csv = output_root / "report.csv"
    result = build_summary(
        "quality_assessment",
        input_path,
        output_root,
        len(items),
        valid_count,
        failed_items,
        report_csv,
        {
            "valid": valid_count,
            "invalid": len(failed_items) - corrupt_count,
            "corrupt": corrupt_count,
            "copy_valid": copy_valid,
        },
    )
    write_json(report_json, {"summary": result, "items": rows})
    write_csv(report_csv, rows)
    result["report_json_path"] = str(report_json)
    return result

