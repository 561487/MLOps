import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from .io import (
    build_summary,
    default_output_path,
    ensure_dir,
    iter_image_files,
    normalize_bool,
    output_image_path,
    parse_number_list,
    read_image,
    save_array,
    save_image,
    write_csv,
    write_json,
)


Processor = Callable[[np.ndarray, Path], List[Tuple[np.ndarray, Path, str]]]


def _odd_kernel(value: Any, default: int = 3) -> int:
    kernel = int(value or default)
    if kernel < 1:
        kernel = default
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _parse_color(value: Any, default: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[int, int, int]:
    values = parse_number_list(value, int) if value not in (None, "") else list(default)
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError("pad_color must contain 3 RGB values")
    return tuple(max(0, min(255, int(v))) for v in values)


def _write_operation_report(output_path: Path, rows: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    report_json = output_path / "report.json"
    report_csv = output_path / "report.csv"
    write_json(report_json, {"summary": result, "items": rows})
    write_csv(report_csv, rows)
    result["report_path"] = str(report_csv)
    result["report_json_path"] = str(report_json)


def _process_images(
    deal_type: str,
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    processor: Processor,
) -> Dict[str, Any]:
    output_root = Path(output_path) if output_path else default_output_path()
    ensure_dir(output_root / "images")
    recursive = normalize_bool(config.get("recursive"), True)
    fail_fast = normalize_bool(config.get("fail_fast"), False)
    items = iter_image_files(input_path, recursive=recursive)
    rows: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    success = 0

    for item in items:
        row: Dict[str, Any] = {
            "input_path": str(item.path),
            "relative_path": str(item.relative_path),
            "status": "success",
        }
        try:
            image = read_image(item.path)
            if image is None:
                raise ValueError("image decode failed")
            outputs = processor(image, item.relative_path)
            for output_image, relative_path, save_format in outputs:
                if save_format == "npy":
                    dst = output_image_path(output_root, relative_path, ".npy")
                    save_array(dst, output_image)
                else:
                    dst = output_image_path(output_root, relative_path)
                    save_image(dst, output_image)
            row["output_path"] = str(output_image_path(output_root, outputs[0][1], ".npy" if outputs[0][2] == "npy" else None))
            row["output_count"] = len(outputs)
            success += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            failed_items.append(row.copy())
            if fail_fast:
                rows.append(row)
                break
        rows.append(row)

    result = build_summary(deal_type, input_path, output_root, len(items), success, failed_items)
    _write_operation_report(output_root, rows, result)
    return result


def blur_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    method = str(config.get("method", "gaussian")).lower()
    kernel = _odd_kernel(config.get("kernel_size"), 3)

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        if method == "gaussian":
            processed = cv2.GaussianBlur(image, (kernel, kernel), 0)
        elif method == "median":
            processed = cv2.medianBlur(image, kernel)
        elif method == "bilateral":
            processed = cv2.bilateralFilter(image, kernel, 75, 75)
        else:
            raise ValueError(f"unsupported blur method: {method}")
        return [(processed, relative_path, "image")]

    return _process_images("blur", input_path, output_path, config, processor)


def resize_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    width = int(config.get("width", 640))
    height = int(config.get("height", 640))
    keep_ratio = normalize_bool(config.get("keep_ratio"), True)
    pad_color = _parse_color(config.get("pad_color"), (114, 114, 114))

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if not keep_ratio:
            return [(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), relative_path, "image")]

        src_h, src_w = image.shape[:2]
        scale = min(width / src_w, height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            canvas = np.full((height, width), pad_color[0], dtype=image.dtype)
        else:
            canvas = np.full((height, width, image.shape[2]), pad_color, dtype=image.dtype)
        top = (height - new_h) // 2
        left = (width - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return [(canvas, relative_path, "image")]

    return _process_images("resize", input_path, output_path, config, processor)


def normalize_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(config.get("mode", "minmax")).lower()
    save_format = str(config.get("save_format", "image")).lower()
    mean = parse_number_list(config.get("mean", [0.485, 0.456, 0.406]), float)
    std = parse_number_list(config.get("std", [0.229, 0.224, 0.225]), float)

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        array = image.astype(np.float32) / 255.0
        if mode == "imagenet":
            if image.ndim == 2:
                array = np.expand_dims(array, axis=-1)
            mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
            std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
            array = (array - mean_arr) / std_arr
        elif mode == "minmax":
            min_value = float(array.min())
            max_value = float(array.max())
            if max_value > min_value:
                array = (array - min_value) / (max_value - min_value)
        elif mode not in {"none", "raw"}:
            raise ValueError(f"unsupported normalize mode: {mode}")

        if save_format == "npy":
            return [(array, relative_path, "npy")]
        image_out = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return [(image_out, relative_path, "image")]

    return _process_images("normalize", input_path, output_path, config, processor)


def crop_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(config.get("mode", "center")).lower()
    width = int(config.get("width", 512))
    height = int(config.get("height", 512))

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        src_h, src_w = image.shape[:2]
        if mode == "center":
            crop_w = min(width, src_w)
            crop_h = min(height, src_h)
            x1 = (src_w - crop_w) // 2
            y1 = (src_h - crop_h) // 2
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        elif mode == "box":
            box = parse_number_list(config.get("box"), int)
            if len(box) != 4:
                raise ValueError("box must be x1,y1,x2,y2")
            x1, y1, x2, y2 = box
        elif mode == "ratio":
            box = parse_number_list(config.get("box", [0, 0, 1, 1]), float)
            if len(box) != 4:
                raise ValueError("ratio box must be x1,y1,x2,y2")
            x1, y1, x2, y2 = int(box[0] * src_w), int(box[1] * src_h), int(box[2] * src_w), int(box[3] * src_h)
        else:
            raise ValueError(f"unsupported crop mode: {mode}")

        x1, x2 = sorted((max(0, int(x1)), min(src_w, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(src_h, int(y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("crop area is empty")
        return [(image[y1:y2, x1:x2], relative_path, "image")]

    return _process_images("cropping", input_path, output_path, config, processor)


def equalize_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    method = str(config.get("method", "clahe")).lower()
    clip_limit = float(config.get("clip_limit", 2.0))
    tile_grid_size = parse_number_list(config.get("tile_grid_size", [8, 8]), int)
    if len(tile_grid_size) != 2:
        tile_grid_size = [8, 8]

    def equalize_channel(channel: np.ndarray) -> np.ndarray:
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
            return clahe.apply(channel)
        if method in {"global", "hist"}:
            return cv2.equalizeHist(channel)
        raise ValueError(f"unsupported equalize method: {method}")

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        if image.ndim == 2:
            return [(equalize_channel(image), relative_path, "image")]
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = equalize_channel(ycrcb[:, :, 0])
        return [(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), relative_path, "image")]

    return _process_images("equalize", input_path, output_path, config, processor)


def cvt_color_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from_color = str(config.get("from_color", config.get("from", "BGR"))).upper()
    to_color = str(config.get("to_color", config.get("to", "GRAY"))).upper()

    conversion_codes = {
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
        ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
        ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
        ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
        ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
        ("HSV", "RGB"): cv2.COLOR_HSV2RGB,
        ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
        ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
        ("LAB", "BGR"): cv2.COLOR_LAB2BGR,
        ("LAB", "RGB"): cv2.COLOR_LAB2RGB,
    }

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        if from_color == to_color:
            return [(image, relative_path, "image")]
        code = conversion_codes.get((from_color, to_color))
        if code is None:
            raise ValueError(f"unsupported color conversion: {from_color}->{to_color}")
        if image.ndim == 2 and from_color != "GRAY":
            raise ValueError("input image is grayscale but from_color is not GRAY")
        if image.ndim != 2 and from_color == "GRAY":
            image_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_in = image
        return [(cv2.cvtColor(image_in, code), relative_path, "image")]

    return _process_images("cvtcolor", input_path, output_path, config, processor)


def augment_images(input_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    num_outputs = max(1, int(config.get("num_outputs", 1)))
    transforms = config.get("transforms") or [{"type": "flip", "direction": "horizontal", "p": 1.0}]
    if isinstance(transforms, str):
        import json

        transforms = json.loads(transforms)

    def apply_transform(image: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        probability = float(transform.get("p", 1.0))
        if random.random() > probability:
            return image
        transform_type = str(transform.get("type", "")).lower()
        if transform_type == "flip":
            direction = str(transform.get("direction", "horizontal")).lower()
            code = 1 if direction == "horizontal" else 0 if direction == "vertical" else -1
            return cv2.flip(image, code)
        if transform_type == "rotate":
            angle = float(transform.get("angle", 15))
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if transform_type == "brightness_contrast":
            brightness = float(transform.get("brightness", 0.0)) * 255.0
            contrast = 1.0 + float(transform.get("contrast", 0.0))
            return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        if transform_type == "noise":
            sigma = float(transform.get("sigma", 10.0))
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        raise ValueError(f"unsupported transform type: {transform_type}")

    def processor(image: np.ndarray, relative_path: Path) -> List[Tuple[np.ndarray, Path, str]]:
        outputs: List[Tuple[np.ndarray, Path, str]] = []
        for index in range(num_outputs):
            augmented = image.copy()
            for transform in transforms:
                augmented = apply_transform(augmented, transform)
            augmented_path = relative_path.with_name(f"{relative_path.stem}_aug{index + 1}{relative_path.suffix}")
            outputs.append((augmented, augmented_path, "image"))
        return outputs

    return _process_images("augmentation", input_path, output_path, config, processor)

