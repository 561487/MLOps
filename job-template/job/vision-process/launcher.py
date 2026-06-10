import argparse
import json
from typing import Any, Dict

from image_pipeline.metrics import write_metrics
from image_pipeline.ops import (
    augment_images,
    blur_images,
    crop_images,
    cvt_color_images,
    equalize_images,
    normalize_images,
    resize_images,
)
from image_pipeline.quality import assess_dataset


HANDLERS = {
    "quality_assessment": assess_dataset,
    "blur": blur_images,
    "resize": resize_images,
    "normalize": normalize_images,
    "cropping": crop_images,
    "equalize": equalize_images,
    "cvtcolor": cvt_color_images,
    "augmentation": augment_images,
}





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("vision process launcher")
    parser.add_argument("--deal_type", required=True, choices=sorted(HANDLERS.keys()))
    parser.add_argument("--input_path", default="", help="输入图片目录或单张图片路径")
    parser.add_argument("--output_path", default="", help="输出目录")
    parser.add_argument("--config", default="{}", help="JSON 字符串形式的扩展参数")
    parser.add_argument("--recursive", nargs="?", const="true", default=None)
    parser.add_argument("--fail_fast", nargs="?", const="true", default=None)

    parser.add_argument("--min_width", default=None)
    parser.add_argument("--min_height", default=None)
    parser.add_argument("--min_file_size", default=None)
    parser.add_argument("--blur_threshold", default=None)
    parser.add_argument("--brightness_min", default=None)
    parser.add_argument("--brightness_max", default=None)
    parser.add_argument("--copy_valid", nargs="?", const="true", default=None)

    parser.add_argument("--method", default=None)
    parser.add_argument("--kernel_size", default=None)
    parser.add_argument("--width", default=None)
    parser.add_argument("--height", default=None)
    parser.add_argument("--keep_ratio", nargs="?", const="true", default=None)
    parser.add_argument("--pad_color", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--mean", default=None)
    parser.add_argument("--std", default=None)
    parser.add_argument("--save_format", default=None)
    parser.add_argument("--box", default=None)
    parser.add_argument("--clip_limit", default=None)
    parser.add_argument("--tile_grid_size", default=None)
    parser.add_argument("--from_color", default=None)
    parser.add_argument("--to_color", default=None)
    parser.add_argument("--num_outputs", default=None)
    parser.add_argument("--transforms", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        config = json.loads(args.config or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config must be valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("--config must be a JSON object")

    skip_keys = {"deal_type", "input_path", "output_path", "config"}
    for key, value in vars(args).items():
        if key in skip_keys or value is None or value == "":
            continue
        config[key] = value
    return config


def main() -> None:
    args = parse_args()
    if not args.input_path:
        raise ValueError("--input_path is required")

    config = build_config(args)
    handler = HANDLERS[args.deal_type]
    result = handler(args.input_path, args.output_path, config)
    write_metrics(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

