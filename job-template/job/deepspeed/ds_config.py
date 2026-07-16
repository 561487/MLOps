#!/usr/bin/env python3
import argparse
import json
import os
from collections.abc import Mapping


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_override_json(raw):
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("ds_config_override must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("ds_config_override must be a JSON object")
    return value


def _auto_or_number(value):
    if value == "auto":
        return value
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value


def _offload_config(device, nvme_path):
    if device == "none":
        return None
    config = {"device": device}
    if device == "nvme":
        config["nvme_path"] = nvme_path
    return config


def build_ds_config(args):
    config = {
        "train_batch_size": _auto_or_number(args.train_batch_size),
        "train_micro_batch_size_per_gpu": _auto_or_number(
            args.train_micro_batch_size_per_gpu
        ),
        "gradient_accumulation_steps": _auto_or_number(
            args.gradient_accumulation_steps
        ),
        "gradient_clipping": float(args.gradient_clipping),
        "steps_per_print": int(args.steps_per_print),
        "wall_clock_breakdown": False,
    }

    zero_stage = int(args.zero_stage)
    if zero_stage > 0:
        zero_config = {"stage": zero_stage}
        optimizer_offload = _offload_config(args.offload_optimizer, args.nvme_path)
        param_offload = _offload_config(args.offload_param, args.nvme_path)
        if optimizer_offload:
            zero_config["offload_optimizer"] = optimizer_offload
        if param_offload:
            zero_config["offload_param"] = param_offload
        config["zero_optimization"] = zero_config

    precision = args.precision
    config["fp16"] = {"enabled": precision == "fp16"}
    config["bf16"] = {"enabled": precision == "bf16"}

    override = parse_override_json(args.ds_config_override)
    return deep_merge(config, override)


def validate_nvme_paths(config):
    zero_config = config.get("zero_optimization", {})
    for key in ("offload_optimizer", "offload_param"):
        offload = zero_config.get(key, {})
        if offload.get("device") != "nvme":
            continue
        path = offload.get("nvme_path")
        if not path:
            raise ValueError("zero_optimization.%s.nvme_path is required" % key)
        if not os.path.isdir(path):
            raise ValueError(
                "NVMe offload path %s does not exist; mount local NVMe storage at this path"
                % path
            )
        if not os.access(path, os.W_OK):
            raise ValueError("NVMe offload path %s is not writable" % path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("DeepSpeed config generator")
    parser.add_argument("--zero_stage", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument(
        "--precision", choices=["fp16", "bf16", "fp32"], default="fp16"
    )
    parser.add_argument("--train_batch_size", default="auto")
    parser.add_argument("--train_micro_batch_size_per_gpu", default="auto")
    parser.add_argument("--gradient_accumulation_steps", default="auto")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--steps_per_print", type=int, default=10)
    parser.add_argument(
        "--offload_optimizer", choices=["none", "cpu", "nvme"], default="none"
    )
    parser.add_argument(
        "--offload_param", choices=["none", "cpu", "nvme"], default="none"
    )
    parser.add_argument("--nvme_path", default="/local_nvme")
    parser.add_argument("--ds_config_override", default="")
    parser.add_argument("--output", default="/tmp/ds_config.json")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    config = build_ds_config(args)
    validate_nvme_paths(config)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    print("DeepSpeed config written to %s" % args.output, flush=True)


if __name__ == "__main__":
    main()
