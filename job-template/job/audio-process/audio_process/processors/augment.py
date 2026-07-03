import os
import random

import numpy as np
import pandas as pd

from audio_process.common.audio_io import load_audio, save_audio
from audio_process.common.logger import get_logger
from audio_process.common.manifest import load_input_as_manifest, save_manifest
from audio_process.common.ops import (
    add_noise,
    change_speed,
    change_volume,
    limit_amplitude,
    random_speed,
    random_volume_gain,
)
from audio_process.common.paths import ensure_dir, safe_stem
from audio_process.common.report import build_report, now_str, save_report

logger = get_logger("audio-augment")


def _require_augment_outputs(args) -> None:
    required = ["output_audio_dir", "output_manifest_path", "output_report_path"]
    for name in required:
        if not getattr(args, name, ""):
            raise ValueError(f"{name} cannot be empty for augment")


def _build_augmented_audio_path(output_audio_dir: str, source_path: str, idx: int, aug_idx: int) -> str:
    filename = f"{idx:06d}_{safe_stem(source_path)}_aug{aug_idx}.wav"
    return os.path.join(output_audio_dir, filename)


def _should_augment(row_dict: dict, split_col: str, augment_split: str) -> bool:
    if not split_col or split_col not in row_dict:
        return True
    return str(row_dict.get(split_col)) == str(augment_split)


def _apply_random_augment(y, args):
    y_aug = np.asarray(y, dtype=np.float32).copy()
    augment_types = []

    if args.enable_noise and random.random() < 0.5:
        y_aug = add_noise(y_aug, noise_level=args.noise_level)
        augment_types.append("noise")

    if args.enable_speed and random.random() < 0.5:
        speed = random_speed(args.speed_min, args.speed_max)
        y_aug = change_speed(y_aug, speed)
        augment_types.append(f"speed_{speed:.2f}")

    if args.enable_volume and random.random() < 0.5:
        gain = random_volume_gain(args.volume_gain_min, args.volume_gain_max)
        y_aug = change_volume(y_aug, gain)
        augment_types.append(f"volume_{gain:.2f}db")

    if not augment_types:
        if args.enable_noise:
            y_aug = add_noise(y_aug, noise_level=args.noise_level)
            augment_types.append("noise")
        elif args.enable_volume:
            gain = random_volume_gain(args.volume_gain_min, args.volume_gain_max)
            y_aug = change_volume(y_aug, gain)
            augment_types.append(f"volume_{gain:.2f}db")
        else:
            augment_types.append("copy")

    return limit_amplitude(y_aug), "+".join(augment_types)


def run_audio_augment(args) -> None:
    _require_augment_outputs(args)
    start_time = now_str()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logger.info("Start audio augment")
    logger.info(f"input_manifest={args.input_manifest}")
    logger.info(f"input_audio_dir={args.input_audio_dir}")
    logger.info(f"output_audio_dir={args.output_audio_dir}")

    ensure_dir(args.output_audio_dir)

    df = load_input_as_manifest(
        input_manifest=args.input_manifest,
        input_audio_dir=args.input_audio_dir,
        audio_path_col=args.audio_path_col,
        recursive=args.recursive,
    )

    output_rows = []
    failed_records = []
    augmented_count = 0

    if args.keep_original:
        for _, row in df.iterrows():
            item = row.to_dict()
            item["source_audio_path"] = ""
            item["is_augmented"] = False
            item["augment_type"] = "original"
            output_rows.append(item)

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        source_audio_path = row_dict.get(args.audio_path_col, "")

        if not _should_augment(row_dict, args.split_col, args.augment_split):
            continue

        try:
            y, sr = load_audio(source_audio_path, sr=None, mono=True)

            for aug_idx in range(1, args.augment_times + 1):
                y_aug, augment_type = _apply_random_augment(y, args)
                output_audio_path = _build_augmented_audio_path(
                    args.output_audio_dir, source_audio_path, idx, aug_idx
                )
                save_audio(output_audio_path, y_aug, sr)

                output_item = row_dict.copy()
                output_item[args.audio_path_col] = output_audio_path
                output_item["source_audio_path"] = source_audio_path
                output_item["is_augmented"] = True
                output_item["augment_type"] = augment_type
                output_rows.append(output_item)
                augmented_count += 1

        except Exception as exc:
            logger.exception(f"Failed to augment audio idx={idx}, path={source_audio_path}")
            failed_item = row_dict.copy()
            failed_item["reason"] = "augment_failed"
            failed_item["error"] = str(exc)
            failed_records.append(failed_item)

    output_df = pd.DataFrame(output_rows)
    save_manifest(output_df, args.output_manifest_path)

    report = build_report(
        node_name="audio-process",
        process_type="augment",
        input_manifest=args.input_manifest,
        input_audio_dir=args.input_audio_dir,
        output_manifest_path=args.output_manifest_path,
        output_audio_dir=args.output_audio_dir,
        total_count=len(df),
        success_count=len(output_df),
        failed_records=failed_records,
        start_time=start_time,
        extra={
            "augmented_count": augmented_count,
            "keep_original": args.keep_original,
            "augment_split": args.augment_split,
            "augment_times": args.augment_times,
        },
    )
    save_report(report, args.output_report_path)

    logger.info(f"Total input: {len(df)}")
    logger.info(f"Total output rows: {len(output_df)}")
    logger.info(f"Augmented files: {augmented_count}")
    logger.info(f"Failed: {len(failed_records)}")
    logger.info(f"Output manifest: {args.output_manifest_path}")
    logger.info(f"Report: {args.output_report_path}")
