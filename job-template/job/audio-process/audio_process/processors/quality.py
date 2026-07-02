from typing import List, Tuple

import pandas as pd

from audio_process.common.audio_io import load_audio, to_mono, get_channel_count
from audio_process.common.constants import (
    REASON_CLIPPING,
    REASON_CORRUPT,
    REASON_LOW_VOLUME,
    REASON_SAMPLE_RATE_MISMATCH,
    REASON_TOO_LONG,
    REASON_TOO_MUCH_SILENCE,
    REASON_TOO_SHORT,
    STATUS_INVALID,
    STATUS_VALID,
)
from audio_process.common.logger import get_logger
from audio_process.common.manifest import load_input_as_manifest, save_manifest
from audio_process.common.metrics import (
    calc_clipping_ratio,
    calc_duration,
    calc_rms_db,
    calc_silence_ratio,
)
from audio_process.common.paths import ensure_parent_dir
from audio_process.common.report import build_report, now_str, save_report

logger = get_logger("audio-quality-assessment")


def _require_quality_outputs(args) -> None:
    required = [
        "output_report_path",
        "output_summary_csv",
        "valid_manifest_path",
        "invalid_manifest_path",
    ]
    for name in required:
        if not getattr(args, name, ""):
            raise ValueError(f"{name} cannot be empty for quality_assessment")


def _judge_quality(item: dict, args) -> Tuple[str, str]:
    reasons: List[str] = []

    if item["duration"] < args.min_duration:
        reasons.append(REASON_TOO_SHORT)
    if item["duration"] > args.max_duration:
        reasons.append(REASON_TOO_LONG)
    if item["rms_db"] < args.min_rms_db:
        reasons.append(REASON_LOW_VOLUME)
    if item["silence_ratio"] > args.max_silence_ratio:
        reasons.append(REASON_TOO_MUCH_SILENCE)
    if item["clipping_ratio"] > args.max_clipping_ratio:
        reasons.append(REASON_CLIPPING)
    if args.target_sample_rate and item["sample_rate"] != args.target_sample_rate:
        reasons.append(REASON_SAMPLE_RATE_MISMATCH)

    if reasons:
        return STATUS_INVALID, ";".join(reasons)
    return STATUS_VALID, ""


def run_quality_assessment(args) -> None:
    _require_quality_outputs(args)
    start_time = now_str()

    logger.info("Start audio quality assessment")
    logger.info(f"input_manifest={args.input_manifest}")
    logger.info(f"input_audio_dir={args.input_audio_dir}")

    df = load_input_as_manifest(
        input_manifest=args.input_manifest,
        input_audio_dir=args.input_audio_dir,
        audio_path_col=args.audio_path_col,
        recursive=args.recursive,
    )

    summary_records = []
    valid_rows = []
    invalid_rows = []
    failed_records = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        audio_path = row_dict.get(args.audio_path_col, "")

        try:
            y, sr = load_audio(audio_path, sr=None, mono=False)
            channels = get_channel_count(y)
            y_mono = to_mono(y)

            item = row_dict.copy()
            item.update(
                {
                    "sample_rate": sr,
                    "channels": channels,
                    "duration": calc_duration(y_mono, sr),
                    "rms_db": calc_rms_db(y_mono),
                    "silence_ratio": calc_silence_ratio(
                        y_mono, silence_threshold_db=args.silence_threshold_db
                    ),
                    "clipping_ratio": calc_clipping_ratio(y_mono),
                }
            )

            status, reason = _judge_quality(item, args)
            item["status"] = status
            item["reason"] = reason
            summary_records.append(item)

            if status == STATUS_VALID:
                valid_rows.append(row_dict)
            else:
                invalid_item = row_dict.copy()
                invalid_item["reason"] = reason
                invalid_rows.append(invalid_item)
                failed_records.append(invalid_item)

        except Exception as exc:
            logger.exception(f"Failed to assess audio idx={idx}, path={audio_path}")
            invalid_item = row_dict.copy()
            invalid_item["status"] = STATUS_INVALID
            invalid_item["reason"] = REASON_CORRUPT
            invalid_item["error"] = str(exc)
            summary_records.append(invalid_item)
            invalid_rows.append(invalid_item)
            failed_records.append(invalid_item)

    summary_df = pd.DataFrame(summary_records)
    valid_df = pd.DataFrame(valid_rows, columns=df.columns)
    invalid_df = pd.DataFrame(invalid_rows)

    ensure_parent_dir(args.output_summary_csv)
    summary_df.to_csv(args.output_summary_csv, index=False)
    save_manifest(valid_df, args.valid_manifest_path)
    save_manifest(invalid_df, args.invalid_manifest_path)

    report = build_report(
        node_name="audio-process",
        process_type="quality_assessment",
        input_manifest=args.input_manifest,
        input_audio_dir=args.input_audio_dir,
        output_manifest_path=args.valid_manifest_path,
        output_audio_dir="",
        total_count=len(df),
        success_count=len(valid_df),
        failed_records=failed_records,
        start_time=start_time,
        extra={
            "output_summary_csv": args.output_summary_csv,
            "invalid_manifest_path": args.invalid_manifest_path,
        },
    )
    save_report(report, args.output_report_path)

    logger.info(f"Total: {len(df)}")
    logger.info(f"Valid: {len(valid_df)}")
    logger.info(f"Invalid: {len(invalid_df)}")
    logger.info(f"Summary: {args.output_summary_csv}")
    logger.info(f"Report: {args.output_report_path}")
