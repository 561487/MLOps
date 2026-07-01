import os

import pandas as pd

from audio_process.common.audio_io import load_audio, resample_audio, save_audio, to_mono
from audio_process.common.logger import get_logger
from audio_process.common.manifest import load_input_as_manifest, save_manifest
from audio_process.common.metrics import calc_duration
from audio_process.common.ops import limit_amplitude, normalize_volume, trim_silence
from audio_process.common.paths import ensure_dir, safe_stem
from audio_process.common.report import build_report, now_str, save_report

logger = get_logger("audio-clean")


def _require_clean_outputs(args) -> None:
    required = ["output_audio_dir", "output_manifest_path", "output_report_path"]
    for name in required:
        if not getattr(args, name, ""):
            raise ValueError(f"{name} cannot be empty for clean")


def _build_output_audio_path(output_audio_dir: str, source_path: str, idx: int, output_format: str) -> str:
    ext = output_format.lower().lstrip(".") or "wav"
    filename = f"{idx:06d}_{safe_stem(source_path)}.{ext}"
    return os.path.join(output_audio_dir, filename)


def run_audio_clean(args) -> None:
    _require_clean_outputs(args)
    start_time = now_str()

    logger.info("Start audio clean")
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

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        source_audio_path = row_dict.get(args.audio_path_col, "")

        try:
            y, sr = load_audio(source_audio_path, sr=None, mono=False)

            if args.target_channels == 1:
                y = to_mono(y)
            else:
                raise ValueError("Only target_channels=1 is supported in the first version")

            if args.target_sample_rate and sr != args.target_sample_rate:
                y = resample_audio(y, sr, args.target_sample_rate)
                sr = args.target_sample_rate

            if args.trim_silence:
                top_db = abs(float(args.silence_threshold_db))
                y = trim_silence(y, top_db=top_db)

            if args.normalize_volume:
                y = normalize_volume(y, target_dbfs=args.target_dbfs)

            y = limit_amplitude(y)
            duration = calc_duration(y, sr)

            if duration < args.min_duration:
                failed_item = row_dict.copy()
                failed_item["reason"] = "too_short_after_clean"
                failed_records.append(failed_item)
                continue

            if duration > args.max_duration:
                failed_item = row_dict.copy()
                failed_item["reason"] = "too_long_after_clean"
                failed_records.append(failed_item)
                continue

            output_audio_path = _build_output_audio_path(
                args.output_audio_dir, source_audio_path, idx, args.output_format
            )
            save_audio(output_audio_path, y, sr)

            output_item = row_dict.copy()
            output_item["source_audio_path"] = source_audio_path
            output_item[args.audio_path_col] = output_audio_path
            output_item["sample_rate"] = sr
            output_item["duration"] = duration
            output_rows.append(output_item)

        except Exception as exc:
            logger.exception(f"Failed to clean audio idx={idx}, path={source_audio_path}")
            failed_item = row_dict.copy()
            failed_item["reason"] = "clean_failed"
            failed_item["error"] = str(exc)
            failed_records.append(failed_item)
            if not args.remove_corrupt:
                raise

    output_df = pd.DataFrame(output_rows)
    save_manifest(output_df, args.output_manifest_path)

    report = build_report(
        node_name="audio-process",
        process_type="clean",
        input_manifest=args.input_manifest,
        input_audio_dir=args.input_audio_dir,
        output_manifest_path=args.output_manifest_path,
        output_audio_dir=args.output_audio_dir,
        total_count=len(df),
        success_count=len(output_df),
        failed_records=failed_records,
        start_time=start_time,
    )
    save_report(report, args.output_report_path)

    logger.info(f"Total: {len(df)}")
    logger.info(f"Cleaned: {len(output_df)}")
    logger.info(f"Failed: {len(failed_records)}")
    logger.info(f"Output manifest: {args.output_manifest_path}")
    logger.info(f"Report: {args.output_report_path}")
