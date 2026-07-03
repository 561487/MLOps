import argparse
import sys

from audio_process.common.logger import get_logger
from audio_process.processors.augment import run_audio_augment
from audio_process.processors.clean import run_audio_clean
from audio_process.processors.quality import run_quality_assessment

logger = get_logger("audio-process")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y", "on"}


def normalize_process_type(value: str) -> str:
    v = (value or "").strip().lower().replace("-", "_")
    mapping = {
        "qa": "quality_assessment",
        "quality": "quality_assessment",
        "quality_assessment": "quality_assessment",
        "audio_quality_assessment": "quality_assessment",
        "clean": "clean",
        "audio_clean": "clean",
        "augment": "augment",
        "audio_augment": "augment",
    }
    if v not in mapping:
        raise ValueError(
            "process_type must be one of: quality_assessment, clean, augment; "
            f"got {value}"
        )
    return mapping[v]


def parse_args():
    parser = argparse.ArgumentParser("audio-process")

    # Node selector. Three UI nodes can share one image and one launcher.py.
    parser.add_argument("--process_type", default="", help="quality_assessment | clean | augment")

    # Common input args.
    parser.add_argument("--input_manifest", default="")
    parser.add_argument("--input_audio_dir", default="")
    parser.add_argument("--audio_path_col", default="audio_path")
    parser.add_argument("--recursive", type=str2bool, default=True)

    # Common output args.
    parser.add_argument("--output_audio_dir", default="")
    parser.add_argument("--output_manifest_path", default="")
    parser.add_argument("--output_report_path", default="")

    # Quality assessment outputs.
    parser.add_argument("--output_summary_csv", default="")
    parser.add_argument("--valid_manifest_path", default="")
    parser.add_argument("--invalid_manifest_path", default="")

    # Quality / clean thresholds.
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--min_rms_db", type=float, default=-50.0)
    parser.add_argument("--silence_threshold_db", type=float, default=-40.0)
    parser.add_argument("--max_silence_ratio", type=float, default=0.8)
    parser.add_argument("--max_clipping_ratio", type=float, default=0.01)

    # Clean args.
    parser.add_argument("--target_channels", type=int, default=1)
    parser.add_argument("--output_format", default="wav")
    parser.add_argument("--normalize_volume", type=str2bool, default=True)
    parser.add_argument("--target_dbfs", type=float, default=-20.0)
    parser.add_argument("--trim_silence", type=str2bool, default=True)
    parser.add_argument("--remove_corrupt", type=str2bool, default=True)

    # Augment args.
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--augment_split", default="train")
    parser.add_argument("--augment_times", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--enable_noise", type=str2bool, default=True)
    parser.add_argument("--noise_level", type=float, default=0.005)
    parser.add_argument("--enable_speed", type=str2bool, default=True)
    parser.add_argument("--speed_min", type=float, default=0.9)
    parser.add_argument("--speed_max", type=float, default=1.1)
    parser.add_argument("--enable_volume", type=str2bool, default=True)
    parser.add_argument("--volume_gain_min", type=float, default=-6.0)
    parser.add_argument("--volume_gain_max", type=float, default=6.0)
    parser.add_argument("--keep_original", type=str2bool, default=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.process_type = normalize_process_type(args.process_type)
        logger.info(f"Resolved process_type={args.process_type}")

        if args.process_type == "quality_assessment":
            run_quality_assessment(args)
        elif args.process_type == "clean":
            run_audio_clean(args)
        elif args.process_type == "augment":
            run_audio_augment(args)
        else:
            raise ValueError(f"unsupported process_type={args.process_type}")

        return 0
    except Exception as exc:
        logger.exception(f"Audio process failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
