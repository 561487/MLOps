import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any

from .paths import ensure_parent_dir


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_report(
    node_name: str,
    process_type: str,
    input_manifest: str,
    input_audio_dir: str,
    output_manifest_path: str,
    output_audio_dir: str,
    total_count: int,
    success_count: int,
    failed_records: List[Dict[str, Any]],
    start_time: str,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    reasons = Counter([str(x.get("reason", "unknown")) for x in failed_records])
    report = {
        "node_name": node_name,
        "process_type": process_type,
        "input_manifest": input_manifest or "",
        "input_audio_dir": input_audio_dir or "",
        "output_manifest_path": output_manifest_path or "",
        "output_audio_dir": output_audio_dir or "",
        "total_count": int(total_count),
        "success_count": int(success_count),
        "failed_count": int(len(failed_records)),
        "failed_reasons": dict(reasons),
        "start_time": start_time,
        "end_time": now_str(),
        "status": "success",
    }
    if extra:
        report["extra"] = extra
    return report


def save_report(report: Dict[str, Any], output_report_path: str) -> None:
    ensure_parent_dir(output_report_path)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
