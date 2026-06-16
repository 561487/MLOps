import json
from pathlib import Path
from typing import Any, Dict, List


def write_metrics(result: Dict[str, Any], metric_path: str = "/metric.json") -> None:
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)
    report_path = result.get("report_path") or str(Path(result.get("output_path", "")) / "report.csv")

    metrics: List[Dict[str, Any]] = [
        {
            "metric_type": "text",
            "describe": "图像处理统计",
            "text": f"total={total}, success={success}, failed={failed}",
        }
    ]
    if report_path:
        metrics.append(
            {
                "metric_type": "table",
                "describe": "图像处理报告",
                "file_path": report_path,
            }
        )

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

