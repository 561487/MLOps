"""
推理监控公共时间范围模块。
所有引擎共享同一套时间范围定义和 QueryWindow 生成逻辑。

职责：
- 时间范围名称 → display step / display bucket 映射
- QueryWindow 生成（start_seconds/end_seconds 对齐）
- 不允许在此模块中定义 Prometheus value 规范化逻辑（已移至 metric_utils）

核心语义约定：
- raw_calculation_window_seconds：真实指标计算窗口，固定短窗口（默认 120s），
  绝不随时间范围变化。Counter rate 与 Histogram quantile 都在此窗口上计算。
- display_bucket_seconds：展示降采样桶，随时间范围增大。长窗口下对每个展示桶
  取「桶内峰值」（max_over_time 子查询），只降低时间分辨率，不稀释事件幅值。
- display_step_seconds：query_range 的 step（数据点密度）。
"""
import time
from dataclasses import dataclass
from typing import Optional
from myapp import conf

ALLOWED_RANGES = frozenset({"5m", "15m", "1h", "6h", "24h", "3d"})

# 真实指标计算窗口（秒）：固定短窗口，所有 range 一致。
# 可通过环境变量 INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS 覆盖。
# 必须至少覆盖多个 Prometheus 抓取周期。
RAW_CALCULATION_WINDOW_SECONDS = int(
    conf.get("INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS", "120")
)

# 向后兼容别名
DEFAULT_CALCULATION_WINDOW_SECONDS = RAW_CALCULATION_WINDOW_SECONDS


@dataclass(frozen=True)
class TimeRangeSpec:
    range_name: str
    duration_seconds: int
    display_step_seconds: int    # query_range step（数据点密度）
    display_bucket_seconds: int  # 展示降采样桶（桶内取峰值）
    scrape_interval_seconds: int = 15


# 所有时间范围的唯一权威定义
# display_bucket >= display_step：每个展示点对应一个完整覆盖的统计桶
TIME_RANGE_SPECS: dict[str, TimeRangeSpec] = {
    "5m":  TimeRangeSpec("5m",  300,    15,   15,   15),
    "15m": TimeRangeSpec("15m", 900,    15,   30,   15),
    "1h":  TimeRangeSpec("1h",  3600,   30,   60,   15),
    "6h":  TimeRangeSpec("6h",  21600,  60,   600,  15),
    "24h": TimeRangeSpec("24h", 86400,  300,  1800, 15),
    "3d":  TimeRangeSpec("3d",  259200, 900,  3600, 15),
}


@dataclass(frozen=True)
class QueryWindow:
    range_name: str
    start_seconds: int
    end_seconds: int
    display_step_seconds: int
    display_bucket_seconds: int
    raw_calculation_window_seconds: int = RAW_CALCULATION_WINDOW_SECONDS
    scrape_interval_seconds: int = 15

    # 向后兼容：step_seconds 是 display_step_seconds 的别名
    @property
    def step_seconds(self) -> int:
        return self.display_step_seconds

    # 向后兼容：calculation_window_seconds 是 raw_calculation_window_seconds 的别名
    @property
    def calculation_window_seconds(self) -> int:
        return self.raw_calculation_window_seconds

    @property
    def start_ms(self) -> int:
        return self.start_seconds * 1000

    @property
    def end_ms(self) -> int:
        return self.end_seconds * 1000

    def to_api_dict(self) -> dict:
        return {
            "range": self.range_name,
            "query_start_ms": self.start_ms,
            "query_end_ms": self.end_ms,
            "display_step_seconds": self.display_step_seconds,
            "display_bucket_seconds": self.display_bucket_seconds,
            "raw_calculation_window_seconds": self.raw_calculation_window_seconds,
            # 向后兼容字段
            "calculation_window_seconds": self.raw_calculation_window_seconds,
            "step_seconds": self.display_step_seconds,
        }


def resolve_query_window(range_name: str) -> Optional[QueryWindow]:
    """解析时间范围名称，生成对齐的查询窗口。所有引擎共用。"""
    if range_name not in ALLOWED_RANGES:
        return None

    spec = TIME_RANGE_SPECS[range_name]
    now = int(time.time())
    # end 按 display_step 向下对齐（floor）：query_end_ms <= now_ms 始终成立
    end = (now // spec.display_step_seconds) * spec.display_step_seconds
    start = end - spec.duration_seconds

    return QueryWindow(
        range_name=range_name,
        start_seconds=start,
        end_seconds=end,
        display_step_seconds=spec.display_step_seconds,
        display_bucket_seconds=spec.display_bucket_seconds,
        raw_calculation_window_seconds=RAW_CALCULATION_WINDOW_SECONDS,
        scrape_interval_seconds=spec.scrape_interval_seconds,
    )
