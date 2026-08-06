"""
推理监控公共指标工具函数。
所有 vLLM、SGLang 及未来引擎的 Adapter 共用此模块。

职责：
- Prometheus value 规范化（NaN/Inf/null/去重/排序）
- 窗口外数据过滤
- 时间戳 ms 化

不得将引擎特定的指标名、PromQL、label 处理放入此模块。
"""
import math


def normalize_prometheus_matrix_values(values):
    """
    将 Prometheus query_range 返回的 [timestamp, value] 列表规范化为统一格式。

    规则：
    - 正常有限值 → {"timestamp_ms": int(ts*1000), "value": float}
    - NaN/Inf/"NaN"/"+Inf"/"-Inf" → {"timestamp_ms": int(ts*1000), "value": None}
    - 无效时间戳（None/0/NaN/Inf/≤0） → 跳过该点
    - 相同时间戳去重，保留最后一个有效值
    - 有效值优先于同时间戳 null
    - 结果按时间戳升序

    所有 Adapter 必须共用此函数，不得内联不同版本的规范化逻辑。
    """
    if not values:
        return []

    seen: dict[int, dict] = {}
    stale_ts: set[int] = set()

    for ts, val in values:
        # 安全检查时间戳
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts_f) or ts_f <= 0:
            continue
        timestamp_ms = int(ts_f * 1000)

        # 安全检查值：NaN/Inf/None → null
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = float('nan')

        if math.isfinite(v):
            # 有效值覆盖同时间戳的 null
            seen[timestamp_ms] = {"timestamp_ms": timestamp_ms, "value": v}
            stale_ts.discard(timestamp_ms)
        else:
            # NaN/Inf → 保留时间戳但值为 null（用于前端断线），
            # 仅当该时间戳还没有有效值时才记录
            if timestamp_ms not in seen and timestamp_ms not in stale_ts:
                seen[timestamp_ms] = {"timestamp_ms": timestamp_ms, "value": None}

    return sorted(seen.values(), key=lambda x: x["timestamp_ms"])


def filter_points_by_window(points, query_start_ms, query_end_ms):
    """
    从 normalize_prometheus_matrix_values 返回的点列表中过滤出窗口内的数据点。
    用于需要独立过滤窗口的场景。
    """
    if not points:
        return []
    return [
        p for p in points
        if p["timestamp_ms"] >= query_start_ms and p["timestamp_ms"] <= query_end_ms
    ]
