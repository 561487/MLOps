"""
vLLM 指标 PromQL 适配器。
所有 PromQL 使用 job + kubernetes_namespace + mlops_service_id 三重标签限定。
"""
import math
import time

# 统计窗口常量
SUMMARY_RATE_WINDOW = "5m"
TIMESERIES_RATE_WINDOW = "5m"

# 支持的指标白名单
SUPPORTED_METRICS = {
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "itl_p50",
    "itl_p95",
    "qps",
    "running_requests",
    "waiting_requests",
    "input_tokens_per_second",
    "output_tokens_per_second",
}

# 时间范围到 step 的映射
RANGE_STEP_MAP = {
    "5m": "15s",
    "15m": "15s",
    "1h": "30s",
    "6h": "60s",
    "24h": "300s",
    "3d": "900s",
}

ALLOWED_RANGES = frozenset(RANGE_STEP_MAP.keys())

# 指标单位映射
METRIC_UNITS = {
    "ttft_p50": "ms",
    "ttft_p95": "ms",
    "ttft_p99": "ms",
    "itl_p50": "ms/token",
    "itl_p95": "ms/token",
    "qps": "req/s",
    "running_requests": "requests",
    "waiting_requests": "requests",
    "input_tokens_per_second": "tokens/s",
    "output_tokens_per_second": "tokens/s",
}

# Prometheus label matcher 转义：反斜杠、双引号、换行
def _escape_label_value(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label_filter(service_id, namespace, job="kubernetes-service-endpoints"):
    """构建统一的 Prometheus 标签过滤器"""
    return (
        f'job="{job}",'
        f'kubernetes_namespace="{_escape_label_value(namespace)}",'
        f'mlops_service_id="{_escape_label_value(service_id)}"'
    )


def _build_histogram_quantile(quantile, metric_name, service_id, namespace, rate_window):
    """构建 histogram_quantile PromQL"""
    labels = _label_filter(service_id, namespace)
    return (
        f"1000 * histogram_quantile({quantile}, "
        f"sum by (le) ("
        f"rate({metric_name}_bucket{{{labels}}}[{rate_window}])"
        f"))"
    )


def _build_histogram_count(metric_name, service_id, namespace, rate_window):
    """构建 histogram _count 查询（用于判断是否有数据）"""
    labels = _label_filter(service_id, namespace)
    return (
        f"sum(increase({metric_name}_count{{{labels}}}[{rate_window}]))"
    )


def _build_rate_sum(metric_name, service_id, namespace, rate_window):
    """构建 sum(rate(...)) PromQL"""
    labels = _label_filter(service_id, namespace)
    return f"sum(rate({metric_name}{{{labels}}}[{rate_window}]))"


def _build_gauge_sum(metric_name, service_id, namespace):
    """构建 sum(gauge) PromQL"""
    labels = _label_filter(service_id, namespace)
    return f"sum({metric_name}{{{labels}}})"


# ====== 瞬时查询 PromQL 构建 ======

SUMMARY_PROMQL_BUILDERS = {
    "ttft_p50": lambda sid, ns: _build_histogram_quantile(0.50, "vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "ttft_p95": lambda sid, ns: _build_histogram_quantile(0.95, "vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "ttft_p99": lambda sid, ns: _build_histogram_quantile(0.99, "vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "itl_p50": lambda sid, ns: _build_histogram_quantile(0.50, "vllm:time_per_output_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "itl_p95": lambda sid, ns: _build_histogram_quantile(0.95, "vllm:time_per_output_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "qps": lambda sid, ns: _build_rate_sum("vllm:request_success_total", sid, ns, SUMMARY_RATE_WINDOW),
    "running_requests": lambda sid, ns: _build_gauge_sum("vllm:num_requests_running", sid, ns),
    "waiting_requests": lambda sid, ns: _build_gauge_sum("vllm:num_requests_waiting", sid, ns),
    "input_tokens_per_second": lambda sid, ns: _build_rate_sum("vllm:prompt_tokens_total", sid, ns, SUMMARY_RATE_WINDOW),
    "output_tokens_per_second": lambda sid, ns: _build_rate_sum("vllm:generation_tokens_total", sid, ns, SUMMARY_RATE_WINDOW),
}

# 瞬时查询对应的 _count 检查（仅 histogram 指标需要）
SUMMARY_COUNT_BUILDERS = {
    "ttft_p50": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "ttft_p95": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "ttft_p99": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "itl_p50": lambda sid, ns: _build_histogram_count("vllm:time_per_output_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
    "itl_p95": lambda sid, ns: _build_histogram_count("vllm:time_per_output_token_seconds", sid, ns, SUMMARY_RATE_WINDOW),
}


# ====== 范围查询 PromQL 构建 ======

TIMESERIES_PROMQL_BUILDERS = {
    "ttft_p50": lambda sid, ns: _build_histogram_quantile(0.50, "vllm:time_to_first_token_seconds", sid, ns, TIMESERIES_RATE_WINDOW),
    "ttft_p95": lambda sid, ns: _build_histogram_quantile(0.95, "vllm:time_to_first_token_seconds", sid, ns, TIMESERIES_RATE_WINDOW),
    "ttft_p99": lambda sid, ns: _build_histogram_quantile(0.99, "vllm:time_to_first_token_seconds", sid, ns, TIMESERIES_RATE_WINDOW),
    "itl_p50": lambda sid, ns: _build_histogram_quantile(0.50, "vllm:time_per_output_token_seconds", sid, ns, TIMESERIES_RATE_WINDOW),
    "itl_p95": lambda sid, ns: _build_histogram_quantile(0.95, "vllm:time_per_output_token_seconds", sid, ns, TIMESERIES_RATE_WINDOW),
    "qps": lambda sid, ns: _build_rate_sum("vllm:request_success_total", sid, ns, TIMESERIES_RATE_WINDOW),
    "running_requests": lambda sid, ns: _build_gauge_sum("vllm:num_requests_running", sid, ns),
    "waiting_requests": lambda sid, ns: _build_gauge_sum("vllm:num_requests_waiting", sid, ns),
    "input_tokens_per_second": lambda sid, ns: _build_rate_sum("vllm:prompt_tokens_total", sid, ns, TIMESERIES_RATE_WINDOW),
    "output_tokens_per_second": lambda sid, ns: _build_rate_sum("vllm:generation_tokens_total", sid, ns, TIMESERIES_RATE_WINDOW),
}


def build_batch_up_query():
    """构建批量查询所有 vLLM 服务 up 状态的 PromQL"""
    return (
        'max by (mlops_service_id, mlops_service_name, kubernetes_namespace) ('
        'up{job="kubernetes-service-endpoints", mlops_engine="vllm"}'
        ')'
    )


def build_target_up_query(service_id, namespace):
    """构建单个服务的 up 查询"""
    labels = _label_filter(service_id, namespace)
    return f'up{{{labels}}}'


def get_summary_metrics(prom_client, service_id, namespace):
    """
    获取单个服务的摘要指标。
    返回: {
        metric_name: {value, status, unit, message}
    }
    """
    result = {}
    for metric_name, builder in SUMMARY_PROMQL_BUILDERS.items():
        promql = builder(str(service_id), namespace)
        try:
            data = prom_client.instant_query(promql)
        except Exception:
            result[metric_name] = {
                "value": None,
                "status": "query_failed",
                "unit": METRIC_UNITS.get(metric_name, ""),
                "message": "监控服务暂时不可用",
            }
            continue

        if data is None:
            result[metric_name] = {
                "value": None,
                "status": "query_failed",
                "unit": METRIC_UNITS.get(metric_name, ""),
                "message": "监控服务暂时不可用",
            }
            continue

        if not data:
            result[metric_name] = {
                "value": None,
                "status": "no_data",
                "unit": METRIC_UNITS.get(metric_name, ""),
                "message": "暂无请求数据",
            }
            continue

        # 对于 histogram 指标，额外检查 _count 是否为 0
        if metric_name in SUMMARY_COUNT_BUILDERS:
            try:
                count_promql = SUMMARY_COUNT_BUILDERS[metric_name](str(service_id), namespace)
                count_data = prom_client.instant_query(count_promql)
                if count_data and len(count_data) > 0:
                    count_val = prom_client._safe_float(count_data[0].get("value", [None, None])[1])
                    if count_val is not None and count_val <= 0:
                        result[metric_name] = {
                            "value": None,
                            "status": "no_data",
                            "unit": METRIC_UNITS.get(metric_name, ""),
                            "message": "暂无请求数据",
                        }
                        continue
            except Exception:
                pass

        raw_value = data[0].get("value", [None, None])[1]
        value = prom_client._safe_float(raw_value)
        if value is None:
            result[metric_name] = {
                "value": None,
                "status": "no_data",
                "unit": METRIC_UNITS.get(metric_name, ""),
                "message": "暂无请求数据",
            }
        else:
            result[metric_name] = {
                "value": value,
                "status": "normal",
                "unit": METRIC_UNITS.get(metric_name, ""),
                "message": "",
            }

    return result


def get_timeseries_metrics(prom_client, service_id, namespace, metric_names, range_str):
    """
    获取多个指标的时间序列数据。
    返回: {
        metric_name: [{timestamp, value}, ...]
    }
    """
    if range_str not in ALLOWED_RANGES:
        raise ValueError(f"Unsupported range: {range_str}")

    step = RANGE_STEP_MAP[range_str]
    range_seconds = _range_to_seconds(range_str)
    end = int(time.time())
    start = end - range_seconds

    result = {}
    for metric_name in metric_names:
        if metric_name not in TIMESERIES_PROMQL_BUILDERS:
            continue
        promql = TIMESERIES_PROMQL_BUILDERS[metric_name](str(service_id), namespace)
        try:
            data = prom_client.range_query(promql, start, end, step)
        except Exception:
            result[metric_name] = []
            continue

        if not data:
            result[metric_name] = []
            continue

        values = data[0].get("values", [])
        points = []
        for ts, val in values:
            v = prom_client._safe_float(val)
            # Prometheus 返回 Unix 秒级时间戳，前端 ECharts time 轴需要毫秒
            ts_float = prom_client._safe_float(ts)
            if ts_float is None or ts_float <= 0 or not math.isfinite(ts_float):
                continue
            # 校验时间戳在合理范围内（查询窗口 ±1 天）
            if ts_float < start - 86400 or ts_float > end + 86400:
                continue
            timestamp_ms = int(ts_float * 1000)
            points.append({
                "timestamp_ms": timestamp_ms,
                "value": v,
            })
        result[metric_name] = points

    return result


def _range_to_seconds(range_str):
    """将范围字符串转为秒数"""
    if range_str.endswith("m"):
        return int(range_str[:-1]) * 60
    elif range_str.endswith("h"):
        return int(range_str[:-1]) * 3600
    elif range_str.endswith("d"):
        return int(range_str[:-1]) * 86400
    else:
        return int(range_str)
