"""
vLLM 指标 PromQL 适配器。
所有 PromQL 使用 job + kubernetes_namespace + mlops_service_id 三重标签限定。
"""
import math

from myapp.services.inference_monitor_time_range import (
    QueryWindow,
    TIME_RANGE_SPECS,
    resolve_query_window,
)
from myapp.services.inference_monitor_metric_utils import normalize_prometheus_matrix_values
from myapp import conf

# 固定计算窗口下限（秒）：摘要（instant）查询使用的窗口；范围查询改用
# QueryWindow.calculation_window_seconds（>= display_step，避免大 step 漏短事件）
# 可通过环境变量 INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS 覆盖
CALCULATION_WINDOW_SECONDS = int(
    conf.get("INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS", "120")
)

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


def _instant_histogram_quantile(quantile, metric_name, service_id, namespace, raw_seconds):
    """histogram 分位数（秒→毫秒仅此一次），基于固定短窗口的 bucket 速率"""
    labels = _label_filter(service_id, namespace)
    return (
        f"1000 * histogram_quantile({quantile}, "
        f"sum by (le) ("
        f"rate({metric_name}_bucket{{{labels}}}[{raw_seconds}s])"
        f"))"
    )


def _build_histogram_count(metric_name, service_id, namespace, raw_seconds):
    """构建 histogram _count 查询（用于判断是否有数据）"""
    labels = _label_filter(service_id, namespace)
    return (
        f"sum(increase({metric_name}_count{{{labels}}}[{raw_seconds}s]))"
    )


def _instant_counter_rate(metric_name, service_id, namespace, raw_seconds):
    """Counter 真实瞬时速率，基于固定短窗口"""
    labels = _label_filter(service_id, namespace)
    return f"sum(rate({metric_name}{{{labels}}}[{raw_seconds}s]))"


def _instant_gauge_sum(metric_name, service_id, namespace):
    """构建 sum(gauge) PromQL"""
    labels = _label_filter(service_id, namespace)
    return f"sum({metric_name}{{{labels}}})"


def _bucket_peak(instant_expr, bucket_seconds, scrape_seconds):
    """对瞬时表达式按展示桶取桶内峰值（max_over_time 子查询）。

    只降低时间分辨率，不稀释短事件幅值。subquery 分辨率必须带时间单位。
    桶内无样本时结果为 NaN/缺省，由 normalize 转为 null 断线，不补 0。
    """
    return f"max_over_time(({instant_expr})[{bucket_seconds}s:{scrape_seconds}s])"


# ====== 瞬时查询 PromQL 构建 ======

SUMMARY_PROMQL_BUILDERS = {
    "ttft_p50": lambda sid, ns: _instant_histogram_quantile(0.50, "vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "ttft_p95": lambda sid, ns: _instant_histogram_quantile(0.95, "vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "ttft_p99": lambda sid, ns: _instant_histogram_quantile(0.99, "vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "itl_p50": lambda sid, ns: _instant_histogram_quantile(0.50, "vllm:inter_token_latency_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "itl_p95": lambda sid, ns: _instant_histogram_quantile(0.95, "vllm:inter_token_latency_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "qps": lambda sid, ns: _instant_counter_rate("vllm:request_success_total", sid, ns, CALCULATION_WINDOW_SECONDS),
    "running_requests": lambda sid, ns: _instant_gauge_sum("vllm:num_requests_running", sid, ns),
    "waiting_requests": lambda sid, ns: _instant_gauge_sum("vllm:num_requests_waiting", sid, ns),
    "input_tokens_per_second": lambda sid, ns: _instant_counter_rate("vllm:prompt_tokens_total", sid, ns, CALCULATION_WINDOW_SECONDS),
    "output_tokens_per_second": lambda sid, ns: _instant_counter_rate("vllm:generation_tokens_total", sid, ns, CALCULATION_WINDOW_SECONDS),
}

# 瞬时查询对应的 _count 检查（仅 histogram 指标需要）
SUMMARY_COUNT_BUILDERS = {
    "ttft_p50": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "ttft_p95": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "ttft_p99": lambda sid, ns: _build_histogram_count("vllm:time_to_first_token_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "itl_p50": lambda sid, ns: _build_histogram_count("vllm:inter_token_latency_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
    "itl_p95": lambda sid, ns: _build_histogram_count("vllm:inter_token_latency_seconds", sid, ns, CALCULATION_WINDOW_SECONDS),
}


# ====== 范围查询 PromQL 构建 ======
# 所有 PromQL 由 _get_timeseries_with_qw() 根据 QueryWindow 动态构建


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


def get_timeseries_metrics(prom_client, service_id, namespace, metric_names, qw: QueryWindow):
    """
    获取多个指标的时间序列数据。
    qw: 统一 QueryWindow — 由 Service 层通过 resolve_query_window() 生成，
        所有引擎共用，Adapter 内部不再调用 time.time()。
    返回: (data_dict, query_window_api_dict)
    """
    return _get_timeseries_with_qw(prom_client, service_id, namespace, metric_names, qw)


def _get_timeseries_with_qw(prom_client, service_id, namespace, metric_names, qw: QueryWindow):
    """使用统一 QueryWindow 构建 PromQL 并返回 (data, query_window_api_dict)。

    关键：真实指标在固定 raw_calculation_window 上计算，长窗口只对瞬时曲线
    做 display_bucket 桶内峰值降采样（max_over_time 子查询），
    同一事件在 5m~3d 各窗口中的幅值保持一致，只有时间分辨率变化。
    """
    raw = qw.raw_calculation_window_seconds
    bucket = qw.display_bucket_seconds
    scrape = qw.scrape_interval_seconds
    # display step 仅用于 Prometheus query_range 的 step 参数（数据点密度）
    step_s = str(qw.display_step_seconds)

    def peak(expr):
        return _bucket_peak(expr, bucket, scrape)

    # 动态构建 PromQL（固定 raw 窗口 + 展示桶峰值）
    _qw_builders = {
        "ttft_p50": lambda: peak(_instant_histogram_quantile(0.50, "vllm:time_to_first_token_seconds", service_id, namespace, raw)),
        "ttft_p95": lambda: peak(_instant_histogram_quantile(0.95, "vllm:time_to_first_token_seconds", service_id, namespace, raw)),
        "ttft_p99": lambda: peak(_instant_histogram_quantile(0.99, "vllm:time_to_first_token_seconds", service_id, namespace, raw)),
        "itl_p50":  lambda: peak(_instant_histogram_quantile(0.50, "vllm:inter_token_latency_seconds", service_id, namespace, raw)),
        "itl_p95":  lambda: peak(_instant_histogram_quantile(0.95, "vllm:inter_token_latency_seconds", service_id, namespace, raw)),
        "qps":      lambda: peak(_instant_counter_rate("vllm:request_success_total", service_id, namespace, raw)),
        "running_requests": lambda: peak(_instant_gauge_sum("vllm:num_requests_running", service_id, namespace)),
        "waiting_requests": lambda: peak(_instant_gauge_sum("vllm:num_requests_waiting", service_id, namespace)),
        "input_tokens_per_second":  lambda: peak(_instant_counter_rate("vllm:prompt_tokens_total", service_id, namespace, raw)),
        "output_tokens_per_second": lambda: peak(_instant_counter_rate("vllm:generation_tokens_total", service_id, namespace, raw)),
    }

    result = {}
    for metric_name in metric_names:
        if metric_name not in _qw_builders:
            continue
        try:
            data = prom_client.range_query(_qw_builders[metric_name](), qw.start_seconds, qw.end_seconds, step_s)
        except Exception:
            result[metric_name] = []
            continue

        if not data:
            result[metric_name] = []
            continue

        result[metric_name] = normalize_prometheus_matrix_values(data[0].get("values", []))

    return result, qw.to_api_dict()

