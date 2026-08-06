"""
SGLang 指标 PromQL 适配器。
使用 sglang: 前缀的指标名。
所有 PromQL 使用 job + kubernetes_namespace + mlops_service_id 三重标签限定。

窗口语义：
- 真实指标在固定 raw_calculation_window（默认 120s）上计算（rate / histogram_quantile）
- 长时间范围仅通过 display_bucket 降采样：对每个展示桶取桶内峰值
  （max_over_time 子查询），只降低时间分辨率，不稀释事件幅值
- subquery 分辨率必须带时间单位（如 15s），裸数字会导致 PromQL 解析失败
"""
import math

from myapp.services.inference_monitor_time_range import (
    QueryWindow,
    TIME_RANGE_SPECS,
    resolve_query_window,
)
from myapp.services.inference_monitor_metric_utils import normalize_prometheus_matrix_values
from myapp import conf

# 真实指标计算窗口（秒）：摘要（instant）查询与范围查询的瞬时表达式统一使用。
# 可通过环境变量 INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS 覆盖
CALCULATION_WINDOW_SECONDS = int(
    conf.get("INFERENCE_MONITOR_CALCULATION_WINDOW_SECONDS", "120")
)

# 支持的指标白名单
SUPPORTED_METRICS = {
    "ttft_p50", "ttft_p95", "ttft_p99",
    "itl_p50", "itl_p95",
    "qps", "running_requests", "waiting_requests",
    "input_tokens_per_second", "output_tokens_per_second",
}

METRIC_UNITS: dict[str, str] = {
    "ttft_p50": "ms", "ttft_p95": "ms", "ttft_p99": "ms",
    "itl_p50": "ms/token", "itl_p95": "ms/token",
    "qps": "req/s",
    "running_requests": "requests", "waiting_requests": "requests",
    "input_tokens_per_second": "tokens/s", "output_tokens_per_second": "tokens/s",
}

# cu13 (SGLang 0.0.0.dev1) 提供 inter_token_latency_seconds Histogram，ITL 已支持。
# v0.4.10.post2 不提供该指标，Prometheus 查询返回空结果。
SGLANG_UNSUPPORTED_METRICS: frozenset[str] = frozenset()


def _escape_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label_filter(service_id, namespace, job="kubernetes-service-endpoints") -> str:
    return (
        f'job="{job}",'
        f'kubernetes_namespace="{_escape_label_value(namespace)}",'
        f'mlops_service_id="{_escape_label_value(service_id)}"'
    )


# ====== 瞬时表达式（固定 raw_calculation_window）======

def _instant_histogram_quantile(quantile: float, metric_name: str,
                                 service_id, namespace, raw_seconds: int) -> str:
    """histogram 分位数（秒→毫秒仅此一次），基于固定短窗口的 bucket 速率"""
    labels = _label_filter(service_id, namespace)
    return (
        f"1000 * histogram_quantile({quantile}, "
        f"sum by (le) ("
        f"rate({metric_name}_bucket{{{labels}}}[{raw_seconds}s])"
        f"))"
    )


def _instant_counter_rate(metric_name: str, service_id, namespace, raw_seconds: int) -> str:
    """Counter 真实瞬时速率，基于固定短窗口"""
    labels = _label_filter(service_id, namespace)
    return f"sum(rate({metric_name}{{{labels}}}[{raw_seconds}s]))"


def _instant_gauge_sum(metric_name: str, service_id, namespace) -> str:
    labels = _label_filter(service_id, namespace)
    return f"sum({metric_name}{{{labels}}})"


def _build_histogram_count(metric_name: str, service_id, namespace, raw_seconds: int) -> str:
    labels = _label_filter(service_id, namespace)
    return f"sum(increase({metric_name}_count{{{labels}}}[{raw_seconds}s]))"


# ====== 展示桶峰值（长窗口降采样）======

def _bucket_peak(instant_expr: str, bucket_seconds: int, scrape_seconds: int) -> str:
    """对瞬时表达式按展示桶取桶内峰值（max_over_time 子查询）。

    只降低时间分辨率，不稀释短事件幅值。
    若桶内无任何样本（histogram 全为 NaN / 序列不存在），结果为 NaN/缺省，
    由 normalize 转为 null 断线，不补 0。
    """
    return f"max_over_time(({instant_expr})[{bucket_seconds}s:{scrape_seconds}s])"


# ====== 摘要查询（instant query — 固定 calculation window）======

def get_summary_metrics(prom_client, service_id, namespace):
    """摘要查询使用固定 calculation window（默认 120s）。"""
    cw = CALCULATION_WINDOW_SECONDS
    result = {}

    for mn in SGLANG_UNSUPPORTED_METRICS:
        result[mn] = {"value": None, "status": "unsupported",
                       "unit": METRIC_UNITS.get(mn, ""),
                       "message": "当前 SGLang 版本未提供该指标"}

    summary_builders = {
        "ttft_p50": lambda: _instant_histogram_quantile(0.50, "sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "ttft_p95": lambda: _instant_histogram_quantile(0.95, "sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "ttft_p99": lambda: _instant_histogram_quantile(0.99, "sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "qps":      lambda: _instant_counter_rate("sglang:num_requests_total", service_id, namespace, cw),
        "running_requests": lambda: _instant_gauge_sum("sglang:num_running_reqs", service_id, namespace),
        "waiting_requests": lambda: _instant_gauge_sum("sglang:num_queue_reqs", service_id, namespace),
        "itl_p50":  lambda: _instant_histogram_quantile(0.50, "sglang:inter_token_latency_seconds", service_id, namespace, cw),
        "itl_p95":  lambda: _instant_histogram_quantile(0.95, "sglang:inter_token_latency_seconds", service_id, namespace, cw),
        "input_tokens_per_second":  lambda: _instant_counter_rate("sglang:prompt_tokens_total", service_id, namespace, cw),
        "output_tokens_per_second": lambda: _instant_counter_rate("sglang:generation_tokens_total", service_id, namespace, cw),
    }

    count_builders = {
        "ttft_p50": lambda: _build_histogram_count("sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "ttft_p95": lambda: _build_histogram_count("sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "ttft_p99": lambda: _build_histogram_count("sglang:time_to_first_token_seconds", service_id, namespace, cw),
        "itl_p50":  lambda: _build_histogram_count("sglang:inter_token_latency_seconds", service_id, namespace, cw),
        "itl_p95":  lambda: _build_histogram_count("sglang:inter_token_latency_seconds", service_id, namespace, cw),
    }

    for mn, builder in summary_builders.items():
        try:
            data = prom_client.instant_query(builder())
        except Exception:
            result[mn] = {"value": None, "status": "query_failed",
                           "unit": METRIC_UNITS.get(mn, ""),
                           "message": "监控服务暂时不可用"}
            continue

        if data is None:
            result[mn] = {"value": None, "status": "query_failed",
                           "unit": METRIC_UNITS.get(mn, ""),
                           "message": "监控服务暂时不可用"}
            continue
        if not data:
            result[mn] = {"value": None, "status": "no_data",
                           "unit": METRIC_UNITS.get(mn, ""),
                           "message": "暂无请求数据"}
            continue

        if mn in count_builders:
            try:
                cd = prom_client.instant_query(count_builders[mn]())
                if cd and len(cd) > 0:
                    cv = prom_client._safe_float(cd[0].get("value", [None, None])[1])
                    if cv is not None and cv <= 0:
                        result[mn] = {"value": None, "status": "no_data",
                                       "unit": METRIC_UNITS.get(mn, ""),
                                       "message": "暂无请求数据"}
                        continue
            except Exception:
                pass

        raw = data[0].get("value", [None, None])[1]
        v = prom_client._safe_float(raw)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            result[mn] = {"value": None, "status": "no_data",
                           "unit": METRIC_UNITS.get(mn, ""),
                           "message": "暂无请求数据"}
        else:
            result[mn] = {"value": v, "status": "normal",
                           "unit": METRIC_UNITS.get(mn, ""), "message": ""}
    return result


# ====== 范围查询（query_range — 固定 raw 窗口 + 展示桶峰值）======

def get_timeseries_metrics(prom_client, service_id, namespace, metric_names, qw: QueryWindow):
    """
    获取 SGLang 服务时间序列数据。
    qw: 公共 QueryWindow，由 Service 层统一生成。

    关键：真实指标在固定 raw_calculation_window 上计算，长窗口只对瞬时曲线
    做 display_bucket 桶内峰值降采样（max_over_time 子查询），
    同一事件在 5m~3d 各窗口中的幅值保持一致，只有时间分辨率变化。
    返回: (data_dict, query_window_api_dict)
    """
    raw = qw.raw_calculation_window_seconds
    bucket = qw.display_bucket_seconds
    scrape = qw.scrape_interval_seconds
    step_s = str(qw.display_step_seconds)

    def peak(expr: str) -> str:
        return _bucket_peak(expr, bucket, scrape)

    timeseries_builders = {
        "ttft_p50": lambda: peak(_instant_histogram_quantile(0.50, "sglang:time_to_first_token_seconds", service_id, namespace, raw)),
        "ttft_p95": lambda: peak(_instant_histogram_quantile(0.95, "sglang:time_to_first_token_seconds", service_id, namespace, raw)),
        "ttft_p99": lambda: peak(_instant_histogram_quantile(0.99, "sglang:time_to_first_token_seconds", service_id, namespace, raw)),
        "itl_p50":  lambda: peak(_instant_histogram_quantile(0.50, "sglang:inter_token_latency_seconds", service_id, namespace, raw)),
        "itl_p95":  lambda: peak(_instant_histogram_quantile(0.95, "sglang:inter_token_latency_seconds", service_id, namespace, raw)),
        "qps":      lambda: peak(_instant_counter_rate("sglang:num_requests_total", service_id, namespace, raw)),
        "running_requests": lambda: peak(_instant_gauge_sum("sglang:num_running_reqs", service_id, namespace)),
        "waiting_requests": lambda: peak(_instant_gauge_sum("sglang:num_queue_reqs", service_id, namespace)),
        "input_tokens_per_second":  lambda: peak(_instant_counter_rate("sglang:prompt_tokens_total", service_id, namespace, raw)),
        "output_tokens_per_second": lambda: peak(_instant_counter_rate("sglang:generation_tokens_total", service_id, namespace, raw)),
    }

    result = {}
    for mn in metric_names:
        if mn in SGLANG_UNSUPPORTED_METRICS:
            result[mn] = []
            continue
        if mn not in timeseries_builders:
            continue

        try:
            data = prom_client.range_query(timeseries_builders[mn](), qw.start_seconds, qw.end_seconds, step_s)
        except Exception:
            result[mn] = []
            continue

        if not data:
            result[mn] = []
            continue

        result[mn] = normalize_prometheus_matrix_values(data[0].get("values", []))

    return result, qw.to_api_dict()


# ====== 全局查询 ======

def build_batch_up_query() -> str:
    return (
        'max by (mlops_service_id, mlops_service_name, kubernetes_namespace) ('
        'up{job="kubernetes-service-endpoints", mlops_engine="sglang"}'
        ')'
    )


def build_target_up_query(service_id, namespace) -> str:
    return f'up{{{_label_filter(service_id, namespace)}}}'
