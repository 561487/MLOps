"""
推理监控 metadata 构建模块 — 零依赖。

此模块不导入 myapp 中的任何其他模块（Flask、DB、K8s、Adapter 均不导入）。
因此不会被循环导入阻塞，适合在任何部署路径的早期阶段安全导入。

仅供 view_inferenceserving.py、inference_monitor_service.py 等在模块顶部正常导入。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 固定推理监控配置：引擎 → 默认 metrics 端口/路径
INFERENCE_MONITOR_CONFIG = {
    "vllm": {"metrics": "8000:/metrics", "port": "8000", "path": "/metrics"},
    "sglang": {"metrics": "30000:/metrics", "port": "30000", "path": "/metrics"},
}

# 引擎别名映射：统一路由到正确的 Adapter
# vllm-distributed 使用与 vllm 相同的 vllm:* 指标协议，复用 vLLM Adapter
ENGINE_ADAPTER_ALIASES = {
    "vllm": "vllm",
    "vllm-distributed": "vllm",
    "sglang": "sglang",
}

# 统一监控作用域：包含所有引擎类型及其别名
MONITORED_SERVICE_TYPES = frozenset(ENGINE_ADAPTER_ALIASES.keys())

# 内部 Service 必需的 metadata
REQUIRED_LABELS = frozenset({
    "mlops-monitoring",
    "mlops_engine",
    "mlops_service_name",
    "mlops_service_id",
})

REQUIRED_ANNOTATIONS = frozenset({
    "prometheus.io/scrape",
    "prometheus.io/port",
    "prometheus.io/path",
})


def resolve_engine_type(service_type: str) -> str | None:
    """
    将可能含别名、大小写变体的 service_type 规范化为标准引擎类型。

    返回:
        标准引擎类型字符串 ('vllm' 或 'sglang')；若不在监控范围则返回 None。
    """
    if not service_type:
        return None
    return ENGINE_ADAPTER_ALIASES.get(str(service_type).lower())


def resolve_metrics_endpoint(service) -> dict:
    """
    解析推理服务的 metrics 端口和路径，按优先级：

    1. service.metrics （如 '8004:/metrics'）
    2. service.ports 的首个端口（如 '8004'）
    3. INFERENCE_MONITOR_CONFIG 中的引擎默认值（使用 resolved_engine）

    返回:
        {"port": str, "path": str}
    引擎不在监控范围时返回 {"port": None, "path": None}。
    """
    service_type = getattr(service, 'service_type', '')
    # 先规范化引擎类型，再查默认配置（否则 vllm-distributed 找不到配置）
    resolved_engine = resolve_engine_type(service_type)
    config = INFERENCE_MONITOR_CONFIG.get(resolved_engine or service_type, {})
    default_port = config.get("port", "") if config else ""
    default_path = config.get("path", "/metrics") if config else "/metrics"

    # 优先从 service.metrics 解析
    metrics_str = (getattr(service, 'metrics', '') or "").strip()
    if metrics_str and ":" in metrics_str:
        port_str, _, path_str = metrics_str.partition(":")
        port = port_str.strip()
        path = path_str.strip() if path_str.strip() else default_path
        if port.isdigit():
            return {"port": port, "path": path}

    # 其次从 service.ports 取首个端口
    ports_str = (getattr(service, 'ports', '') or "").strip()
    if ports_str:
        first_port = ports_str.replace("，", ",").split(",")[0].strip()
        if first_port.isdigit():
            return {"port": first_port, "path": default_path}

    # 最后回退到引擎默认值
    return {"port": default_port, "path": default_path}


def build_inference_monitor_metadata(
    service_id,
    service_name: str,
    service_type: str,
    metrics_port: str | None = None,
    metrics_path: str | None = None,
) -> tuple[dict, dict]:
    """
    为受监控推理引擎生成 K8s Service 的监控 annotations 和 labels。

    metrics_port/metrics_path: 可选覆盖。不传时从 INFERENCE_MONITOR_CONFIG 取默认值。

    返回:
        (annotations: dict, monitoring_labels: dict)
        若引擎不在监控范围或无法解析端口，返回 ({}, {})。

    此函数不包含任何 service_id/服务名/模型名的特判逻辑。
    """
    resolved_engine = resolve_engine_type(service_type)
    if not resolved_engine:
        logger.debug("inference_monitor_metadata: engine not monitored: %s", service_type)
        return {}, {}

    config = INFERENCE_MONITOR_CONFIG.get(resolved_engine, {})
    port = metrics_port or config.get("port", "")
    path = metrics_path or config.get("path", "/metrics")

    if not port:
        logger.warning("inference_monitor_metadata: no port resolved for %s/%s", service_id, service_type)
        return {}, {}

    annotations = {
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(port),
        "prometheus.io/path": str(path),
    }
    labels = {
        "mlops-monitoring": "true",
        "mlops_engine": str(resolved_engine),
        "mlops_service_name": str(service_name),
        "mlops_service_id": str(service_id),
    }
    return annotations, labels


def validate_inference_monitor_metadata(annotations: dict, labels: dict) -> list[str]:
    """
    校验 metadata 的完整性。返回缺失项列表；列表为空表示 metadata 完整。
    """
    missing = []

    for key in REQUIRED_ANNOTATIONS:
        if not annotations.get(key):
            missing.append(f"annotation:{key}")

    for key in REQUIRED_LABELS:
        if not labels.get(key):
            missing.append(f"label:{key}")

    return missing
