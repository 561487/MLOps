"""
推理监控业务 Service 层。
- 权限校验
- 状态判定
- 缓存管理
- 引擎适配器注册表
"""
from flask import g

from myapp import db, cache, conf, security_manager
from myapp.models.model_serving import InferenceService
from myapp.utils.py.py_prometheus import Prometheus
from myapp.services import vllm_metric_adapter
from myapp.services import sglang_metric_adapter
from myapp.services.inference_monitor_time_range import (
    QueryWindow,
    resolve_query_window,
    ALLOWED_RANGES,
)
from myapp.services.inference_monitor_metadata import (
    resolve_engine_type,
    resolve_metrics_endpoint,
    build_inference_monitor_metadata,
    validate_inference_monitor_metadata,
    ENGINE_ADAPTER_ALIASES,
    MONITORED_SERVICE_TYPES,
    INFERENCE_MONITOR_CONFIG,
)

# 共用 vLLM 的 SUPPORTED_METRICS 和 ALLOWED_RANGES（两个适配器一致）
SUPPORTED_METRICS = vllm_metric_adapter.SUPPORTED_METRICS

# 引擎适配器注册表：通过 service_type 路由到对应的适配器
ENGINE_ADAPTERS = {
    "vllm": vllm_metric_adapter,
    "sglang": sglang_metric_adapter,
}

# 向后兼容：旧代码引用
MONITORED_ENGINE_TYPES = frozenset(INFERENCE_MONITOR_CONFIG.keys())


def ensure_inference_monitor_registration(service):
    """
    幂等补齐推理服务的 Prometheus 监控 metadata。

    规则：
    - 仅处理 service_type 在 MONITORED_SERVICE_TYPES 中的服务
    - 使用 resolve_metrics_endpoint() 解析实际端口，绝不硬编码 8000
    - 仅在内部 Service 缺失 metadata 时通过 K8s API 补齐
    - 不修改 external Service
    - 不调用 shell kubectl

    返回:
        dict: {"action": "skipped"|"patched"|"error", "detail": str}
    """
    resolved_engine = resolve_engine_type(service.service_type)
    if not resolved_engine:
        return {"action": "skipped", "detail": f"引擎 {service.service_type} 不在监控范围内"}

    # 解析实际 metrics 端点（动态端口，不硬编码）
    endpoint = resolve_metrics_endpoint(service)
    port = endpoint.get("port")
    if not port:
        return {"action": "error", "detail": f"无法解析 metrics 端口"}

    # 确保数据库 metrics 字段已填充
    metrics_value = f"{port}:{endpoint['path']}"
    if not service.metrics or not service.metrics.strip():
        service.metrics = metrics_value
        try:
            from myapp import db
            db.session.flush()
        except Exception:
            pass

    service_id = str(service.id)
    service_name = str(service.name)

    # 获取 K8s 客户端
    try:
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service.project.cluster.get("KUBECONFIG", ""))
    except Exception as e:
        return {"action": "error", "detail": f"K8s 客户端初始化失败: {e}"}

    namespace = service.namespace or service.project.service_namespace
    internal_svc_name = service_name

    # 读取当前内部 Service
    try:
        svc = k8s_client.v1.read_namespaced_service(
            name=internal_svc_name, namespace=namespace, _request_timeout=5
        )
    except Exception as e:
        return {"action": "error", "detail": f"读取内部 Service 失败: {e}"}

    current_annotations = svc.metadata.annotations or {}
    current_labels = svc.metadata.labels or {}

    expected_annotations, expected_labels = build_inference_monitor_metadata(
        service_id, service_name, service.service_type,
        metrics_port=port, metrics_path=endpoint["path"],
    )

    # 检查是否需要更新
    missing_annotations = {
        k: v for k, v in expected_annotations.items()
        if current_annotations.get(k) != v
    }
    missing_labels = {
        k: v for k, v in expected_labels.items()
        if current_labels.get(k) != v
    }

    # 结构化日志：记录 ensure 调用的完整状态
    import logging
    _elog = logging.getLogger(__name__)
    _elog.info(
        "INFERENCE_MONITOR_ENSURE_TRACE service_id=%s service_name=%s expected_labels=%s "
        "current_labels=%s missing_labels=%s expected_annotations=%s current_annotations=%s "
        "missing_annotations=%s",
        service_id, service_name, expected_labels, current_labels, missing_labels,
        {k:v for k,v in expected_annotations.items() if 'prometheus' in k},
        {k:v for k,v in current_annotations.items() if 'prometheus' in k},
        {k:v for k,v in missing_annotations.items() if 'prometheus' in k},
    )

    if not missing_annotations and not missing_labels:
        # 即使 metadata 已完整，也要清理可能存在的负缓存
        _invalidate_monitor_cache(service_id)
        return {"action": "skipped", "detail": "metadata 已完整"}

    # 补齐缺失项
    patch_body = {"metadata": {}}
    if missing_annotations:
        patch_body["metadata"]["annotations"] = {**current_annotations, **missing_annotations}
    if missing_labels:
        patch_body["metadata"]["labels"] = {**current_labels, **missing_labels}

    try:
        k8s_client.v1.patch_namespaced_service(
            name=internal_svc_name, namespace=namespace, body=patch_body
        )
        detail_parts = []
        if missing_annotations:
            detail_parts.append(f"annotations: {sorted(missing_annotations.keys())}")
        if missing_labels:
            detail_parts.append(f"labels: {sorted(missing_labels.keys())}")
        # Patch 成功后，清理该 service_id 的所有监控缓存
        _invalidate_monitor_cache(service_id)
        return {"action": "patched", "detail": "; ".join(detail_parts)}
    except Exception as e:
        return {"action": "error", "detail": f"patch Service 失败: {e}"}


def _invalidate_monitor_cache(service_id_str):
    """清理指定 service_id 的所有推理监控缓存（规避负缓存导致页面显示未接入）"""
    from myapp import cache as _app_cache
    import logging
    _clog = logging.getLogger(__name__)
    patterns = [
        f"inference_monitor_summary:user_",
        f"inference_monitor_timeseries:user_",
    ]
    # 尝试匹配缓存 key（Redis FLASK_CACHE 前缀 + 模式匹配）
    try:
        # 获取所有可能的 user_* 匹配
        for pattern_prefix in patterns:
            for user_suffix in ['admin', 'anon']:
                candidate = f"{pattern_prefix}{user_suffix}:service_id_{service_id_str}"
                # 不是完整 key，但这是我们能构造的最佳候选
                # 实际 key 包含 user_<id>:service_id_<id>:...
                pass
        # 简单方式：删除按 user 前缀构造的常见 key
        if hasattr(_app_cache, 'delete'):
            for uid in ['admin', 'anon', '1']:
                try:
                    # summary cache key 格式: inference_monitor_summary:user_<uid>:service_id_<sid>
                    _app_cache.delete(f"inference_monitor_summary:user_{uid}:service_id_{service_id_str}")
                    _app_cache.delete(f"inference_monitor_target_status:user_{uid}:service_id_{service_id_str}")
                except Exception:
                    pass
        _clog.info("INFERENCE_MONITOR_CACHE_INVALIDATE service_id=%s", service_id_str)
    except Exception as _cache_err:
        _clog.warning("INFERENCE_MONITOR_CACHE_INVALIDATE_ERROR service_id=%s error=%s",
                      service_id_str, str(_cache_err))


def is_inference_service_running(service):
    """推理服务是否处于运行状态（可被监控）"""
    if service.model_status == 'offline':
        return False
    if not service.ready:
        return False
    return True


def _get_user_project_ids():
    """从 session 获取当前用户有权限的项目 ID 列表"""
    if g.user.is_admin():
        return None  # None 表示无限制
    return security_manager.get_join_projects_id(db.session)


def _get_prom_client():
    """获取 Prometheus 客户端实例，未配置时返回 None"""
    base_url = conf.get("PROMETHEUS_BASE_URL", "")
    if not base_url:
        return None
    timeout = int(conf.get("PROMETHEUS_QUERY_TIMEOUT", 10))
    return Prometheus(host=base_url, query_timeout=timeout)


def _build_cache_key(prefix, **kwargs):
    """构建缓存 Key，包含 user_id 防止跨用户数据泄露"""
    user_id = g.user.id if g.user and hasattr(g.user, 'id') else 'anon'
    parts = [prefix, f"user_{user_id}"]
    for key, value in sorted(kwargs.items()):
        parts.append(f"{key}_{value}")
    return ":".join(parts)


# 向后兼容别名
SUPPORTED_INFERENCE_ENGINES = {"vllm", "sglang"}


def _get_adapter(service_type):
    """根据 service_type 获取对应的指标适配器模块，支持别名。未知引擎返回 None"""
    resolved = resolve_engine_type(service_type)
    if not resolved:
        return None
    return ENGINE_ADAPTERS.get(resolved)


def _get_batch_up_query():
    """构建批量查询所有受监控引擎的 up 状态的 PromQL（合并两个引擎的查询）"""
    queries = []
    for engine, adapter in ENGINE_ADAPTERS.items():
        try:
            queries.append(adapter.build_batch_up_query())
        except AttributeError:
            continue
    if not queries:
        return ""
    return " or ".join(queries)


def get_services(project_id=None, engine_filter=None, status_filter=None):
    """
    获取用户可访问的推理服务列表 + 批量 up 状态。
    engine_filter: None/'all' → 全部受支持引擎；'vllm'/'sglang' → 单个引擎。
    """
    user_project_ids = _get_user_project_ids()
    query = db.session.query(InferenceService)

    # 引擎筛选：all 或不传 → 查询所有受监控类型（含别名如 vllm-distributed）
    if engine_filter and engine_filter != "all":
        resolved = resolve_engine_type(engine_filter)
        if resolved:
            # 用户选了规范化引擎 → 匹配所有映射到同一个 Adapter 的类型
            matching_types = [k for k, v in ENGINE_ADAPTER_ALIASES.items() if v == resolved]
            query = query.filter(InferenceService.service_type.in_(matching_types))
        else:
            query = query.filter(InferenceService.service_type.in_(list(MONITORED_SERVICE_TYPES)))
    else:
        query = query.filter(InferenceService.service_type.in_(list(MONITORED_SERVICE_TYPES)))

    # 权限过滤
    if user_project_ids is not None:
        query = query.filter(InferenceService.project_id.in_(user_project_ids))

    # 项目筛选
    if project_id:
        pid = int(project_id)
        if user_project_ids is not None and pid not in user_project_ids:
            return []
        query = query.filter(InferenceService.project_id == pid)

    # 运行状态筛选
    if status_filter == 'online':
        query = query.filter(InferenceService.model_status != 'offline')
    elif status_filter == 'offline':
        query = query.filter(InferenceService.model_status == 'offline')

    services = query.order_by(InferenceService.id.desc()).all()

    # 构建服务列表基本信息（不逐条调用 service.ready）
    service_list = []
    monitored_service_ids = []
    for service in services:
        item = {
            "service_id": service.id,
            "service_name": service.name,
            "label": service.label,
            "model_name": service.model_name,
            "model_status": service.model_status,
            "service_type": service.service_type,
            "project_id": service.project_id,
            "project_name": service.project.name if service.project else "",
            "namespace": service.namespace,
            "ready": None,  # 不在列表接口中逐条查询
            "up_status": None,
        }
        service_list.append(item)
        if resolve_engine_type(service.service_type) is not None:
            monitored_service_ids.append(str(service.id))

    # 批量查询 Prometheus up 状态
    up_map = {}
    if monitored_service_ids:
        prom_client = _get_prom_client()
        if prom_client:
            try:
                up_data = prom_client.instant_query(_get_batch_up_query())
                if up_data:
                    authorized_ids = {str(s.id) for s in services}
                    for item in up_data:
                        sid = item.get("metric", {}).get("mlops_service_id", "")
                        if sid in authorized_ids:
                            up_map[sid] = prom_client._safe_float(
                                item.get("value", [None, None])[1]
                            )
            except Exception:
                pass

    # 填充 up 状态
    for item in service_list:
        sid = str(item["service_id"])
        item["up_status"] = up_map.get(sid)

    return service_list


def get_service_summary(service_id, user_project_ids=None, force_refresh=False):
    """
    获取单个服务的摘要指标。先校验权限，再查 Prometheus。
    force_refresh=True 时绕过缓存，直接查询 Prometheus。
    """
    if user_project_ids is None:
        user_project_ids = _get_user_project_ids()

    service = db.session.query(InferenceService).filter_by(id=int(service_id)).first()
    if not service:
        return {"error": "service_not_found", "message": "服务不存在"}

    # 权限校验
    if user_project_ids is not None and service.project_id not in user_project_ids:
        return {"error": "access_denied", "message": "无权访问该服务"}

    # 状态判定
    monitor_status = _determine_monitor_status(service)

    result = {
        "service_id": service.id,
        "service_name": service.name,
        "label": service.label,
        "model_name": service.model_name,
        "model_status": service.model_status,
        "service_type": service.service_type,
        "namespace": service.namespace,
        "ready": service.ready if service.model_status != 'offline' else False,
        "monitor_status": monitor_status,
    }

    # 如果服务不可监控，直接返回
    if monitor_status in ("service_stopped", "unsupported", "not_configured"):
        result["metrics"] = {}
        return result

    # 缓存（force_refresh 时跳过缓存读取，但仍写入新结果）
    cache_key = _build_cache_key("inference_monitor_summary", service_id=str(service.id))
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            result["metrics"] = cached
            return result

    # 查询 Prometheus
    prom_client = _get_prom_client()
    if prom_client is None:
        result["monitor_status"] = "not_configured"
        result["metrics"] = {}
        return result

    # 获取对应引擎的适配器
    adapter = _get_adapter(service.service_type)
    if adapter is None:
        result["monitor_status"] = "unsupported_engine"
        result["metrics"] = {}
        return result

    # 先检查 up 状态
    up_data = prom_client.instant_query(
        adapter.build_target_up_query(str(service.id), service.namespace)
    )
    if up_data is None:
        result["monitor_status"] = "query_failed"
        result["metrics"] = {}
        return result
    if not up_data:
        result["monitor_status"] = "not_configured"
        result["metrics"] = {}
        return result

    up_value = prom_client._safe_float(up_data[0].get("value", [None, None])[1])
    if up_value == 0:
        result["monitor_status"] = "scrape_failed"
        result["metrics"] = {}
        return result

    # 获取指标（通过引擎适配器）
    metrics = adapter.get_summary_metrics(prom_client, str(service.id), service.namespace)
    result["metrics"] = metrics

    # 判断整体 monitor_status
    has_normal = any(m.get("status") == "normal" for m in metrics.values())
    has_any = any(m.get("value") is not None for m in metrics.values())
    if has_normal:
        result["monitor_status"] = "normal"
    elif has_any:
        result["monitor_status"] = "normal"
    else:
        result["monitor_status"] = "no_data"

    # 缓存
    ttl = int(conf.get("INFERENCE_MONITOR_SUMMARY_CACHE_TTL", 10))
    try:
        cache.set(cache_key, metrics, timeout=ttl)
    except Exception:
        pass

    return result


def get_service_timeseries(service_id, metric_names, range_str, user_project_ids=None, force_refresh=False):
    """
    获取单个服务的时间序列数据。先校验权限。
    force_refresh=True 时绕过缓存，直接查询 Prometheus。
    """
    if user_project_ids is None:
        user_project_ids = _get_user_project_ids()

    # 白名单校验
    if range_str not in ALLOWED_RANGES:
        raise ValueError(f"Unsupported range: {range_str}")
    for m in metric_names:
        if m not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {m}")

    service = db.session.query(InferenceService).filter_by(id=int(service_id)).first()
    if not service:
        return {"error": "service_not_found", "message": "服务不存在"}

    if user_project_ids is not None and service.project_id not in user_project_ids:
        return {"error": "access_denied", "message": "无权访问该服务"}

    if not is_inference_service_running(service):
        return {"error": "service_stopped", "message": "服务未运行"}

    # 获取对应引擎的适配器
    adapter = _get_adapter(service.service_type)
    if adapter is None:
        return {"error": "unsupported_engine", "message": f"不支持的推理引擎: {service.service_type}"}

    prom_client = _get_prom_client()
    if prom_client is None:
        return {"error": "not_configured", "message": "监控未配置"}

    # 解析公共时间窗口（所有引擎共用）
    qw = resolve_query_window(range_str)
    if qw is None:
        return {"error": "invalid_range", "message": f"不支持的时间范围: {range_str}"}

    # 缓存（force_refresh 时跳过缓存读取，但仍写入新结果）
    # v3：PromQL 算法升级为「固定 raw 窗口 + 展示桶峰值」，key 携带
    # display_bucket 与 raw_window，避免命中旧算法（桶平均稀释）的缓存
    cache_key = _build_cache_key(
        "inference_monitor_timeseries:v3",
        service_id=str(service.id),
        engine=str(service.service_type),
        metrics=",".join(sorted(metric_names)),
        range=range_str,
        display_step=str(qw.display_step_seconds),
        display_bucket=str(qw.display_bucket_seconds),
        raw_window=str(qw.raw_calculation_window_seconds),
        start=str(qw.start_seconds),
        end=str(qw.end_seconds),
    )
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # 统一接口：传入 QueryWindow，返回 (data, query_window_api_dict) 元组
    raw_result = adapter.get_timeseries_metrics(
        prom_client,
        str(service.id),
        service.namespace,
        metric_names,
        qw,
    )

    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        result_data, _query_api_dict = raw_result
    else:
        # 旧 Adapter 向后兼容：只返回 data dict
        result_data = raw_result

    # 构建完整 payload：窗口信息 + 数据
    payload = qw.to_api_dict() | {"data": result_data}

    ttl = int(conf.get("INFERENCE_MONITOR_TIMESERIES_CACHE_TTL", 30))
    try:
        cache.set(cache_key, payload, timeout=ttl)
    except Exception:
        pass

    return payload


def get_target_status(service_id, user_project_ids=None):
    """
    获取单个服务的 Prometheus Target 状态。
    """
    if user_project_ids is None:
        user_project_ids = _get_user_project_ids()

    service = db.session.query(InferenceService).filter_by(id=int(service_id)).first()
    if not service:
        return {"error": "service_not_found", "message": "服务不存在"}

    if user_project_ids is not None and service.project_id not in user_project_ids:
        return {"error": "access_denied", "message": "无权访问该服务"}

    prom_client = _get_prom_client()
    if prom_client is None:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "not_configured",
            "up": None,
            "message": "未配置 Prometheus 地址",
        }

    # 获取对应引擎的适配器
    adapter = _get_adapter(service.service_type)
    if adapter is None:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "not_configured",
            "up": None,
            "message": f"不支持的推理引擎: {service.service_type}",
        }

    up_data = prom_client.instant_query(
        adapter.build_target_up_query(str(service.id), service.namespace)
    )

    if up_data is None:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "query_failed",
            "up": None,
            "message": "Prometheus 查询失败",
        }

    if not up_data:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "not_configured",
            "up": None,
            "message": "未发现监控 Target",
        }

    up_value = prom_client._safe_float(up_data[0].get("value", [None, None])[1])
    if up_value == 1:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "normal",
            "up": 1,
            "message": "采集正常",
        }
    elif up_value == 0:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "scrape_failed",
            "up": 0,
            "message": "Target 已发现但抓取失败",
        }
    else:
        return {
            "service_id": service.id,
            "service_name": service.name,
            "monitor_status": "not_configured",
            "up": up_value,
            "message": "状态未知",
        }


def _determine_monitor_status(service):
    """
    判定服务的监控状态。
    优先级：service_stopped > unsupported > not_configured > (查询后) query_failed/scrape_failed/no_data/normal
    """
    if service.model_status == 'offline':
        return "service_stopped"
    if not service.ready:
        return "service_stopped"
    if resolve_engine_type(service.service_type) is None:
        return "unsupported"

    prom_client = _get_prom_client()
    if prom_client is None:
        return "not_configured"

    # 不做深度查询，留给调用方根据实际查询结果判定
    return "normal"  # 临时，实际在 get_summary 中进一步判定
