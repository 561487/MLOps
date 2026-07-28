"""
推理监控业务 Service 层。
- 权限校验
- 状态判定
- 缓存管理
"""
from flask import g

from myapp import db, cache, conf, security_manager
from myapp.models.model_serving import InferenceService
from myapp.utils.py.py_prometheus import Prometheus
from myapp.services.vllm_metric_adapter import (
    get_summary_metrics,
    get_timeseries_metrics,
    build_batch_up_query,
    build_target_up_query,
    SUPPORTED_METRICS,
    ALLOWED_RANGES,
)


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


# 第一阶段只支持已验证的普通 vLLM 引擎
SUPPORTED_INFERENCE_ENGINES = {"vllm"}


def get_services(project_id=None, engine_filter=None, status_filter=None):
    """
    获取用户可访问的 vLLM 推理服务列表 + 批量 up 状态。
    第一阶段只查询 service_type='vllm' 的普通推理服务。
    """
    user_project_ids = _get_user_project_ids()
    query = db.session.query(InferenceService)

    # 第一阶段：始终只查询 vLLM 引擎
    query = query.filter(InferenceService.service_type.in_(SUPPORTED_INFERENCE_ENGINES))

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
        if service.service_type in SUPPORTED_INFERENCE_ENGINES:
            monitored_service_ids.append(str(service.id))

    # 批量查询 Prometheus up 状态
    up_map = {}
    if monitored_service_ids:
        prom_client = _get_prom_client()
        if prom_client:
            try:
                up_data = prom_client.instant_query(build_batch_up_query())
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

    # 先检查 up 状态
    up_data = prom_client.instant_query(
        build_target_up_query(str(service.id), service.namespace)
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

    # 获取指标
    metrics = get_summary_metrics(prom_client, str(service.id), service.namespace)
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

    prom_client = _get_prom_client()
    if prom_client is None:
        return {"error": "not_configured", "message": "监控未配置"}

    # 缓存（force_refresh 时跳过缓存读取，但仍写入新结果）
    cache_key = _build_cache_key(
        "inference_monitor_timeseries",
        service_id=str(service.id),
        metrics=",".join(sorted(metric_names)),
        range=range_str,
    )
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return {"data": cached}

    data = get_timeseries_metrics(
        prom_client,
        str(service.id),
        service.namespace,
        metric_names,
        range_str,
    )

    ttl = int(conf.get("INFERENCE_MONITOR_TIMESERIES_CACHE_TTL", 30))
    try:
        cache.set(cache_key, data, timeout=ttl)
    except Exception:
        pass

    return {"data": data}


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

    up_data = prom_client.instant_query(
        build_target_up_query(str(service.id), service.namespace)
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
    if service.service_type not in SUPPORTED_INFERENCE_ENGINES:
        return "unsupported"

    prom_client = _get_prom_client()
    if prom_client is None:
        return "not_configured"

    # 不做深度查询，留给调用方根据实际查询结果判定
    return "normal"  # 临时，实际在 get_summary 中进一步判定
