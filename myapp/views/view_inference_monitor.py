"""
推理监控 API View。
提供推理服务性能监控数据查询接口。
"""
from flask import g, jsonify, request
from flask_appbuilder.api import BaseApi, expose

from myapp import appbuilder, security_manager
from myapp.services.inference_monitor_service import (
    get_services,
    get_service_summary,
    get_service_timeseries,
    get_target_status,
)
from myapp.services.vllm_metric_adapter import SUPPORTED_METRICS, ALLOWED_RANGES


class InferenceMonitorApi(BaseApi):
    resource_name = "inference_monitor"
    route_base = "/inference_monitor/api"
    openapi_spec_tag = "推理监控"

    def _check_auth(self):
        """校验用户登录状态"""
        if not g.user or not g.user.get_id():
            return False
        return True

    def _json_response(self, result, status=0, message="success"):
        """统一 JSON 响应格式，遵循项目现有标准"""
        return jsonify({
            "status": status,
            "message": message,
            "result": result,
        })

    def _error_response(self, message, status_code=400, result=None):
        return self.response(
            status_code,
            status=-1,
            message=message,
            result=result or {},
        )

    @staticmethod
    def response(code, **kwargs):
        from flask import make_response
        import json
        resp = make_response(json.dumps(kwargs, ensure_ascii=False), code)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp

    @expose("/services", methods=["GET"])
    def api_services(self):
        """
        服务列表接口。
        GET /inference_monitor/api/services?project_id=1&engine=vllm&status=online
        批量查询 up 状态，不逐条调用 service.ready。
        """
        if not self._check_auth():
            return self._error_response("未登录", 401)

        project_id = request.args.get("project_id")
        engine_filter = request.args.get("engine")
        status_filter = request.args.get("status")

        # 白名单校验引擎
        if engine_filter and engine_filter not in ("vllm",):
            return self._error_response("不支持的引擎类型")

        try:
            services = get_services(
                project_id=project_id,
                engine_filter=engine_filter,
                status_filter=status_filter,
            )
        except Exception:
            return self._error_response("查询服务列表失败", 500)

        return self._json_response(services)

    @expose("/summary", methods=["GET"])
    def api_summary(self):
        """
        单个服务摘要指标。
        GET /inference_monitor/api/summary?service_id=123
        """
        if not self._check_auth():
            return self._error_response("未登录", 401)

        service_id = request.args.get("service_id")
        if not service_id:
            return self._error_response("缺少 service_id 参数")

        force_refresh = str(request.args.get("force_refresh", "0")).lower() in ("1", "true", "yes")

        try:
            result = get_service_summary(service_id, force_refresh=force_refresh)
        except Exception:
            return self._error_response("查询摘要指标失败", 500)

        if isinstance(result, dict) and "error" in result:
            return self._error_response(result.get("message", "请求失败"), 400, result)

        return self._json_response(result)

    @expose("/timeseries", methods=["GET"])
    def api_timeseries(self):
        """
        时间序列数据。
        GET /inference_monitor/api/timeseries?service_id=123&metrics=ttft_p95,itl_p95,qps&range=1h
        """
        if not self._check_auth():
            return self._error_response("未登录", 401)

        service_id = request.args.get("service_id")
        metrics_str = request.args.get("metrics", "")
        range_str = request.args.get("range", "5m")

        if not service_id:
            return self._error_response("缺少 service_id 参数")
        if not metrics_str:
            return self._error_response("缺少 metrics 参数")

        metric_names = [m.strip() for m in metrics_str.split(",") if m.strip()]

        # 白名单校验
        if range_str not in ALLOWED_RANGES:
            return self._error_response(f"不支持的时间范围: {range_str}")
        for m in metric_names:
            if m not in SUPPORTED_METRICS:
                return self._error_response(f"不支持的指标: {m}")

        force_refresh = str(request.args.get("force_refresh", "0")).lower() in ("1", "true", "yes")

        try:
            result = get_service_timeseries(service_id, metric_names, range_str, force_refresh=force_refresh)
        except Exception:
            return self._error_response("查询时间序列失败", 500)

        if isinstance(result, dict) and "error" in result:
            return self._error_response(result.get("message", "请求失败"), 400, result)

        return self._json_response(result)

    @expose("/target_status", methods=["GET"])
    def api_target_status(self):
        """
        Prometheus Target 状态。
        GET /inference_monitor/api/target_status?service_id=123
        """
        if not self._check_auth():
            return self._error_response("未登录", 401)

        service_id = request.args.get("service_id")
        if not service_id:
            return self._error_response("缺少 service_id 参数")

        try:
            result = get_target_status(service_id)
        except Exception:
            return self._error_response("查询 Target 状态失败", 500)

        if isinstance(result, dict) and "error" in result:
            return self._error_response(result.get("message", "请求失败"), 400, result)

        return self._json_response(result)


appbuilder.add_api(InferenceMonitorApi)
