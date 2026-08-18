import json
import os
import sys
import time
import traceback

import requests
from flask import request, jsonify, current_app
from flask_appbuilder import expose, BaseView
from flask_login import current_user
from sqlalchemy import or_

from myapp import db
from myapp.models.model_market import (
    ModelMarketModel,
    ModelMarketAction,
    ModelMarketService,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ok(data=None, message="success"):
    return jsonify({
        "code": 0,
        "message": message,
        "data": data or {}
    })


def fail(message, code=1, data=None, http_status=200):
    return jsonify({
        "code": code,
        "message": message,
        "data": data or {}
    }), http_status


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False)


def get_user_id():
    try:
        return current_user.id
    except Exception:
        return None


def _call_api(method, path, json_data=None, timeout=60):
    """Call platform internal API directly."""
    host = current_app.config.get("MODEL_MARKET_INTERNAL_HOST", "http://127.0.0.1")
    url = host.rstrip("/") + path
    cookies = request.cookies.to_dict()
    headers = {"Content-Type": "application/json"}
    if request.headers.get("Authorization"):
        headers["Authorization"] = request.headers.get("Authorization")
    return requests.request(method, url, json=json_data, cookies=cookies,
                            headers=headers, timeout=timeout)


def extract_href_from_html(value):
    """从 Notebook.name_url 这类 HTML 字符串中提取 href。"""
    if not value:
        return ""

    text = str(value).strip()

    # 完整 URL
    if text.startswith("http://") or text.startswith("https://"):
        return text

    # 协议相对 URL，例如 //10.121.177.20/notebook/...
    # 必须补 http:，避免前端当成普通 / 路径拼到 window.location.origin
    if text.startswith("//"):
        return "http:" + text

    # 普通站内相对路径
    if text.startswith("/"):
        return text

    # 如果是 <a href="...">...</a>，提取 href
    import re
    match = re.search(r'href=[\'"]([^\'"]+)[\'"]', text)
    if match:
        href = match.group(1).strip()
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("//"):
            return "http:" + href
        if href.startswith("/"):
            return href
        return href

    return ""


def get_notebook_jupyter_url(notebook, fallback_url):
    """
    复用平台 Notebook 原生跳转地址。
    优先使用 notebook.name_url，失败时回退到 notebook_shell。
    """
    if not notebook:
        return fallback_url

    try:
        raw_url = notebook.name_url
        url = extract_href_from_html(raw_url)
        if url:
            # The cluster host differs from the Docker frontend host in the
            # development deployment. Use the same-origin Nginx /notebook/
            # proxy so the browser keeps the Cube Studio login cookie.
            notebook_path_at = url.find("/notebook/")
            if notebook_path_at >= 0:
                return url[notebook_path_at:]
            return url
    except Exception:
        pass

    # 最后 fallback：只在 name_url 不可用时手动拼基础 lab 地址
    try:
        namespace = notebook.namespace or "jupyter"
        name = notebook.name
        if name:
            return f"/notebook/{namespace}/{name}/lab"
    except Exception:
        pass

    return fallback_url

def unwrap_api_result(result):
    """Extract inner data from platform API response."""
    if not isinstance(result, dict):
        return {}

    if isinstance(result.get("data"), dict) and result.get("data"):
        return result["data"]

    if isinstance(result.get("result"), dict) and result.get("result"):
        return result["result"]

    return result


def extract_target_id(data, result, *keys):
    for key in keys:
        if isinstance(data, dict) and data.get(key):
            return data.get(key)
        if isinstance(result, dict) and result.get(key):
            return result.get(key)
    return None


def create_action(model_id, action_type, req_json):
    action = ModelMarketAction(
        model_id=model_id,
        action_type=action_type,
        status="running",
        project_id=req_json.get("project_id"),
        namespace=req_json.get("namespace"),
        request_json=json_dumps(req_json),
        created_by=get_user_id(),
    )
    db.session.add(action)
    db.session.commit()
    return action


def mark_action_success(action, target_type, target_id, target_name, target_url, response):
    action.status = "success"
    action.target_type = target_type
    action.target_id = target_id
    action.target_name = target_name
    action.target_url = target_url
    action.response_json = json_dumps(response)
    db.session.commit()


def mark_action_failed(action, err):
    action.status = "failed"
    action.error_msg = str(err)
    db.session.commit()


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------

def normalize_memory(value, default="4G"):
    """Ensure memory value ends with 'G' for platform API compatibility."""
    if value is None or value == "":
        return default
    value = str(value).strip()
    upper = value.upper()
    if upper.endswith("G"):
        return value
    if upper.endswith("GB"):
        return value[:-2] + "G"
    if value.isdigit():
        return value + "G"
    return value


def normalize_resource_value(value, default):
    """Ensure resource value is a clean string, falling back to default."""
    if value is None or value == "":
        return str(default).strip() if default else "0"
    return str(value).strip()


def build_notebook_kwargs(model, params):
    name = params.get("name") or f"{model.name}-dev-{int(time.time())}"

    env = model.env()
    env.update(params.get("env") or {})
    # InferenceService 会为同一模型生成唯一版本号，容器路由必须使用同一版本。

    requested_image = params.get("image")
    # A model-detail page opened before the Notebook image was updated can still
    # submit the inference image. Never use that image for a development Pod.
    if not requested_image or requested_image == model.inference_image:
        requested_image = model.notebook_image

    kwargs = {
        "name": name,
        "describe": f"模型市场一键开发：{model.display_name}",
        "images": requested_image,
        "resource_cpu": normalize_resource_value(params.get("cpu"), model.default_cpu),
        "resource_memory": normalize_memory(params.get("memory"), model.default_memory),
        "resource_gpu": normalize_resource_value(params.get("gpu"), model.default_gpu),
        "volume_mount": params.get("volume_mount") or model.volume_mount or "",
        "working_dir": params.get("working_dir") or params.get("work_dir") or "/mnt",
        "expand": json_dumps({
            "source": "model_market",
            "model_market_id": model.id,
            "model_name": model.name,
            "model_path": model.model_path,
            "template": model.notebook_template,
            "project_id": params.get("project_id"),
            "namespace": params.get("namespace"),
        }),
    }

    # Always set project (default to 1 if not provided)
    kwargs["project"] = params.get("project_id") or 1

    return kwargs


def _safe_param(params, key, default=None):
    """Get param value without treating 0 as falsy."""
    v = params.get(key)
    if v is None or v == "":
        return default
    return v


def build_pipeline_kwargs(model, params):
    """DEPRECATED — now using copy_from_template instead."""
    name = params.get("name") or f"{model.name}-finetune-{int(time.time())}"

    kwargs = {
        "name": name,
        "describe": f"模型市场一键微调：{model.display_name}",
        "images": _safe_param("image") or model.finetune_image or model.notebook_image,
        "resource_cpu": normalize_resource_value(_safe_param("cpu"), model.default_cpu),
        "resource_memory": normalize_memory(_safe_param("memory"), model.default_memory),
        "resource_gpu": normalize_resource_value(_safe_param("gpu"), model.default_gpu),
        "expand": json_dumps({
            "source": "model_market",
            "model_market_id": model.id,
            "model_name": model.name,
            "dataset_id": _safe_param("dataset_id"),
            "epochs": _safe_param("epochs", 10),
            "batch_size": _safe_param("batch_size", 32),
            "learning_rate": str(_safe_param("learning_rate", 0.001)),
            "output_model_name": _safe_param("output_model_name") or f"{model.name}-finetuned",
            "auto_register": bool(params.get("auto_register", True)),
        }),
        "parameter": "{}",
        "dag_json": "{}",
    }
    kwargs["project"] = params.get("project_id") or 1
    return kwargs


def build_inference_kwargs(model, params):
    ts = int(time.time())
    name = params.get("name") or f"{model.name}-svc-{ts}"

    # Use unique model_version to avoid duplicate inferenceservice.name
    # Platform generates name as {model_name}-{model_version_without_v}
    mv = params.get("model_version") or model.default_version
    if params.get("name"):
        # User specified name explicitly, use unique version stamp
        mv = f"{mv}-{ts}"

    env = model.env()
    env.update(params.get("env") or {})
    env.setdefault("KUBEFLOW_MODEL_NAME", model.name)
    # Keep the container route aligned with the generated InferenceService version.
    env["KUBEFLOW_MODEL_VERSION"] = mv

    kwargs = {
        "name": name,
        "label": f"模型市场一键部署：{model.display_name}",
        "describe": f"模型市场一键部署：{model.display_name}",
        "model_name": model.name,
        "model_version": mv,
        "service_type": params.get("service_type") or "serving",
        "images": params.get("image") or model.inference_image,
        "model_path": params.get("model_path") or model.model_path,
        "resource_cpu": normalize_resource_value(params.get("cpu"), model.default_cpu),
        "resource_memory": normalize_memory(params.get("memory"), model.default_memory),
        "resource_gpu": normalize_resource_value(params.get("gpu"), model.default_gpu),
        "ports": str(params.get("ports") or model.default_ports),
        "volume_mount": params.get("volume_mount") or model.volume_mount or "dshm(emptyDir):/dev/shm,mnt(emptyDir):/mnt",
        "command": params.get("command") or model.command or "",
        "env": json_dumps(env),
        "model_status": "offline",
        "expand": json_dumps({
            "source": "model_market",
            "model_market_id": model.id,
            "model_name": model.name,
            "task_type": model.task_type,
            "demo_input_type": model.demo_input_type,
            "demo_output_type": model.demo_output_type,
            "api_schema": model.api_schema(),
            "project_id": params.get("project_id"),
            "namespace": params.get("namespace"),
        }),
    }

    kwargs["project"] = params.get("project_id") or 1

    return kwargs


# ---------------------------------------------------------------------------
# ModelMarketApiView
# ---------------------------------------------------------------------------

class ModelMarketApiView(BaseView):
    route_base = "/model_market/api"

    # ---- 模型列表 ----
    @expose("/models", methods=["GET"])
    def list_models(self):
        category = request.args.get("category")
        task_type = request.args.get("task_type")
        keyword = request.args.get("keyword")

        query = db.session.query(ModelMarketModel).filter(
            ModelMarketModel.status == "online"
        )

        if category:
            query = query.filter(ModelMarketModel.category == category)

        if task_type:
            query = query.filter(ModelMarketModel.task_type == task_type)

        if keyword:
            like = f"%{keyword}%"
            query = query.filter(
                or_(
                    ModelMarketModel.name.like(like),
                    ModelMarketModel.display_name.like(like),
                    ModelMarketModel.description.like(like),
                    ModelMarketModel.tags.like(like),
                )
            )

        models = query.order_by(ModelMarketModel.id.desc()).all()

        return ok({
            "items": [m.to_dict() for m in models],
            "total": len(models),
        })

    # ---- 模型详情 ----
    @expose("/models/<int:model_id>", methods=["GET"])
    def model_detail(self, model_id):
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model or model.status != "online":
            return fail("模型不存在或已下线", code=404, http_status=404)

        return ok(model.to_dict())

    # ---- 模型版本列表 ----
    @expose("/models/<int:model_id>/versions", methods=["GET"])
    def model_versions(self, model_id):
        """Return available model versions with deploy metadata."""
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        # Base model: use pre-configured inference parameters
        model_env = model.env()
        base_image = model.inference_image or ""
        base_model_path = model.model_path or ""
        base_command = model.command or "python server.py"
        base_workdir = model_env.get("WORKING_DIR") or "/"

        versions = [{
            "version": "base",
            "model_path": base_model_path,
            "source": "base",
            "label": f"{model.display_name} (基础模型)",
            "image": base_image,
            "command": base_command,
            "working_dir": base_workdir,
            "deployable": bool(base_image and base_model_path),
        }]

        # Collect finetuned versions from successful finetune actions
        finetune_actions = db.session.query(ModelMarketAction).filter_by(
            model_id=model_id,
            action_type="finetune",
            status="success",
        ).order_by(ModelMarketAction.id.desc()).limit(20).all()

        for action in finetune_actions:
            try:
                resp = json.loads(action.response_json) if action.response_json else {}
                name = resp.get("output_model_name")
                ver = resp.get("output_model_version")
                path = resp.get("output_model_path")
                deploy_meta = resp.get("deploy_metadata") or {}
                if not name and not path:
                    continue

                # Determine deployability: needs real .pt path from deploy_metadata
                real_pt_path = deploy_meta.get("model_path", "") if deploy_meta else ""
                deployable = bool(real_pt_path and real_pt_path.endswith('.pt'))
                reason = "" if deployable else "微调模型未找到可部署权重文件，请开启自动注册模型或手动注册模型后再部署"

                versions.append({
                    "version": ver or f"finetune-{action.id}",
                    "model_path": real_pt_path or path or model.model_path,
                    "source": "finetune",
                    "label": f"{name or '微调模型'} ({ver or ''})",
                    "pipeline_id": action.target_id,
                    "action_id": action.id,
                    "workflow_name": resp.get("workflow_name"),
                    "auto_register": resp.get("auto_register"),
                    "auto_deploy": resp.get("auto_deploy"),
                    "image": deploy_meta.get("image") or base_image,
                    "command": deploy_meta.get("command") or base_command,
                    "working_dir": deploy_meta.get("working_dir") or base_workdir,
                    "deployable": deployable,
                    "reason": reason,
                })
            except Exception:
                pass

        return ok({
            "model_id": model_id,
            "model_name": model.name,
            "versions": versions,
        })

    # ---- 模型已部署服务列表 ----
    @expose("/models/<int:model_id>/services", methods=["GET"])
    def model_services(self, model_id):
        """返回当前模型关联的推理服务列表"""
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        active_only = request.args.get("active_only", "false").lower() == "true"
        model_version = request.args.get("model_version")
        service_name = request.args.get("service_name")

        ACTIVE_STATES = ["created", "Pending", "Running", "Ready", "online", "offline", "test", "deploying"]
        DEAD_STATES = ["unloaded", "deleted", "failed", "stopped"]

        query = db.session.query(ModelMarketService).filter_by(model_id=model_id)

        if active_only:
            query = query.filter(
                ~ModelMarketService.service_status.in_(DEAD_STATES)
            )

        if model_version:
            query = query.filter(
                ModelMarketService.extra_json.like(f'%"model_version": "{model_version}"%')
            )

        if service_name:
            query = query.filter_by(service_name=service_name)

        services = query.order_by(ModelMarketService.id.desc()).all()

        # For active_only, verify platform inferenceservice still exists
        # Only auto-unload if the platform record is GONE — services that are
        # still deploying (created/Pending/Deploying) MUST stay active so the
        # user can unload them.
        if active_only:
            from myapp.models.model_serving import InferenceService as InfSvc
            cleaned_ids = []
            for svc in list(services):
                if svc.service_status not in ACTIVE_STATES:
                    continue
                # Check if platform service still exists
                plat_svc = db.session.query(InfSvc).get(svc.service_id)
                if not plat_svc:
                    # Platform service deleted from DB — auto-unload this market record
                    svc.service_status = "unloaded"
                    svc.endpoint = ""
                    extra = json.loads(svc.extra_json or "{}")
                    extra["endpoint"] = ""
                    extra["unload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    extra["unload_reason"] = "platform_service_deleted"
                    svc.extra_json = json_dumps(extra)
                    db.session.commit()
                    cleaned_ids.append(svc.id)
                    print(f"[model_market services] stale service {svc.id} (service_id={svc.service_id}) marked unloaded: platform inferenceservice not found")
                # If platform service exists, KEEP it active — even if endpoint empty / deploying
                # The frontend needs to show the card so the user can unload if stuck

            # Remove only truly-stale (platform-deleted) services from results
            services = [s for s in services if s.id not in cleaned_ids]

        result = []
        for svc in services:
            extra = json.loads(svc.extra_json or "{}")
            is_ready = svc.service_status == "Ready"
            result.append({
                "market_service_id": svc.id,
                "service_id": svc.service_id,
                "service_name": svc.service_name,
                "model_id": svc.model_id,
                "model_version": extra.get("model_version", ""),
                "model_path": extra.get("model_path", ""),
                "service_status": svc.service_status,
                "ready": is_ready,
                "endpoint": svc.endpoint or extra.get("endpoint", ""),
                "api_url": f"/model_market/api/services/{svc.id}/predict",
                "url": f"/frontend/service/model_market_group/service/{svc.id}",
            })

        return ok({"services": result, "total": len(result)})

    # ---- 在线体验 ----
    @expose("/models/<int:model_id>/experience", methods=["POST"])
    def experience(self, model_id):
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        if not model.support_experience:
            return fail("该模型不支持在线体验", code=400, http_status=400)

        service = db.session.query(ModelMarketService).filter_by(
            model_id=model.id
        ).order_by(ModelMarketService.id.desc()).first()

        if not service:
            return fail("暂无可体验服务，请先一键部署", code=400, http_status=400)

        return ok({
            "market_service_id": service.id,
            "service_id": service.service_id,
            "service_name": service.service_name,
            "status": service.service_status,
            "endpoint": service.endpoint,
            "pc_demo_url": service.pc_demo_url,
            "mobile_demo_url": service.mobile_demo_url,
        })

    # ---- 一键开发 ----
    @expose("/models/<int:model_id>/develop", methods=["POST"])
    def develop(self, model_id):
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        if not model.support_develop:
            return fail("该模型不支持一键开发", code=400, http_status=400)

        params = request.get_json(silent=True) or {}

        # ---- 名称去重 ----
        raw_name = (params.get("name") or "").strip()
        if raw_name:
            # Notebook names are reused as Pod, container and Service names.
            # Normalize user-friendly names such as "8.11" to DNS-1123 labels.
            import re
            raw_name = re.sub(r"[^a-z0-9-]+", "-", raw_name.lower())
            raw_name = re.sub(r"-+", "-", raw_name).strip("-")[:54].rstrip("-")
            if not raw_name:
                return fail("开发环境名称至少需要包含一个字母或数字", code=422, http_status=422)
            # 用户指定了名称 → 检查是否已存在
            exists = db.session.execute(
                db.text("SELECT id FROM notebook WHERE name = :name LIMIT 1"),
                {"name": raw_name}
            ).fetchone()
            if exists:
                return fail(
                    f"开发环境名称「{raw_name}」已存在，请修改名称后重试",
                    code=422,
                    http_status=422,
                )
        else:
            # 未指定名称 → 自动生成唯一名称
            raw_name = f"{model.name}-dev-{int(time.time())}"

        params["name"] = raw_name
        # ------------------------

        action = create_action(model.id, "develop", params)

        try:
            kwargs = build_notebook_kwargs(model, params)
            result = _call_api("POST", "/notebook_modelview/api/", kwargs).json()
            data = unwrap_api_result(result)

            notebook_id = extract_target_id(data, result, "id", "notebook_id")
            notebook_name = data.get("name") or kwargs["name"]

            if not notebook_id:
                raise RuntimeError(f"Notebook 创建成功但未返回 notebook_id，原始响应：{json_dumps(result)[:500]}")

            target_url = f"/frontend/dev/dev_online/notebook_shell?id={notebook_id}"

# 创建成功后，查询 Notebook 对象，复用平台原生 JupyterLab 跳转地址
            from myapp.models.model_notebook import Notebook
            notebook_obj = db.session.query(Notebook).get(notebook_id)

            jupyter_url = get_notebook_jupyter_url(notebook_obj, target_url)

            mark_action_success(
                action,
                target_type="notebook",
                target_id=notebook_id,
                target_name=notebook_name,
                target_url=target_url,
                response=result,
            )

            return ok({
                "action_id": action.id,
                "target_type": "notebook",
                "target_id": notebook_id,
                "target_name": notebook_name,
                "url": target_url,
                "jupyter_url": jupyter_url,
            }, "Notebook 创建成功")

        except Exception as e:
            traceback.print_exc()
            err_str = str(e)
            # 兜底：ops-sdk 透了 DB 1062 → 转友好提示
            if "1062" in err_str and "Duplicate" in err_str:
                err_str = f"开发环境名称「{raw_name}」已存在，请修改名称后重试"
            mark_action_failed(action, err_str)
            return fail(err_str, code=500, http_status=500)
            return fail(f"Notebook 创建失败：{str(e)}", code=500, http_status=500)

    # ---- 一键微调 ----
    @expose("/models/<int:model_id>/finetune", methods=["POST"])
    def finetune(self, model_id):
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        if not model.support_finetune:
            return fail("该模型不支持一键微调", code=400, http_status=400)

        params = request.get_json(silent=True) or {}
        action = create_action(model.id, "finetune", params)

        try:
            import datetime as dt_mod, uuid as uuid_mod

            # 1. Find template pipeline
            from myapp.models.model_job import Pipeline as PipelineModel, Task
            from myapp.views.view_pipeline import run_pipeline as run_pipeline_fn, dag_to_pipeline

            template_name = model.finetune_template or "yolov8"
            template = db.session.query(PipelineModel).filter_by(name=template_name).first()
            if not template:
                return fail(
                    f"微调模板 '{template_name}' 不存在，请先在平台创建可运行的 YOLOv8 微调 Pipeline 模板",
                    code=400, http_status=400,
                )

            template_tasks = db.session.query(Task).filter_by(pipeline_id=template.id).all()
            if not template_tasks:
                return fail(
                    f"微调模板 '{template_name}' 没有任务节点，无法启动微调任务",
                    code=400, http_status=400,
                )

            # Read auto_register before task filtering
            auto_register = bool(params.get("auto_register", True))
            auto_deploy = bool(params.get("auto_deploy", False))
            print(f"[model_market finetune] auto_register={auto_register} auto_deploy={auto_deploy}")

            # Filter out deploy tasks when auto_register is disabled
            if not auto_register:
                DEPLOY_KEYWORDS = ['deploy', 'web-service', 'register', '注册']
                filtered_tasks = []
                removed_tasks = []
                for t in template_tasks:
                    name_lower = (t.name or '').lower()
                    label_lower = (t.label or '').lower()
                    is_deploy = any(kw in name_lower or kw in label_lower for kw in DEPLOY_KEYWORDS)
                    if is_deploy:
                        removed_tasks.append(t.name)
                    else:
                        filtered_tasks.append(t)
                print(f"[model_market finetune] auto_register=false, removed tasks: {removed_tasks}, kept: {[t.name for t in filtered_tasks]}")
                template_tasks = filtered_tasks

            # 2. Copy pipeline from template
            new_name = f"admin-pipeline-{int(time.time() * 1000)}-{uuid_mod.uuid4().hex[:4]}"
            new_pipeline = PipelineModel()
            # Copy pipeline-level fields from template
            for col in ['describe', 'dag_json', 'namespace', 'project_id',
                         'node_selector', 'image_pull_policy',
                         'schedule_type', 'cron_time', 'depends_on_past', 'max_active_runs',
                         'parallelism', 'global_env', 'alert_user']:
                if hasattr(template, col):
                    setattr(new_pipeline, col, getattr(template, col))

            new_pipeline.name = new_name
            new_pipeline.describe = f"模型市场一键微调：{model.display_name}"
            new_pipeline.created_by_fk = get_user_id() or 1
            new_pipeline.changed_by_fk = get_user_id() or 1
            new_pipeline.created_on = dt_mod.datetime.now()
            new_pipeline.changed_on = dt_mod.datetime.now()
            new_pipeline.parameter = "{}"
            new_pipeline.expired_limit = 0
            new_pipeline.alert_status = template.alert_status or ''

            # Inject model market params
            ts = int(time.time())
            output_model_name = _safe_param(params, "output_model_name") or f"{model.name}-finetuned"
            output_model_version = f"finetune-{ts}"
            output_model_path = f"/mnt/models/{model.name}/{output_model_version}"
            dataset_id = _safe_param(params, "dataset_id") or model.demo_dataset_url

            market_params = {
                "source": "model_market",
                "model_market_id": model.id,
                "model_name": model.name,
                "task_type": model.task_type,
                "base_model_path": model.model_path,
                "dataset_id": dataset_id,
                "epochs": _safe_param(params, "epochs", 10),
                "batch_size": _safe_param(params, "batch_size", 32),
                "learning_rate": str(_safe_param(params, "learning_rate", 0.001)),
                "output_model_name": output_model_name,
                "output_model_version": output_model_version,
                "output_model_path": output_model_path,
                "auto_register": auto_register,
                "auto_deploy": auto_deploy,
                "project_id": params.get("project_id"),
                "namespace": params.get("namespace"),
            }
            # Copy template's expand (vison editor node array), filter deploy nodes when needed
            template_expand_raw = json.loads(template.expand) if template.expand else []
            if not auto_register and isinstance(template_expand_raw, list):
                # Collect names of tasks that were filtered out
                removed_names = set()
                for item in template_expand_raw:
                    if isinstance(item, dict) and item.get("type") != "yolov8__edge" and "data" in item:
                        node_name = (item.get("data", {}).get("name") or "").lower()
                        if any(kw in node_name for kw in ['deploy', 'web-service', 'register']):
                            removed_names.add(item.get("id", ""))
                # Remove deploy nodes and edges connected to them
                filtered_expand = []
                for item in template_expand_raw:
                    if not isinstance(item, dict):
                        filtered_expand.append(item)
                        continue
                    # Skip deploy nodes
                    if item.get("id", "") in removed_names:
                        continue
                    # Skip edges that connect to/from removed nodes
                    if item.get("source", "") in removed_names or item.get("target", "") in removed_names:
                        continue
                    filtered_expand.append(item)
                print(f"[model_market finetune] auto_register=false, expand: removed node ids={list(removed_names)}, nodes remaining={len([i for i in filtered_expand if i.get('type')!='yolov8__edge'])}")
                new_pipeline.expand = json_dumps(filtered_expand)
            else:
                new_pipeline.expand = template.expand  # Copy as-is when auto_register=true
            new_pipeline.parameter = json_dumps(market_params)

            db.session.add(new_pipeline)
            db.session.commit()

            # 3. Copy tasks from template
            old_to_new = {}
            for task in template_tasks:
                new_task = Task()
                for col in ['label', 'job_template_id', 'working_dir', 'command', 'args',
                             'volume_mount', 'node_selector', 'resource_memory', 'resource_cpu',
                             'resource_gpu', 'timeout', 'retry', 'outputs', 'monitoring',
                             'expand', 'namespace', 'resource_rdma']:
                    if hasattr(task, col):
                        setattr(new_task, col, getattr(task, col))
                new_task.pipeline_id = new_pipeline.id
                new_task.name = f"{task.name}-{uuid_mod.uuid4().hex[:4]}"
                new_task.label = task.label or task.name
                new_task.created_on = dt_mod.datetime.now()
                new_task.changed_on = dt_mod.datetime.now()
                new_task.created_by_fk = get_user_id() or 1
                new_task.changed_by_fk = get_user_id() or 1
                db.session.add(new_task)
                db.session.commit()
                old_to_new[task.id] = new_task.id

            # Update DAG JSON: replace old task NAMES with new names (dag_json keys are task names)
            template_dag = json.loads(template.dag_json) if template.dag_json else {}
            if isinstance(template_dag, dict):
                # Build old_name -> new_name mapping from old_to_new (which is old_id -> new_id)
                name_map = {}
                for old_id, new_id in old_to_new.items():
                    old_task = db.session.query(Task).get(old_id)
                    new_task = db.session.query(Task).get(new_id)
                    if old_task and new_task:
                        name_map[old_task.name] = new_task.name

                new_dag = {}
                for old_name, node_data in template_dag.items():
                    new_name = name_map.get(old_name, old_name)
                    new_node = dict(node_data) if isinstance(node_data, dict) else {}
                    # Remap upstream task names
                    if isinstance(new_node.get('upstream'), list):
                        new_node['upstream'] = [
                            name_map.get(u, u) for u in new_node['upstream']
                        ]
                    new_dag[new_name] = new_node

                # Remove deploy-task entries from DAG when auto_register is off
                # (entries where old_name is not in name_map were not copied as tasks)
                if not auto_register:
                    removed_dag_keys = [old_name for old_name in template_dag
                                        if old_name not in name_map]
                    for rk in removed_dag_keys:
                        new_dag.pop(rk, None)
                    # Also clean upstream references to removed task names
                    for k in new_dag:
                        up = new_dag[k].get('upstream', [])
                        if isinstance(up, list):
                            new_dag[k]['upstream'] = [
                                u for u in up
                                if u not in removed_dag_keys
                            ]

                new_pipeline.dag_json = json_dumps(new_dag)
                print(f"[model_market finetune] dag_json name_map={name_map}")
                print(f"[model_market finetune] dag_json keys={list(new_dag.keys())}")
                for k, v in new_dag.items():
                    print(f"  dag[{k}] upstream={v.get('upstream', [])}")
            else:
                new_pipeline.dag_json = template.dag_json

            # Update expand (vison editor node array) with new task IDs, names + edge references
            # Use already-filtered expand from new_pipeline (not template) to preserve auto_register filtering
            current_expand = json.loads(new_pipeline.expand) if new_pipeline.expand else []
            if isinstance(current_expand, list):
                new_expand = json.loads(json_dumps(current_expand))
                # Build lookup: old_id (str) → new_id (str)  AND  old_name → new_name
                id_map = {str(old): str(new) for old, new in old_to_new.items()}
                # name_map already built above from dag_json section; rebuild if not present
                expand_name_map = {}
                for old_id, new_id in old_to_new.items():
                    old_task = db.session.query(Task).get(old_id)
                    new_task = db.session.query(Task).get(new_id)
                    if old_task and new_task:
                        expand_name_map[old_task.name] = new_task.name

                for node in new_expand:
                    if isinstance(node, dict):
                        # Update vison node IDs
                        old_id = str(node.get("id", ""))
                        if old_id in id_map:
                            node["id"] = id_map[old_id]
                        # Update data.name to match new task name
                        if isinstance(node.get("data"), dict):
                            old_data_name = node["data"].get("name", "")
                            if old_data_name in expand_name_map:
                                node["data"]["name"] = expand_name_map[old_data_name]
                        # Update edge source/target references
                        for key in ("source", "target"):
                            old_ref = str(node.get(key, ""))
                            if old_ref in id_map:
                                node[key] = id_map[old_ref]
                new_pipeline.expand = json_dumps(new_expand)
            db.session.commit()

            print(f"[model_market finetune] expand_name_map={expand_name_map}")

            # 4. Generate DAG and run pipeline
            new_pipeline.pipeline_file, new_pipeline.run_id = dag_to_pipeline(
                new_pipeline, db.session, workflow_label={"schedule_type": "once"}
            )
            if not new_pipeline.pipeline_file:
                return fail("Pipeline DAG 生成失败，请检查模板任务配置", code=500, http_status=500)

            crd_name = run_pipeline_fn(new_pipeline, json.loads(new_pipeline.pipeline_file))
            new_pipeline.pipeline_argo_id = crd_name
            db.session.commit()
            workflow_name = crd_name

            # Log created tasks for debugging
            created_tasks = db.session.query(Task).filter_by(pipeline_id=new_pipeline.id).order_by(Task.id).all()
            created_dag = json.loads(new_pipeline.dag_json) if new_pipeline.dag_json else {}
            print(f"[model_market finetune] Pipeline id={new_pipeline.id}, tasks ({len(created_tasks)}):")
            for ct in created_tasks:
                upstream = created_dag.get(ct.name, {}).get('upstream', [])
                print(f"  - {ct.name} (label={ct.label}) upstream={upstream}")

            # 5. Build DAG page URL
            namespace = new_pipeline.namespace or "pipeline"
            target_url = f"/frontend/commonRelation?backurl=/workflow_modelview/api/web/dag/dev/{namespace}/{workflow_name}"

            # Extract deploy metadata from pipeline deploy task (if auto_register=true)
            deploy_metadata = {}
            if auto_register:
                try:
                    deploy_tasks = db.session.query(Task).filter(
                        Task.pipeline_id == new_pipeline.id,
                        Task.name.like('%deploy%')
                    ).all()
                    for dt in deploy_tasks:
                        try:
                            dt_args = json.loads(dt.args) if dt.args else {}
                        except Exception:
                            dt_args = {}
                        real_model_path = dt_args.get('--model_path', '') or dt_args.get('model_path', '')
                        real_image = dt_args.get('--images', '') or dt_args.get('images', '')
                        real_command = dt_args.get('--command', '') or dt_args.get('command', '')
                        real_workdir = dt_args.get('--working_dir', '') or dt_args.get('working_dir', '')
                        if real_model_path:
                            deploy_metadata = {
                                "model_path": real_model_path,
                                "image": real_image or model.inference_image or '',
                                "command": real_command or 'python server.py',
                                "working_dir": real_workdir or '/yolov8',
                            }
                            print(f"[model_market finetune] deploy_metadata extracted from task {dt.name}: {deploy_metadata}")
                            break
                except Exception as e:
                    print(f"[model_market finetune] deploy_metadata extraction failed: {e}")

            finetune_response = {
                "workflow_name": workflow_name,
                "dataset_id": dataset_id,
                "output_model_name": output_model_name,
                "output_model_version": output_model_version,
                "output_model_path": output_model_path,
                "auto_register": auto_register,
                "auto_deploy": auto_deploy,
                "template_name": template_name,
                "task_count": len(template_tasks),
                "deploy_metadata": deploy_metadata,
            }

            mark_action_success(
                action,
                target_type="pipeline",
                target_id=new_pipeline.id,
                target_name=new_pipeline.name,
                target_url=target_url,
                response=finetune_response,
            )

            pipeline_edit_url = f"/frontend/showOutLink?url=%2Fstatic%2Fappbuilder%2Fvison%2Findex.html%3Fpipeline_id%3D{new_pipeline.id}"

            return ok({
                "action_id": action.id,
                "target_type": "pipeline",
                "target_id": new_pipeline.id,
                "target_name": new_pipeline.name,
                "url": target_url,
                "pipeline_edit_url": pipeline_edit_url,
                "workflow_name": workflow_name,
                "status": "Running",
                "dataset_id": dataset_id,
                "output_model_name": output_model_name,
                "output_model_version": output_model_version,
                "output_model_path": output_model_path,
                "auto_register": auto_register,
                "auto_deploy": auto_deploy,
            }, "微调 Pipeline 已创建并启动")

        except Exception as e:
            traceback.print_exc()
            mark_action_failed(action, str(e))
            return fail(f"微调 Pipeline 创建失败：{str(e)}", code=500, http_status=500)

    # ---- 一键部署 ----
    @expose("/models/<int:model_id>/deploy", methods=["POST"])
    def deploy(self, model_id):
        model = db.session.query(ModelMarketModel).get(model_id)
        if not model:
            return fail("模型不存在", code=404, http_status=404)

        if not model.support_deploy:
            return fail("该模型不支持一键部署", code=400, http_status=400)

        params = request.get_json(silent=True) or {}
        service_name = params.get("name") or f"{model.name}-svc"

        # Check for existing active service (idempotent) — unloaded/deleted/failed are NOT active
        ACTIVE_STATES = ["created", "Pending", "Running", "Ready", "online", "offline", "test", "deploying"]
        existing = db.session.query(ModelMarketService).filter_by(
            model_id=model.id,
            service_name=service_name,
        ).filter(
            ModelMarketService.service_status.in_(ACTIVE_STATES)
        ).order_by(ModelMarketService.id.desc()).first()

        if existing:
            svc_url = f"/frontend/service/model_market_group/service/{existing.id}"
            extra = json.loads(existing.extra_json or "{}")
            return ok({
                "action_id": existing.action_id,
                "target_type": "inferenceservice",
                "target_id": existing.service_id,
                "target_name": existing.service_name,
                "market_service_id": existing.id,
                "url": svc_url,
                "status_url": f"/model_market/api/services/{existing.id}/status",
                "api_url": f"/model_market/api/services/{existing.id}/predict",
                "api_example_url": f"/model_market/api/services/{existing.id}/api-example",
                "stats_url": f"/model_market/api/services/{existing.id}/stats",
                "model_version": extra.get("model_version", ""),
                "model_path": extra.get("model_path", ""),
                "reused": True,
            }, "服务已存在，已返回现有服务")

        action = create_action(model.id, "deploy", params)

        try:
            model_path = _safe_param(params, "model_path") or model.model_path
            model_version = params.get("model_version") or "base"
            image = _safe_param(params, "image") or model.inference_image
            command = _safe_param(params, "command") or model.command or "python server.py"
            model_env = model.env()
            working_dir = _safe_param(params, "working_dir") or model_env.get("WORKING_DIR") or "/"

            is_finetune = model_version.startswith("finetune") or "finetune" in str(model_version)

            if is_finetune:
                # Finetune versions MUST use the real model_path from version metadata
                # (extracted from Pipeline deploy task args). Never guess /best.pt.
                if not model_path or not model_path.lower().endswith('.pt'):
                    return fail(
                        "微调模型未找到可部署权重文件，请开启自动注册模型或手动注册模型后再部署。",
                        code=400, http_status=400,
                        data={
                            "model_version": model_version,
                            "model_path": model_path or "",
                            "reason": "missing_weight_file"
                        }
                    )
                # Always use the inference image, not training image
                if not image:
                    image = model.inference_image or "10.121.177.20:8082/mlops/yolov8:20250801"
                print(f"[model_market deploy] finetune: model_path={model_path} image={image} command={command} workdir={working_dir}")
            else:
                # Base models may be a file (.pt/.mar) or a directory (Hugging Face/CT2).
                if not model_path:
                    return fail("基础模型路径未配置", code=400, http_status=400)
                if not image:
                    return fail("推理镜像未配置", code=400, http_status=400)
                print(f"[model_market deploy] base: model_path={model_path} image={image}")

            kwargs = build_inference_kwargs(model, params)
            kwargs["model_path"] = model_path
            kwargs["images"] = image
            kwargs["command"] = command
            kwargs["working_dir"] = working_dir

            result = _call_api("POST", "/inferenceservice_modelview/api/", kwargs).json()
            data = unwrap_api_result(result)

            service_id = extract_target_id(data, result, "id", "service_id")
            svc_name = data.get("name") or kwargs["name"]
            model_version = kwargs.get("model_version", "base")

            if not service_id:
                raise RuntimeError(f"InferenceService 创建成功但未返回 service_id，原始响应：{json_dumps(result)[:500]}")

            market_service = ModelMarketService(
                model_id=model.id,
                action_id=action.id,
                service_id=service_id,
                service_name=svc_name,
                service_status="created",
                project_id=params.get("project_id"),
                namespace=params.get("namespace"),
                endpoint="",
                pc_demo_url=f"/model-market/demo/pc/{service_id}",
                mobile_demo_url=f"/model-market/demo/mobile/{service_id}",
                api_schema_json=model.api_schema_json,
                extra_json=json_dumps({
                    "model_id": model.id,
                    "model_name": model.name,
                    "model_version": model_version,
                    "model_path": model_path,
                    "task_type": model.task_type,
                    "demo_input_type": model.demo_input_type,
                    "demo_output_type": model.demo_output_type,
                    "call_count": 0, "success_count": 0, "failed_count": 0,
                }),
                created_by=get_user_id(),
            )
            db.session.add(market_service)
            db.session.commit()

            # Trigger deployment: call platform deploy/prod to create K8s resources
            try:
                _call_api("POST", f"/inferenceservice_modelview/api/deploy/prod/{service_id}")
            except Exception as dep_err:
                print(f"Warning: InferenceService deploy trigger failed: {dep_err}")

            service_url = f"/frontend/service/model_market_group/service/{market_service.id}"

            mark_action_success(
                action,
                target_type="inferenceservice",
                target_id=service_id,
                target_name=svc_name,
                target_url=service_url,
                response={
                    "market_service_id": market_service.id,
                    "service_id": service_id,
                    "service_name": svc_name,
                    "model_version": model_version,
                    "model_path": model_path,
                },
            )

            return ok({
                "action_id": action.id,
                "target_type": "inferenceservice",
                "target_id": service_id,
                "target_name": svc_name,
                "market_service_id": market_service.id,
                "url": service_url,
                "status_url": f"/model_market/api/services/{market_service.id}/status",
                "api_url": f"/model_market/api/services/{market_service.id}/predict",
                "api_example_url": f"/model_market/api/services/{market_service.id}/api-example",
                "stats_url": f"/model_market/api/services/{market_service.id}/stats",
                "model_version": model_version,
                "model_path": model_path,
                "reused": False,
            }, "推理服务创建成功")

        except Exception as e:
            traceback.print_exc()
            mark_action_failed(action, e)
            return fail(f"推理服务创建失败：{str(e)}", code=500, http_status=500)

    # ---- 卸载服务 ----
    @expose("/services/<int:market_service_id>/unload", methods=["POST"])
    def service_unload(self, market_service_id):
        """卸载推理服务：调用平台 InferenceService API 删除 K8s 资源"""
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        delete_ok = False
        delete_msg = ""
        try:
            # Call platform InferenceService delete API
            result = _call_api("DELETE", f"/inferenceservice_modelview/api/{market_service.service_id}").json()
            delete_ok = True
            delete_msg = "deleted via API"
        except Exception as e:
            # If the platform service is already gone (404 or connection error),
            # we should still mark the market service as unloaded
            err_str = str(e).lower()
            if "404" in err_str or "not found" in err_str or "doesn't exist" in err_str \
                    or "connection" in err_str or "refused" in err_str:
                delete_ok = True
                delete_msg = f"platform service already removed ({e})"
            else:
                traceback.print_exc()
                return fail(f"卸载失败：{str(e)}", code=500, http_status=500)

        # Clear service state regardless of how it was removed
        market_service.service_status = "unloaded"
        market_service.endpoint = ""
        extra = json.loads(market_service.extra_json or "{}")
        extra["unload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        extra["unload_method"] = delete_msg
        extra["endpoint"] = ""
        market_service.extra_json = json_dumps(extra)
        db.session.commit()

        mark_action_success(
            create_action(market_service.model_id, "unload",
                          {"market_service_id": market_service_id, "service_id": market_service.service_id}),
            target_type="inferenceservice",
            target_id=market_service.service_id,
            target_name=market_service.service_name,
            target_url=f"/frontend/service/model_market_group/detail/{market_service.model_id}?tab=deploy",
            response={"unloaded": True, "method": delete_msg},
        )

        # Also clean up any other stale services for the same model
        cleaned_stale = []
        try:
            from myapp.models.model_serving import InferenceService as InfSvc
            other_services = db.session.query(ModelMarketService).filter_by(
                model_id=market_service.model_id
            ).filter(
                ~ModelMarketService.service_status.in_(['unloaded', 'deleted', 'failed', 'stopped'])
            ).filter(
                ModelMarketService.id != market_service_id
            ).all()
            for osvc in other_services:
                plat = db.session.query(InfSvc).get(osvc.service_id)
                if not plat:
                    osvc.service_status = "unloaded"
                    osvc.endpoint = ""
                    extra_o = json.loads(osvc.extra_json or "{}")
                    extra_o["endpoint"] = ""
                    extra_o["unload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    extra_o["unload_reason"] = "stale_cleanup_during_unload"
                    osvc.extra_json = json_dumps(extra_o)
                    cleaned_stale.append(osvc.id)
            if cleaned_stale:
                db.session.commit()
                print(f"[model_market unload] cleaned stale services: {cleaned_stale}")
        except Exception:
            pass

        return ok({
            "unloaded": True,
            "market_service_id": market_service_id,
            "model_id": market_service.model_id,
            "service_id": market_service.service_id,
            "service_status": "unloaded",
            "active_remaining": 0,
            "cleaned_stale_service_ids": cleaned_stale,
            "message": delete_msg,
        }, "服务已卸载")

    # ---- 操作记录 ----
    @expose("/actions/<int:action_id>", methods=["GET"])
    def action_detail(self, action_id):
        action = db.session.query(ModelMarketAction).get(action_id)
        if not action:
            return fail("操作记录不存在", code=404, http_status=404)

        return ok({
            "id": action.id,
            "model_id": action.model_id,
            "action_type": action.action_type,
            "status": action.status,
            "target_type": action.target_type,
            "target_id": action.target_id,
            "target_name": action.target_name,
            "target_url": action.target_url,
            "project_id": action.project_id,
            "namespace": action.namespace,
            "error_msg": action.error_msg,
        })

    # ---- 服务状态 ----
    @expose("/services/<int:market_service_id>/status", methods=["GET"])
    def service_status(self, market_service_id):
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        from myapp.models.model_serving import InferenceService
        from myapp.utils.py.py_k8s import K8s

        svc = db.session.query(InferenceService).get(market_service.service_id)
        pod_status = "Unknown"
        pod_ready = False
        service_ready = False
        endpoint_ready = False
        resolved_endpoint = ""
        message = ""

        try:
            cluster = svc.project.cluster if svc and svc.project else current_app.config.get('CLUSTERS',{}).get(current_app.config.get('ENVIRONMENT','dev'),{})
            k8s_client = K8s(cluster.get('KUBECONFIG',''))
            namespace = svc.namespace or "service"
            name = svc.name

            # 1. Check Pod via label selector (pod name includes Deployment suffix)
            pods = k8s_client.get_pods(namespace=namespace, labels={"app": name})
            if pods:
                pod = pods[0]
                pod_status = pod.get('status', 'Unknown')
                pod_ready = (pod_status == 'Running')

            # 2. Resolve endpoint via K8s Service ClusterIP (reachable from myapp container)
            if pod_ready:
                try:
                    k8s_svc = k8s_client.v1.read_namespaced_service(name=name, namespace=namespace)
                    if k8s_svc and k8s_svc.spec and k8s_svc.spec.cluster_ip:
                        svc_ip = k8s_svc.spec.cluster_ip
                        port = k8s_svc.spec.ports[0].port if k8s_svc.spec.ports else 8080
                        resolved_endpoint = f"http://{svc_ip}:{port}"
                        endpoint_ready = True
                        service_ready = True
                except Exception:
                    pass

            # 3. Verify service is actually reachable before declaring Ready
            if endpoint_ready and resolved_endpoint:
                # Health check: confirm the service responds to HTTP
                health_ok = False
                health_timed_out = False
                health_msg = ""
                try:
                    import requests as req_lib
                    health_url = resolved_endpoint.rstrip("/") + "/openapi.json"
                    hr = req_lib.get(health_url, timeout=5)
                    if hr.status_code == 200:
                        health_ok = True
                        health_msg = "health check passed"
                    else:
                        health_msg = f"health check returned {hr.status_code}"
                except req_lib.exceptions.ConnectionError as ce:
                    health_msg = f"connection refused: {ce}"
                except req_lib.exceptions.Timeout:
                    health_timed_out = True
                    health_msg = "health check timeout"
                except Exception as he:
                    health_msg = f"health check error: {he}"

                if health_ok:
                    status = "Ready"
                    ready = True
                    message = f"服务已就绪 ({health_msg})"
                    # Persist endpoint
                    market_service.endpoint = resolved_endpoint
                    market_service.service_status = "Ready"
                    extra = json.loads(market_service.extra_json or "{}")
                    extra["endpoint"] = resolved_endpoint
                    extra["pod_status"] = pod_status
                    extra["endpoint_ready"] = True
                    extra["health_check"] = health_msg
                    market_service.extra_json = json_dumps(extra)
                    db.session.commit()
                elif health_timed_out and market_service.service_status == "Ready":
                    # A CPU-bound inference can temporarily occupy a single-worker
                    # model server. Keep an already healthy, running Pod Ready while
                    # that request is in flight instead of disabling the UI.
                    status = "Ready"
                    ready = True
                    service_ready = True
                    message = "服务正在处理推理请求"
                else:
                    status = "Deploying"
                    ready = False
                    message = f"Pod 已运行，但服务未就绪：{health_msg}"
            elif pod_ready:
                status = "Deploying"
                ready = False
                message = "Pod 已运行，等待网络就绪（endpoints 为空）"
            elif pod_status == "Pending":
                status = pod_status
                ready = False
                message = "Pod 调度中，可能因资源或 PVC 未绑定"
            else:
                status = pod_status
                ready = False
                message = f"Pod 状态: {pod_status}"

        except Exception as e:
            status = market_service.service_status or "Unknown"
            ready = False
            message = f"状态查询异常: {str(e)}"

        # Build predict_url for frontend
        extra_json = json.loads(market_service.extra_json or "{}")
        market_model = db.session.query(ModelMarketModel).get(market_service.model_id)
        model_version = extra_json.get("model_version", "v1")
        model_name = extra_json.get("model_name") or (market_model.name if market_model else "")
        real_endpoint = resolved_endpoint or market_service.endpoint or ""
        predict_url = ""
        if real_endpoint and model_name and model_version:
            predict_url = real_endpoint.rstrip("/") + "/v1/models/{}/versions/{}/predict".format(model_name, model_version)

        return ok({
            "market_service_id": market_service.id,
            "service_id": market_service.service_id,
            "service_name": market_service.service_name,
            "model_id": market_service.model_id,
            "model_name": model_name,
            "display_name": market_model.display_name if market_model else model_name,
            "task_type": market_model.task_type if market_model else extra_json.get("task_type", ""),
            "demo_input_type": market_model.demo_input_type if market_model else extra_json.get("demo_input_type", ""),
            "demo_output_type": market_model.demo_output_type if market_model else extra_json.get("demo_output_type", ""),
            "model_version": model_version,
            "status": status,
            "ready": ready,
            "pod_status": pod_status,
            "pod_ready": pod_ready,
            "service_ready": service_ready,
            "endpoint_ready": endpoint_ready,
            "endpoint": real_endpoint,
            "predict_url": predict_url,
            "message": message,
        })

    # ---- 推理代理 ----
    @expose("/services/<int:market_service_id>/predict", methods=["POST"])
    def service_predict(self, market_service_id):
        """代理推理请求到真实 InferenceService"""
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        import time as _time
        start = _time.time()

        # Resolve endpoint via status check (cached)
        endpoint = market_service.endpoint
        if not endpoint:
            extra = json.loads(market_service.extra_json or "{}")
            endpoint = extra.get("endpoint", "")

        if not endpoint:
            # Trigger full status check to try resolving
            try:
                self.service_status(market_service_id)
                db.session.refresh(market_service)
                endpoint = market_service.endpoint
                if not endpoint:
                    extra = json.loads(market_service.extra_json or "{}")
                    endpoint = extra.get("endpoint", "")
            except Exception:
                pass

        if not endpoint:
            return fail(
                "服务 endpoint 为空，Pod 可能尚未就绪。请等待 Pod Running 后再试",
                code=503, http_status=503,
            )

        market_model = db.session.query(ModelMarketModel).get(market_service.model_id)
        task_type = market_model.task_type if market_model else ""
        input_type = market_model.demo_input_type if market_model else "image"

        try:

            # Forward the request
            files = {}
            json_data = None
            if request.files:
                for key in request.files:
                    files[key] = (request.files[key].filename, request.files[key].read(),
                                  request.files[key].content_type or "application/octet-stream")
            elif request.is_json:
                json_data = request.get_json(silent=True)

            # Build predict URL dynamically from stored model version
            import base64 as _b64
            extra = json.loads(market_service.extra_json or "{}")
            model_version = extra.get("model_version")
            model_name = extra.get("model_name") or (market_model.name if market_model else "yolov8")

            # The deployment record version can differ from the route exposed by
            # older images. Prefer the actual OpenAPI route so existing services
            # continue working without a destructive redeploy.
            try:
                openapi_url = endpoint.rstrip("/") + "/openapi.json"
                oa_resp = requests.get(openapi_url, timeout=10)
                if oa_resp.status_code == 200:
                    oa = oa_resp.json()
                    for path_key in (oa.get("paths") or {}):
                        if "/predict" in path_key:
                            parts = path_key.strip("/").split("/")
                            try:
                                vidx = parts.index("versions")
                                model_version = parts[vidx + 1]
                                model_name = parts[parts.index("models") + 1] if "models" in parts else model_name
                                break
                            except (ValueError, IndexError):
                                pass
            except Exception:
                pass
            if not model_version:
                model_version = "v1"

            predict_path = "/v1/models/{}/versions/{}/predict".format(model_name, model_version)
            predict_url = endpoint.rstrip("/") + predict_path

            json_body = None
            if files:
                payload_name = "audio" if input_type == "audio" else "image"
                if market_model:
                    schema_inputs = (market_model.api_schema() or {}).get("input") or []
                    file_inputs = [
                        item for item in schema_inputs
                        if item.get("type") in ("audio", "image", "file")
                    ]
                    if file_inputs:
                        payload_name = file_inputs[0].get("name") or payload_name
                for key in files:
                    _, content, _ = files[key]
                    if input_type == "audio" and len(content) > 100 * 1024 * 1024:
                        return fail("音频文件不能超过 100 MB", code=413, http_status=413)
                    json_body = {payload_name: _b64.b64encode(content).decode()}
                    break
                if json_body is not None:
                    for key in request.form:
                        json_body[key] = request.form.get(key)
            elif json_data:
                json_body = json_data

            upstream_timeout = 300 if input_type == "audio" else 60
            if json_body:
                resp = requests.post(predict_url, json=json_body, timeout=(5, upstream_timeout))
            else:
                resp = requests.post(predict_url, timeout=(5, upstream_timeout))

            latency_ms = int((_time.time() - start) * 1000)

            # Parse response - service returns {"names":[...], "labels":[...], "scores":[...], "xywhns":[...], "orig_shape":[...]}
            try:
                result_obj = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            except Exception:
                result_obj = resp.text

            if task_type == "speech_recognition" and isinstance(result_obj, dict):
                message = (
                    "识别成功" if result_obj.get("text")
                    else "识别成功，未识别到文本"
                )
            elif isinstance(result_obj, dict):
                # Detection services return normalized xywh in "xywhns".
                # Preserve both xywhns (original) and map to xywhs for backward compatibility
                if "xywhns" in result_obj and "xywhs" not in result_obj:
                    result_obj["xywhs"] = result_obj["xywhns"]
                has_detections = bool(result_obj.get("labels"))
                if not has_detections:
                    message = "推理成功，未检测到目标"
                else:
                    message = "推理成功"
            else:
                message = "推理成功"

            # Update stats
            try:
                extra = json.loads(market_service.extra_json or "{}")
                extra["call_count"] = extra.get("call_count", 0) + 1
                extra["last_called_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                if resp.status_code < 400:
                    extra["success_count"] = extra.get("success_count", 0) + 1
                else:
                    extra["failed_count"] = extra.get("failed_count", 0) + 1
                # Running average latency
                old_avg = extra.get("avg_latency_ms", 0)
                old_count = extra.get("call_count", 1) - 1
                extra["avg_latency_ms"] = int((old_avg * old_count + latency_ms) / extra["call_count"])
                # Persist latest model_version / predict_url for next time
                if model_version:
                    extra["model_version"] = model_version
                market_service.extra_json = json_dumps(extra)
                db.session.commit()
            except Exception:
                pass

            return ok({
                "result": result_obj,
                "latency_ms": latency_ms,
                "endpoint": endpoint,
                "predict_url": predict_url,
                "message": message,
            })

        except requests.exceptions.ConnectionError as ce:
            latency_ms = int((_time.time() - start) * 1000)
            # Record failure
            try:
                extra = json.loads(market_service.extra_json or "{}")
                extra["call_count"] = extra.get("call_count", 0) + 1
                extra["failed_count"] = extra.get("failed_count", 0) + 1
                extra["last_called_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                extra["last_error"] = f"connection_refused: {ce}"
                market_service.extra_json = json_dumps(extra)
                db.session.commit()
            except Exception:
                pass
            return fail(
                "推理服务暂不可用，请检查服务是否 Ready 或模型路径是否正确",
                code=503, http_status=503,
                data={"endpoint": endpoint or "", "reason": "connection_refused"}
            )
        except Exception as e:
            latency_ms = int((_time.time() - start) * 1000)
            return fail("推理请求失败: {}".format(str(e)), code=500, http_status=500)

    # ---- API 示例 ----
    @expose("/services/<int:market_service_id>/api-example", methods=["GET"])
    def service_api_example(self, market_service_id):
        """返回服务的 API 调用示例"""
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        market_model = db.session.query(ModelMarketModel).get(market_service.model_id)
        is_audio = bool(market_model and market_model.demo_input_type == "audio")
        file_name = "your_audio.mp3" if is_audio else "your_image.jpg"
        origin = request.host_url.rstrip("/")
        predict_url = f"{origin}/model_market/api/services/{market_service_id}/predict"

        curl_cmd = (
            f'curl -X POST "{predict_url}" '
            f'-F "file=@{file_name}"'
        )
        if is_audio:
            curl_cmd += ' -F "language=auto" -F "timestamps=true"'
        python_code = (
            f"import requests\n\n"
            f"url = '{predict_url}'\n"
            f"with open('{file_name}', 'rb') as f:\n"
            f"    resp = requests.post(url, files={{'file': f}}"
            + (", data={'language': 'auto', 'timestamps': 'true'}" if is_audio else "")
            + ")\n"
            f"print(resp.json())"
        )
        sample_result = (
            {"text": "识别文本", "language": "auto", "duration_seconds": 3.2, "segments": []}
            if is_audio else
            {"labels": [], "scores": [], "xywhns": [], "xywhs": [], "orig_shape": [], "names": []}
        )

        return ok({
            "endpoint": predict_url,
            "method": "POST",
            "content_type": "multipart/form-data",
            "curl": curl_cmd,
            "python": python_code,
            "sample_response": {
                "code": 0,
                "data": {
                    "result": sample_result,
                    "latency_ms": 123,
                    "endpoint": "...",
                    "predict_url": "...",
                    "message": "识别成功" if is_audio else "推理成功",
                }
            }
        })

    # ---- 调用统计 ----
    @expose("/services/<int:market_service_id>/stats", methods=["GET"])
    def service_stats(self, market_service_id):
        """返回服务的调用统计"""
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        extra = json.loads(market_service.extra_json or "{}")
        return ok({
            "call_count": extra.get("call_count", 0),
            "success_count": extra.get("success_count", 0),
            "failed_count": extra.get("failed_count", 0),
            "avg_latency_ms": extra.get("avg_latency_ms", 0),
            "last_called_at": extra.get("last_called_at"),
        })

    # ---- Notebook 状态查询 ----
    @expose("/notebooks/<int:notebook_id>/status", methods=["GET"])
    def notebook_status(self, notebook_id):
        """查询一键开发创建的 Notebook 的 K8s 运行状态"""
        from myapp.models.model_notebook import Notebook
        from myapp.utils.py.py_k8s import K8s

        notebook = db.session.query(Notebook).get(notebook_id)
        if not notebook:
            return fail("Notebook 不存在", code=404, http_status=404)

        name = notebook.name
        namespace = notebook.namespace or "jupyter"
        username = getattr(current_user, 'username', '') or 'admin'
        
        target_url = f"/frontend/dev/dev_online/notebook_shell?id={notebook_id}"
        jupyter_url = get_notebook_jupyter_url(notebook, target_url)

        pod_status = "Unknown"
        pod_ready = False
        service_ready = False
        endpoint_ready = False
        ready = False
        message = ""

        try:
            cluster = notebook.project.cluster if notebook.project else current_app.config.get('CLUSTERS',{}).get(current_app.config.get('ENVIRONMENT','dev'),{})
            k8s_client = K8s(cluster.get('KUBECONFIG', ''))

            # 1. Check Pod
            pods = k8s_client.get_pods(namespace=namespace, pod_name=name)
            if pods:
                pod = pods[0]
                pod_status = pod.get('status', 'Unknown')
                pod_ready = (pod_status == 'Running')
                if not pod_ready:
                    conditions = pod.get('status_more', {}).get('conditions', [])
                    for cond in conditions:
                        if cond.get('status') == 'False':
                            message = cond.get('message', cond.get('reason', ''))
                            break
                    if not message:
                        message = f"Pod 状态: {pod_status}"

            # 2. Check Service endpoints
            if pod_ready:
                try:
                    svc = k8s_client.v1.read_namespaced_service(name=name, namespace=namespace)
                    if svc:
                        service_ready = True
                except Exception:
                    pass

                try:
                    eps = k8s_client.v1.read_namespaced_endpoints(name=name, namespace=namespace)
                    if eps and eps.subsets:
                        for subset in eps.subsets:
                            if subset.addresses:
                                endpoint_ready = True
                                break
                except Exception:
                    pass

            # 3. Determine overall ready — pod + service is enough for JupyterLab
            ready = pod_ready and service_ready
            if ready:
                message = "Notebook 已就绪"
            elif pod_ready and not (service_ready and endpoint_ready):
                message = "Pod 已运行，等待网络就绪"

        except Exception as e:
            message = f"状态查询异常: {str(e)}"

        return ok({
            "notebook_id": notebook_id,
            "name": name,
            "namespace": namespace,
            "status": pod_status,
            "pod_status": pod_status,
            "pod_ready": pod_ready,
            "service_ready": service_ready,
            "endpoint_ready": endpoint_ready,
            "ready": ready,
            "jupyter_url": jupyter_url,
            "message": message or f"当前状态: {pod_status}",
        })

    # ---- 服务指标 ----
    @expose("/services/<int:market_service_id>/metrics", methods=["GET"])
    def metrics(self, market_service_id):
        market_service = db.session.query(ModelMarketService).get(market_service_id)
        if not market_service:
            return fail("服务不存在", code=404, http_status=404)

        return ok({
            "total": 0,
            "success": 0,
            "failed": 0,
            "avg_latency": 0,
        })


# ---- 注册视图（无菜单，通过前端路由访问） ----
from myapp import appbuilder

appbuilder.add_view_no_menu(ModelMarketApiView())
