"""
训练监控接口模块
- POST /training_monitor/api/register : 训练容器回调注册监控 URL
- GET  /training_monitor/api/by_run   : 按 run_id 查询监控记录
- GET  /training_monitor/api/check    : 统一入口，处理 0/1/N 条记录的跳转逻辑
"""
import json
import traceback
import urllib.request

from flask import request, jsonify, redirect
from flask_babel import gettext as __
from sqlalchemy.exc import ProgrammingError, OperationalError

from myapp import app, db, conf
from myapp.models.model_training_monitor import TrainingMonitor

# SwanLab Dashboard 地址
SWANLAB_HOST = conf.get('SWANLAB_WEB_HOST', 'http://10.121.177.20:30092')

# 表未创建时的友好提示
ERR_TABLE_MISSING_MSG = (
    "训练监控表尚未初始化，请先执行数据库迁移："
    "flask db migrate -m 'add mlops training monitor table' && flask db upgrade"
)


def _dashboard_api_get(path, timeout=5):
    """调用 SwanLab Dashboard API"""
    try:
        url = f"{SWANLAB_HOST}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _resolve_experiment_id_from_api(run_dir_name, experiment_name=""):
    """通过 Dashboard API 解析数字 experiment_id"""
    data = _dashboard_api_get("/api/v1/project")
    if not data:
        return None
    experiments = []
    if isinstance(data, dict):
        experiments = data.get("experiments", data.get("data", {}).get("experiments", []))
    for exp in experiments:
        if isinstance(exp, dict):
            if exp.get("run_id") == run_dir_name:
                return exp.get("id")
    for exp in experiments:
        if isinstance(exp, dict):
            if exp.get("name") == experiment_name:
                return exp.get("id")
    return None


def normalize_monitor_url(record):
    """
    统一规范化 monitor_url：
    1. 已是 /experiment/<数字id>/chart → 直接返回
    2. 是 /experiment/run-xxxx/chart → 通过 Dashboard API 解析数字 id 后更新 DB
    3. 是首页 → 尝试通过 experiment_run_id 或 experiment_name 解析
    4. 解析不到 → 返回首页（监控初始化中）

    返回 (url, message) 元组。
    """
    import re
    url = (record.monitor_url or "").strip()
    if not url:
        return (f"{SWANLAB_HOST}/", "监控初始化中，请稍后刷新")

    # 1. 已经是标准数字 id URL → 直接返回
    if re.search(r'/experiment/\d+/chart', url):
        return (url, "")

    # 2. 是旧格式 /experiment/run-xxxx/chart → 提取 run_dir_name 解析
    run_match = re.search(r'/experiment/(run-[^/]+)/chart', url)
    run_dir_name = run_match.group(1) if run_match else ""
    if not run_dir_name:
        run_dir_name = (getattr(record, 'experiment_run_id', '') or '').strip()

    # 3. 尝试解析
    experiment_name = getattr(record, 'experiment_name', '') or ''
    eid = None
    if run_dir_name:
        eid = _resolve_experiment_id_from_api(run_dir_name, experiment_name)

    if eid is not None:
        new_url = f"{SWANLAB_HOST}/experiment/{eid}/chart"
        try:
            record.monitor_url = new_url
            db.session.commit()
        except Exception:
            db.session.rollback()
        return (new_url, "")

    # 4. 解析不到
    if run_match:
        # 旧格式 URL 但 API 查不到 → 回首页
        return (f"{SWANLAB_HOST}/", "监控初始化中，请稍后刷新（实验尚未索引）")
    # 首页 → 保持首页
    return (f"{SWANLAB_HOST}/", "监控初始化中，请稍后刷新")


# =========================================================================
#  POST /training_monitor/api/register
# =========================================================================

@app.route('/training_monitor/api/register', methods=['POST'])
def training_monitor_register():
    """
    训练容器启动 SwanLab 实验后，把 monitor_url 回传给 MLOps 后端保存。

    幂等：同一 run_id + node_name + monitor_type 已存在则更新，否则新增。
    """
    try:
        json_data = request.get_json(silent=True)
        if not json_data:
            return jsonify({"status": "error", "message": "请求体不能为空，需要 JSON"}), 400

        # ---- Token 校验（可选） ----
        register_token = conf.get('MLOPS_MONITOR_REGISTER_TOKEN', '')
        if register_token:
            req_token = json_data.get('token', '') or request.headers.get('X-Monitor-Token', '')
            if req_token != register_token:
                return jsonify({"status": "error", "message": "token 校验失败"}), 403

        # ---- 统一 run_id（兼容 mlops_run_id / run_id 两种字段名） ----
        run_id = (json_data.get('mlops_run_id') or json_data.get('run_id') or '').strip()
        if not run_id:
            return jsonify({"status": "error", "message": "run_id 不能为空"}), 400

        node_name = (json_data.get('node_name') or json_data.get('task_name') or '').strip()
        monitor_type = (json_data.get('monitor_type') or 'swanlab').strip()

        # ---- monitor_url 规范化：自动追加 /chart ----
        monitor_url = (json_data.get('monitor_url') or '').strip()
        if not monitor_url:
            return jsonify({"status": "error", "message": "monitor_url 不能为空"}), 400
        # 如果 URL 以 /runs/<id> 结尾但不是 /runs/<id>/chart，自动追加 /chart
        import re
        if re.search(r'/runs/[^/]+$', monitor_url):
            monitor_url += '/chart'

        # ---- 幂等：查重 -> 更新 或 新增 ----
        existing = None
        if node_name:
            existing = (
                db.session.query(TrainingMonitor)
                .filter_by(run_id=run_id, node_name=node_name, monitor_type=monitor_type)
                .first()
            )
        if not existing:
            existing = (
                db.session.query(TrainingMonitor)
                .filter_by(run_id=run_id)
                .order_by(TrainingMonitor.id.desc())
                .first()
            )

        if existing:
            existing.monitor_url = monitor_url
            existing.monitor_status = json_data.get('monitor_status', 'RUNNING')
            existing.task_name = json_data.get('task_name', existing.task_name)
            existing.pod_name = json_data.get('pod_name', existing.pod_name)
            _erid = (json_data.get('experiment_run_id') or json_data.get('swanlab_run_id') or '').strip()
            if _erid:
                existing.experiment_run_id = _erid
            existing.changed_on = db.func.now()
            db.session.commit()
            return jsonify({
                "status": "ok",
                "id": existing.id,
                "monitor_url": existing.monitor_url,
                "message": "已更新已有监控记录",
                "action": "updated",
            })

        # 使用 task_name 作为 node_name（如果 node_name 为空）
        if not node_name:
            node_name = (json_data.get('task_name') or 'unknown').strip()

        record = TrainingMonitor(
            pipeline_id=(json_data.get('pipeline_id') or ''),
            pipeline_name=(json_data.get('pipeline_name') or ''),
            run_id=run_id,
            workflow_name=(json_data.get('workflow_name') or ''),
            task_id=str(json_data.get('task_id', '')),
            task_name=(json_data.get('task_name') or ''),
            node_name=node_name,
            pod_name=(json_data.get('pod_name') or ''),
            job_template_name=(json_data.get('job_template_name') or ''),
            monitor_type=monitor_type,
            monitor_url=monitor_url,
            monitor_status=json_data.get('monitor_status', 'RUNNING'),
            creator=(json_data.get('creator') or ''),
            experiment_run_id=(json_data.get('experiment_run_id') or json_data.get('swanlab_run_id') or '').strip(),
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            "status": "ok",
            "id": record.id,
            "monitor_url": record.monitor_url,
            "message": "注册成功",
            "action": "created",
        })

    except Exception as e:
        import traceback, logging
        logger = logging.getLogger(__name__)
        logger.exception(f"register error: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": f"服务器内部错误: {str(e)}"}), 500


# =========================================================================
#  GET /training_monitor/api/by_run?run_id=xxx
# =========================================================================

@app.route('/training_monitor/api/by_run', methods=['GET'])
def training_monitor_by_run():
    """
    按运行实例 run_id 查询所有监控记录。
    返回 JSON，供前端弹窗或接口调试使用。
    """
    try:
        run_id = (request.args.get('run_id') or '').strip()
        if not run_id:
            return jsonify({"success": False, "count": 0, "items": [], "message": "run_id 不能为空"}), 400

        records = (
            db.session.query(TrainingMonitor)
            .filter_by(run_id=run_id)
            .order_by(TrainingMonitor.id.desc())
            .all()
        )

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "run_id": r.run_id,
                "workflow_name": r.workflow_name,
                "pipeline_id": r.pipeline_id,
                "pipeline_name": r.pipeline_name,
                "task_id": r.task_id,
                "task_name": r.task_name,
                "node_name": r.node_name,
                "pod_name": r.pod_name,
                "job_template_name": r.job_template_name,
                "monitor_type": r.monitor_type,
                "monitor_url": r.monitor_url,
                "monitor_status": r.monitor_status,
                "creator": r.creator,
            })

        return jsonify({
            "success": True,
            "count": len(items),
            "items": items,
        })

    except (ProgrammingError, OperationalError) as e:
        traceback.print_exc()
        return jsonify({"success": False, "count": 0, "items": [], "message": ERR_TABLE_MISSING_MSG}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "count": 0, "items": [], "message": str(e)}), 500


# =========================================================================
#  GET /training_monitor/api/check?run_id=xxx
# =========================================================================

CHECK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f5f7fa; color: #333; }}
  h2 {{ border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
  .empty {{ text-align: center; padding: 60px 20px; color: #999; }}
  .empty .icon {{ font-size: 64px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #4a90d9; color: #fff; font-weight: 600; }}
  tr:hover {{ background: #f0f6ff; }}
  a {{ color: #4a90d9; text-decoration: none; font-weight: 500; }}
  a:hover {{ text-decoration: underline; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
  .badge-running {{ background: #e6f7e6; color: #1b8; }}
  .badge-succeeded {{ background: #e6f7e6; color: #1b813e; }}
  .badge-failed {{ background: #ffe6e6; color: #cb1b45; }}
  .badge-init {{ background: #f0f0f0; color: #666; }}
  .meta {{ color: #999; font-size: 14px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<p class="meta">运行实例 ID：<code>{run_id}</code>，共 {count} 条监控记录</p>
{content}
</body>
</html>"""


@app.route('/training_monitor/api/check', methods=['GET'])
def training_monitor_check():
    """
    统一入口：运行实例列表点击"监控"时跳转到这里。
    - 0 条记录：提示"暂无监控"
    - 1 条记录：直接 redirect 到 monitor_url
    - N 条记录：展示节点列表供用户选择
    """
    try:
        run_id = (request.args.get('run_id') or '').strip()
        if not run_id:
            return CHECK_HTML_TEMPLATE.format(
                title="参数错误",
                run_id="(空)",
                count=0,
                content='<div class="empty"><div class="icon">⚠️</div><p>缺少 run_id 参数</p></div>',
            ), 400

        records = (
            db.session.query(TrainingMonitor)
            .filter_by(run_id=run_id)
            .order_by(TrainingMonitor.id.desc())
            .all()
        )

        count = len(records)

        # ---- 0 条：暂无监控 ----
        if count == 0:
            # 显示最近 5 条记录摘要帮助调试
            recent = (
                db.session.query(TrainingMonitor)
                .order_by(TrainingMonitor.id.desc())
                .limit(5)
                .all()
            )
            recent_rows = ""
            if recent:
                rows = []
                for r in recent:
                    rows.append(
                        f'<tr><td>{r.id}</td><td><code>{r.run_id[:20]}...</code></td>'
                        f'<td>{r.task_name or "-"}</td><td>{r.monitor_status}</td></tr>'
                    )
                recent_rows = (
                    '<p style="font-size:13px;color:#aaa;margin-top:20px;">数据库最近 5 条记录：</p>'
                    '<table style="font-size:12px;"><thead><tr>'
                    '<th>ID</th><th>Run ID</th><th>Task</th><th>Status</th>'
                    f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
                )

            html = CHECK_HTML_TEMPLATE.format(
                title="训练监控",
                run_id=run_id,
                count=0,
                content=(
                    '<div class="empty"><div class="icon">📊</div>'
                    '<p>当前运行实例没有生成训练监控记录</p>'
                    '<p style="font-size:13px;color:#aaa;">'
                    '请检查：<br>'
                    '1. 训练 Pod 是否成功调用了 /training_monitor/api/register<br>'
                    '2. 查看 myapp 后端日志中 register 相关的错误<br>'
                    '3. SWANLAB_MODE / SWANLAB_API_KEY 是否正确注入'
                    '</p>'
                    f'{recent_rows}'
                    '</div>'
                ),
            )
            return html

        # ---- 有记录：选最佳的一条直接跳转 ----
        # 优先：SUCCEEDED/RUNNING 状态 + hyperparam-search + 有 monitor_url
        best = None
        for r in records:
            url = (r.monitor_url or '').strip()
            if not url:
                continue
            # 优先 hyperparam-search + SUCCEEDED
            if r.monitor_status == 'SUCCEEDED':
                best = r
                break
            if r.monitor_status == 'RUNNING' and best is None:
                best = r
        if best is None:
            for r in records:
                if (r.monitor_url or '').strip():
                    best = r
                    break
        if best is None:
            best = records[0]

        url = (best.monitor_url or '').strip()
        if url:
            return redirect(url)

        # ---- 有记录但 monitor_url 为空：显示节点列表 ----
        rows_html = ""
        for r in records:
            status_badge = f'<span class="badge badge-{r.monitor_status.lower()}">{r.monitor_status}</span>'
            u = (r.monitor_url or '').strip()
            monitor_link = f'<a href="{u}" target="_blank">监控</a>' if u else "暂无"
            rows_html += (
                f'<tr>'
                f'<td>{r.node_name or "-"}</td>'
                f'<td>{r.job_template_name or "-"}</td>'
                f'<td>{status_badge}</td>'
                f'<td>{monitor_link}</td>'
                f'</tr>'
            )

        html = CHECK_HTML_TEMPLATE.format(
            title="训练监控",
            run_id=run_id,
            count=count,
            content=f"""
            <table>
              <thead><tr><th>节点名称</th><th>算子类型</th><th>监控状态</th><th>操作</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            """,
        )
        return html

    except (ProgrammingError, OperationalError) as e:
        traceback.print_exc()
        return CHECK_HTML_TEMPLATE.format(
            title="训练监控",
            run_id=run_id,
            count=0,
            content=f'<div class="empty"><div class="icon">🛠️</div><p>训练监控表尚未初始化</p><p style="font-size:13px;color:#aaa;">请在 MLOps 后端容器内执行:<br><code>flask db migrate -m "add mlops_training_monitor" && flask db upgrade</code><br>或联系管理员初始化监控功能</p></div>',
        ), 500
    except Exception as e:
        traceback.print_exc()
        return CHECK_HTML_TEMPLATE.format(
            title="服务器错误",
            run_id=run_id,
            count=0,
            content=f'<div class="empty"><p>查询监控记录出错：{str(e)}</p></div>',
        ), 500
