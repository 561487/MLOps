"""
Assistant API v1.0 — 浮窗 AI 助手后端
=========================================

路由前缀：/api/v2/assistant

API 列表：
  POST   /chat                SSE 流式对话
                                Body: { "message": "...", "session_id": "..." }
                                Response: text/event-stream
                                          data: {"text": "..."}\n\n
                                          data: {"done": true}\n\n
  GET    /sessions            列出当前用户的会话（懒清理 30 天前的）
  POST   /session/new        创建新会话，返回 session_id
  GET    /history/<sid>      取该会话的所有消息
  DELETE /history/<sid>      删除该会话及其所有消息

设计要点：
  - 复用 myapp 现有 LLM 链路（OpenAI 兼容协议 + sseclient 解析）
  - 配置优先级：环境变量 ASSISTANT_LLM_URL / ASSISTANT_LLM_API_KEY
                → 退化到 conf.CHATGPT_CHAT_URL / conf.CHATGPT_TOKEN
  - 对话历史持久化到 MySQL（assistant_conversation + assistant_message 两张表）
  - 按 username + session_id 维度存，跟现有 ChatLog 一致
  - 不再维护内存字典 _HISTORY，DB 为唯一事实来源
  - 保留策略：30 天前的会话由 GET /sessions 懒清理（不引入 cron）
"""
import os
import json
import random
import logging
import time
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps

from flask import Blueprint, request, jsonify, g, Response
from flask import stream_with_context

from myapp import appbuilder, conf, db
from myapp.models.model_assistant import AssistantConversation, AssistantMessage


# ===== 日志器（审计 + 调试） =====
_logger = logging.getLogger('assistant')

# ===== 简单 rate limit：每用户每分钟最多 N 次请求 =====
RATE_LIMIT_PER_MIN = 20  # 每分钟 20 次足够人工输入了
_RATE_BUCKET = defaultdict(deque)  # username -> deque of timestamps

# ===== 对话历史保留期 =====
HISTORY_RETAIN_DAYS = 30  # 超过 30 天的会话由 GET /sessions 懒清理

# ===== 拼到 system prompt 的历史轮数 =====
MAX_HISTORY_ROUNDS = 10  # 每次调 LLM 时从 DB 取最近 10 轮拼到 prompt

# ===== Blueprint =====
assistant_bp = Blueprint(
    'assistant_api',
    __name__,
    url_prefix='/api/v2/assistant',
)


# ===== 系统提示词（写死，第一版本） =====
SYSTEM_PROMPT = """你是 MLOps 平台的浮窗助手，专门帮助用户解答平台操作问题。

你可以回答以下方面的问题：
- 模型微调：LoRA 参数配置、数据集准备、训练任务启动
- 数据集管理：数据格式、dataset_info.json 配置、ShareGPT/标准格式
- 流水线：创建流水线、任务串联、参数配置
- 任务模板：模板参数、镜像选择、环境变量
- 镜像构建：build.sh 使用、Tag 命名规范、Dockerfile 编写
- 模型合并：LoRA 合并流程、合并后模型位置

如果用户问的问题与平台无关，礼貌引导回平台操作话题。
回答要简洁、步骤清晰，优先用编号列表。"""


# ============================================================
# 工具函数
# ============================================================

def _get_user():
    """获取当前登录用户名，未登录返回 None"""
    try:
        return g.user.username if g.user and hasattr(g.user, 'username') else None
    except Exception:
        return None


def _require_auth(f):
    """请求鉴权装饰器：未登录返回 401"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _get_user():
            return jsonify({"status": 1, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def _check_rate_limit(username):
    """
    简单滑动窗口 rate limit：每用户每分钟最多 RATE_LIMIT_PER_MIN 次
    返回 (allowed, retry_after_seconds)
    """
    now = time.time()
    bucket = _RATE_BUCKET[username]
    # 清理超过 60s 的旧记录
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        retry_after = int(60 - (now - bucket[0])) + 1
        return False, max(retry_after, 1)
    bucket.append(now)
    return True, 0


def _pick_first(value):
    """如果是 list 取第一个非空元素，否则原样返回"""
    if isinstance(value, list):
        for v in value:
            if v:
                return v
        return ''
    return value or ''


def _get_llm_config(stream=True):
    """
    获取 LLM 调用配置（url / headers / model）
    优先级：环境变量 ASSISTANT_LLM_URL / ASSISTANT_LLM_API_KEY / ASSISTANT_LLM_MODEL
            → 退化到 conf.CHATGPT_CHAT_URL / conf.CHATGPT_TOKEN / conf.CHATGPT_ARGS.model
            → 都没配置时返回 None，由调用方决定如何提示

    安全：API key 不再硬编码在源码里，必须通过环境变量或 conf 注入。
          部署时通过 docker-compose env_file: .env.assistant 注入。
    """
    # URL（v1 端点会自动补全 /chat/completions）
    url = os.environ.get('ASSISTANT_LLM_URL') or getattr(conf, 'CHATGPT_CHAT_URL', '')
    if url and '/chat/completions' not in url:
        url = url.rstrip('/') + '/chat/completions'

    # Token（必须从 ENV 或 conf 注入，源码不再硬编码）
    api_key = os.environ.get('ASSISTANT_LLM_API_KEY') or getattr(conf, 'CHATGPT_TOKEN', '')

    # Model
    model = os.environ.get('ASSISTANT_LLM_MODEL') or getattr(getattr(conf, 'CHATGPT_ARGS', None), 'model', None) or 'qwen3'

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if stream else 'application/json',
    }
    if api_key:
        # OpenAI 兼容两种 header
        headers['Authorization'] = 'Bearer ' + api_key
        headers['api-key'] = api_key

    return url, headers, model


def _build_messages(session_id, user_message):
    """拼接 system + 历史（从 DB 取最近 MAX_HISTORY_ROUNDS 轮） + 当前用户消息"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 从 DB 拿该 session 最近 MAX_HISTORY_ROUNDS 轮（每轮 user + assistant）
    history_msgs = (
        db.session.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id)
        .order_by(AssistantMessage.created_on.desc())
        .limit(MAX_HISTORY_ROUNDS * 2)
        .all()
    )[::-1]  # 反转变回时间正序
    for m in history_msgs:
        messages.append({"role": m.role, "content": m.content or ''})
    messages.append({"role": "user", "content": user_message})
    return messages


# ============================================================
# 路由
# ============================================================

@assistant_bp.route('/chat', methods=['POST'])
@_require_auth
def chat():
    """
    POST /api/v2/assistant/chat

    Body: { "message": "...", "session_id": "..." }
    Response: text/event-stream
              data: {"text": "..."}\n\n
              data: {"error": "..."}\n\n   (出错时)
              data: {"done": true}\n\n      (结束)
    """
    username = _get_user()
    body = request.get_json(silent=True) or {}
    message = (body.get('message') or '').strip()
    session_id = body.get('session_id') or f'assistant_{username}'

    if not message:
        return jsonify({"status": 1, "error": "message is required"}), 400

    # Rate limit：每用户每分钟 RATE_LIMIT_PER_MIN 次
    allowed, retry_after = _check_rate_limit(username or 'anonymous')
    if not allowed:
        _logger.warning('rate_limited user=%s session=%s msg=%r', username, session_id, message[:50])
        return jsonify({
            "status": 1,
            "error": f"请求过于频繁，请 {retry_after} 秒后重试",
            "retry_after": retry_after,
        }), 429

    url, headers, model = _get_llm_config(stream=True)
    if not url:
        _logger.error('llm_url_missing user=%s', username)
        return jsonify({
            "status": 1,
            "error": "Assistant LLM URL 未配置，请设置环境变量 ASSISTANT_LLM_URL 或 conf.CHATGPT_CHAT_URL"
        }), 500
    if 'Authorization' not in headers:
        _logger.error('llm_apikey_missing user=%s', username)
        return jsonify({
            "status": 1,
            "error": "Assistant LLM API Key 未配置，请设置环境变量 ASSISTANT_LLM_API_KEY 或 conf.CHATGPT_TOKEN"
        }), 500

    # 审计日志：谁、问了什么（截断 100 字）、session_id
    _logger.info('chat_request user=%s session=%s msg_len=%d msg_preview=%r',
                  username, session_id, len(message), message[:100])

    messages = _build_messages(session_id, message)
    payload = {
        'model': model,
        'messages': messages,
        'stream': True,
        'temperature': 0.7,
        'max_tokens': 2048,
        # qwen3 关闭思考模式（避免回复里带 <think> 标签）
        'chat_template_kwargs': {
            'enable_thinking': False,
        },
    }

    def generate():
        import requests
        import sseclient
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        back_message = ''
        # 上游 LLM 抖动重试：5xx / 超时 最多 1 次
        max_retries = 1
        res = None
        for attempt in range(max_retries + 1):
            try:
                res = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    verify=False,
                    timeout=120,
                )
                if res.status_code == 200:
                    break  # 成功，跳出重试循环
                if res.status_code >= 500 and attempt < max_retries:
                    # 5xx 才重试，4xx 不重试（4xx 是客户端问题，重试无用）
                    _logger.warning('llm_upstream_5xx attempt=%d status=%d, retrying', attempt, res.status_code)
                    time.sleep(1)
                    continue
                # 4xx 或重试用完，直接报错
                err_text = ''
                try:
                    err_text = res.text[:300]
                except Exception:
                    pass
                err_msg = f"LLM 上游返回 HTTP {res.status_code}: {err_text}"
                yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
                return
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    _logger.warning('llm_upstream_timeout attempt=%d, retrying', attempt)
                    time.sleep(1)
                    continue
                yield f"data: {json.dumps({'error': 'LLM 上游请求超时'}, ensure_ascii=False)}\n\n"
                return
            except Exception as e:
                if attempt < max_retries:
                    _logger.warning('llm_upstream_error attempt=%d err=%s, retrying', attempt, str(e))
                    time.sleep(1)
                    continue
                yield f"data: {json.dumps({'error': f'内部错误: {str(e)}'}, ensure_ascii=False)}\n\n"
                return

        if res is None or res.status_code != 200:
            # 重试用完仍失败，已在循环里 yield 过错误，这里直接返回
            return

        # 流式解析上游 SSE
        try:
            client = sseclient.SSEClient(res)
            for event in client.events():
                data = event.data
                if not data:
                    continue
                if data == '[DONE]':
                    break
                try:
                    j = json.loads(data)
                    choices = j.get('choices') or []
                    if not choices:
                        continue
                    delta = choices[0].get('delta', {}) or {}
                    text = delta.get('content', '')
                    if text:
                        back_message += text
                        yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    # 非 JSON 或格式不符，跳过
                    continue
        except Exception as e:
            yield f"data: {json.dumps({'error': f'流式解析错误: {str(e)}'}, ensure_ascii=False)}\n\n"
        finally:
            # 保存对话历史到 DB（仅在拿到回复时）
            if back_message.strip():
                try:
                    now = datetime.now()
                    # 写 user 消息
                    db.session.add(AssistantMessage(
                        session_id=session_id,
                        role='user',
                        content=message,
                        created_on=now,
                    ))
                    # 写 assistant 回复
                    db.session.add(AssistantMessage(
                        session_id=session_id,
                        role='assistant',
                        content=back_message,
                        created_on=now,
                    ))
                    # UPSERT conversation：不存在则插入，存在则更新 changed_on + title（仅首次）
                    conv = db.session.query(AssistantConversation).filter_by(
                        session_id=session_id
                    ).first()
                    if conv is None:
                        # 取首条 user 消息前 30 字作为标题
                        title = message[:30] + ('...' if len(message) > 30 else '')
                        db.session.add(AssistantConversation(
                            username=username,
                            session_id=session_id,
                            title=title,
                            created_on=now,
                            changed_on=now,
                        ))
                    else:
                        conv.changed_on = now
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    _logger.error('save_history_failed user=%s session=%s err=%s',
                                  username, session_id, str(e))
            # 审计日志：回复长度
            _logger.info('chat_response user=%s session=%s reply_len=%d',
                          username, session_id, len(back_message))
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # 禁用 nginx 缓冲，确保实时流式
            'Connection': 'keep-alive',
        },
    )


@assistant_bp.route('/sessions', methods=['GET'])
@_require_auth
def list_sessions():
    """GET /api/v2/assistant/sessions — 列出当前用户会话（懒清理 30 天前）"""
    username = _get_user()
    now = datetime.now()

    # 懒清理：删除 30 天前的会话及其消息（不引入 cron）
    cutoff = now - timedelta(days=HISTORY_RETAIN_DAYS)
    try:
        expired = db.session.query(AssistantConversation).filter(
            AssistantConversation.username == username,
            AssistantConversation.changed_on < cutoff,
        ).all()
        for c in expired:
            db.session.query(AssistantMessage).filter_by(session_id=c.session_id).delete()
            db.session.delete(c)
        if expired:
            db.session.commit()
            _logger.info('lazy_cleaned user=%s expired_count=%d', username, len(expired))
    except Exception as e:
        db.session.rollback()
        _logger.error('lazy_clean_failed user=%s err=%s', username, str(e))

    # 返回当前用户的会话列表，按 changed_on 倒序
    sessions = (
        db.session.query(AssistantConversation)
        .filter(AssistantConversation.username == username)
        .order_by(AssistantConversation.changed_on.desc())
        .limit(50)
        .all()
    )
    return jsonify({
        "status": 0,
        "data": [
            {
                "session_id": s.session_id,
                "title": s.title or '',
                "created_on": s.created_on.strftime('%Y-%m-%d %H:%M:%S') if s.created_on else '',
                "changed_on": s.changed_on.strftime('%Y-%m-%d %H:%M:%S') if s.changed_on else '',
            }
            for s in sessions
        ],
        "count": len(sessions),
    })


@assistant_bp.route('/session/new', methods=['POST'])
@_require_auth
def new_session():
    """POST /api/v2/assistant/session/new — 创建新会话，返回 session_id"""
    username = _get_user()
    # 生成 session_id：assistant_<user>_<uuid 前 8 位>
    sid = f'assistant_{username}_{uuid.uuid4().hex[:8]}'
    # 不立即写库（等首次 /chat 完成时再 UPSERT），避免空会话占行
    return jsonify({
        "status": 0,
        "session_id": sid,
    })


@assistant_bp.route('/history/<session_id>', methods=['GET'])
@_require_auth
def get_history(session_id):
    """GET /api/v2/assistant/history/<session_id> — 取该会话的所有消息"""
    msgs = (
        db.session.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id)
        .order_by(AssistantMessage.created_on.asc(), AssistantMessage.id.asc())
        .all()
    )
    return jsonify({
        "status": 0,
        "data": [
            {"role": m.role, "content": m.content or '', "created_on": m.created_on.strftime('%Y-%m-%d %H:%M:%S') if m.created_on else ''}
            for m in msgs
        ],
        "count": len(msgs),
    })


@assistant_bp.route('/history/<session_id>', methods=['DELETE'])
@_require_auth
def clear_history(session_id):
    """DELETE /api/v2/assistant/history/<session_id> — 删除该会话及其所有消息"""
    username = _get_user()
    # 权限校验：只能删自己的会话（虽然 session_id 是前端传，但验证 user 一致更稳）
    conv = db.session.query(AssistantConversation).filter_by(session_id=session_id).first()
    if conv and conv.username != username:
        return jsonify({"status": 1, "error": "无权删除他人会话"}), 403
    try:
        db.session.query(AssistantMessage).filter_by(session_id=session_id).delete()
        if conv:
            db.session.delete(conv)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 1, "error": f"删除失败: {str(e)}"}), 500
    return jsonify({"status": 0, "msg": "history cleared"})
