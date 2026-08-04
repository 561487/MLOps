"""
Chat API v2.0 — 重构后的对话 Blueprint
=========================================

路由前缀：/api/v2/chat

API 列表：
  GET    /agents                     智能体列表（支持 ?category=robot|knowledge_base|agent）
  GET    /agents/<agent_name>         智能体详情（含 credentials）
  PUT    /agents/<agent_name>/config  更新智能体配置
  POST   /sessions                    创建新会话
  GET    /sessions                    会话列表（?agent=xxx）
  DELETE /sessions/<session_id>       删除会话
  POST   /chat/<agent_name>           SSE 流式对话
  GET    /history/<session_id>        聊天历史

与旧版 view_chat.py 的关系：
  - 并存运行，旧版 `/chat_modelview/api/` 不动
  - 新版 SSE 对话复用 Chat_View_Base.chat() 的核心流式生成逻辑
"""
import json
import uuid
import datetime
from typing import Optional


def _safe_json(value, default=None):
    """Safely parse JSON, returning default on failure."""
    if default is None:
        default = {}
    if not value or not value.strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


from flask import Blueprint, request, jsonify, g, Response
from flask import stream_with_context
from sqlalchemy import or_, desc

from myapp.models.model_chat import Chat, ChatLog
from myapp.chat_providers import get_provider
from myapp import appbuilder, db, cache

# ===== Blueprint 定义 =====
chat_api_bp = Blueprint('chat_api_v2', __name__, url_prefix='/api/v2/chat')


# ============================================================
# 工具函数
# ============================================================

def _get_user() -> Optional[str]:
    """获取当前登录用户名，未登录返回 None"""
    try:
        return g.user.username if g.user and hasattr(g.user, 'username') else None
    except Exception:
        return None


def _is_admin() -> bool:
    """判断当前用户是否是管理员"""
    try:
        return appbuilder.sm.check_user_is_admin(g.user) if g.user else False
    except Exception:
        return False


def _require_auth(f):
    """请求鉴权装饰器：未登录返回 401"""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if not _get_user():
            return jsonify({"status": 1, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated


def _build_agent_item(agent: Chat) -> dict:
    """将 Chat ORM 对象转换为前端可用的智能体摘要"""
    expand = _safe_json(agent.expand, {})
    item = {
        "name": agent.name,
        "label": agent.label,
        "icon": agent.icon,
        "agentCategory": agent.agent_category or 'robot',
        "chatType": agent.chat_type,
        "hello": agent.hello,
        "tips": _safe_json(agent.tips, []),
        "owner": agent.owner,
    }

    # 返回外链跳转信息（任何配置了 externalUrl 的智能体）
    external_url = expand.get("externalUrl", "")
    if external_url:
        item["externalUrl"] = external_url
        item["openInNewTab"] = expand.get("openInNewTab", True)

    return item


def _build_agent_detail(agent: Chat) -> dict:
    """将 Chat ORM 对象转换为前端可用的智能体完整信息（含凭证）"""
    expand = _safe_json(agent.expand, {})
    service_config = _safe_json(agent.service_config, {})
    credentials = _safe_json(agent.credentials, {})

    category = agent.agent_category or 'robot'

    result = {
        "name": agent.name,
        "label": agent.label,
        "icon": agent.icon,
        "agentCategory": category,
        "chatType": agent.chat_type or 'text',
        "serviceType": agent.service_type or '',
        "hello": agent.hello,
        "tips": _safe_json(agent.tips, []),
        "prompt": agent.prompt,
        "sessionNum": int(agent.session_num or 0),
        "serviceConfig": service_config,
        "credentials": credentials,
        "knowledge": _safe_json(agent.knowledge, {}),
        "expand": expand,
        "configFields": expand.get("configFields", _default_config_fields(category)),
        "owner": agent.owner or '*',
    }

    if category == 'knowledge_base':
        result["externalUrl"] = expand.get("externalUrl", "")
        result["openInNewTab"] = expand.get("openInNewTab", True)

    return result


def _default_config_fields(category: str) -> list:
    """
    按智能体分类返回默认的凭证配置字段描述。

    前端根据此描述动态渲染表单，无需硬编码字段。
    用户可以在此追加/修改字段，存入 expand.configFields。

    robot（机器人）：
      - API Key 认证，只需 Endpoint + Key

    knowledge_base（知识库）：
      - 外链跳转，需要 URL + 认证凭据

    agent（智能体）：
      - 多认证方式，API Key / AppID+Secret / Bearer Token
    """
    if category == 'knowledge_base':
        return [
            {"key": "externalUrl", "label": "知识库地址", "type": "text", "required": True},
            {"key": "externalUsername", "label": "用户名", "type": "text"},
            {"key": "externalPassword", "label": "密码", "type": "password"},
        ]
    elif category == 'agent':
        return [
            {"key": "authType", "label": "认证方式", "type": "select",
             "options": [
                 {"label": "API Key", "value": "api_key"},
                 {"label": "App ID + Secret", "value": "app_secret"},
                 {"label": "Bearer Token", "value": "token"},
             ]},
            {"key": "apiKey", "label": "API Key / Token", "type": "password"},
            {"key": "appId", "label": "App ID", "type": "text"},
            {"key": "secretKey", "label": "Secret Key", "type": "password"},
            {"key": "apiEndpoint", "label": "API Endpoint", "type": "text"},
            {"key": "extraParams", "label": "扩展参数(JSON)", "type": "json"},
            {"key": "description", "label": "备注", "type": "textarea"},
        ]
    else:  # robot (默认)
        return [
            {"key": "authType", "label": "认证方式", "type": "select",
             "options": [
                 {"label": "API Key", "value": "api_key"},
                 {"label": "Bearer Token", "value": "token"},
             ]},
            {"key": "apiKey", "label": "API Key", "type": "password", "required": True},
            {"key": "apiEndpoint", "label": "API Endpoint", "type": "text",
             "placeholder": "http://10.80.10.146:9998/v1/chat/completions"},
            {"key": "description", "label": "备注", "type": "textarea"},
        ]


# ============================================================
# API - 智能体列表
# ============================================================

@chat_api_bp.route('/agents', methods=['GET'])
@_require_auth
def list_agents():
    """
    GET /api/v2/chat/agents?category=robot|knowledge_base|agent

    返回当前用户可访问的智能体列表。
    管理员可看全部，普通用户只看 owner 包含自己或 '*' 的。

    响应格式：
    {
      "status": 0,
      "result": [
        {
          "name": "default_robot",
          "label": "默认机器人",
          "icon": "...",
          "agentCategory": "robot",
          "hello": "你好，有什么可以帮助你的？",
          ...
        }
      ]
    }
    """
    query = db.session.query(Chat)

    # 权限过滤：非管理员只看到自己可访问的
    if not _is_admin():
        username = _get_user()
        query = query.filter(
            or_(Chat.owner.contains(username), Chat.owner.contains('*'))
        )

    # 分类过滤
    category = request.args.get('category')
    if category in ('robot', 'knowledge_base', 'agent'):
        if category == 'agent':
            query = query.filter(Chat.agent_category.in_(['agent', 'external_link']))
        else:
            query = query.filter(Chat.agent_category == category)

    agents = query.order_by(Chat.id.asc()).all()

    result = [_build_agent_item(a) for a in agents]
    return jsonify({"status": 0, "result": result})


# ============================================================
# API - 智能体详情
# ============================================================

@chat_api_bp.route('/agents/<agent_name>', methods=['GET'])
@_require_auth
def get_agent(agent_name):
    """
    GET /api/v2/chat/agents/<agent_name>

    返回单个智能体的完整配置，包括 credentials 和 configFields，
    前端据此渲染聊天界面和配置侧边栏。

    响应格式：
    {
      "status": 0,
      "result": {
        "name": "default_robot",
        "credentials": { "apiKey": "sk-xxx", "apiEndpoint": "http://..." },
        "configFields": [ { "key": "apiKey", "label": "API Key", ... } ],
        ...
      }
    }
    """
    agent = db.session.query(Chat).filter_by(name=agent_name).first()
    if not agent:
        return jsonify({"status": 1, "error": f"Agent '{agent_name}' not found"}), 404

    return jsonify({"status": 0, "result": _build_agent_detail(agent)})


# ============================================================
# API - 更新配置
# ============================================================

@chat_api_bp.route('/agents/<agent_name>/config', methods=['PUT'])
@_require_auth
def update_agent_config(agent_name):
    """
    PUT /api/v2/chat/agents/<agent_name>/config

    更新智能体配置，可以单独更新 credentials 或 configFields。
    Body 示例：
    {
      "credentials": { "apiKey": "sk-new-key" },
      "configFields": [ ... ],
      "prompt": "新提示词模板"
    }
    """
    agent = db.session.query(Chat).filter_by(name=agent_name).first()
    if not agent:
        return jsonify({"status": 1, "error": f"Agent '{agent_name}' not found"}), 404

    data = request.get_json(silent=True) or {}

    # 更新凭证（独立字段）
    if 'credentials' in data:
        agent.credentials = json.dumps(data['credentials'], ensure_ascii=False)

    # 更新配置字段描述（存在 expand 中）
    if 'configFields' in data:
        expand = _safe_json(agent.expand, {})
        expand['configFields'] = data['configFields']
        agent.expand = json.dumps(expand, ensure_ascii=False)

    # 更新其他简单字段
    for field in ['prompt', 'hello', 'tips', 'session_num', 'label', 'icon', 'chat_type']:
        if field in data:
            val = data[field]
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            setattr(agent, field, val)

    db.session.commit()
    return jsonify({"status": 0, "message": "Config updated"})


# ============================================================
# ============================================================
# 会话管理（用户隔离 + 缓存持久化）
# ============================================================

SESSION_KEY_PREFIX = 'chat_v2_session'
SESSION_INDEX_PREFIX = 'chat_v2_index'
SESSION_TTL = 86400 * 7


def _session_key(username, agent_name, session_id):
    return f'{SESSION_KEY_PREFIX}:{username}:{agent_name}:{session_id}'


def _session_index_key(username, agent_name):
    return f'{SESSION_INDEX_PREFIX}:{username}:{agent_name}'


def _session_meta(session_id, agent_name, title='',
                  last_message='', msg_count=0):
    import datetime
    return {
        'id': session_id,
        'agentName': agent_name,
        'title': title or '新对话',
        'lastMessage': last_message,
        'createdAt': int(datetime.datetime.now().timestamp() * 1000),
        'messageCount': msg_count,
    }


def _save_session(username, agent_name, session_id, data, ttl=None):
    if ttl is None:
        ttl = SESSION_TTL
    key = _session_key(username, agent_name, session_id)
    cache.set(key, json.dumps(data, ensure_ascii=False), timeout=ttl)


def _load_session(username, agent_name, session_id):
    raw = cache.get(_session_key(username, agent_name, session_id))
    return _safe_json(raw) if raw else None


def _delete_session_cache(username, agent_name, session_id):
    cache.delete(_session_key(username, agent_name, session_id))


def _add_to_index(username, agent_name, session_id, meta):
    idx_key = _session_index_key(username, agent_name)
    raw = cache.get(idx_key)
    index = _safe_json(raw, []) if raw else []
    index = [s for s in index if s.get('id') != session_id]
    index.insert(0, meta)
    index = index[:100]
    cache.set(idx_key, json.dumps(index, ensure_ascii=False), timeout=SESSION_TTL)


def _remove_from_index(username, agent_name, session_id):
    idx_key = _session_index_key(username, agent_name)
    raw = cache.get(idx_key)
    if not raw:
        return
    index = _safe_json(raw, [])
    index = [s for s in index if s.get('id') != session_id]
    cache.set(idx_key, json.dumps(index, ensure_ascii=False), timeout=SESSION_TTL)


def _get_index(username, agent_name):
    raw = cache.get(_session_index_key(username, agent_name))
    return _safe_json(raw, []) if raw else []


@chat_api_bp.route('/sessions', methods=['POST'])
@_require_auth
def create_session():
    data = request.get_json(silent=True) or {}
    agent_name = data.get('agent_name', '')
    username = _get_user()
    session_id = str(uuid.uuid4())[:8]

    meta = _session_meta(session_id, agent_name)
    session_data = {**meta, 'username': username, 'history': []}

    _save_session(username, agent_name, session_id, session_data)
    _add_to_index(username, agent_name, session_id, meta)

    return jsonify({'status': 0, 'result': meta})


@chat_api_bp.route('/sessions', methods=['GET'])
@_require_auth
def list_sessions():
    agent_name = request.args.get('agent', '')
    username = _get_user()

    if not agent_name:
        return jsonify({'status': 1, 'error': 'agent name required'}), 400

    index = _get_index(username, agent_name)

    valid = []
    for meta in index:
        sid = meta.get('id', '')
        if _load_session(username, agent_name, sid):
            valid.append(meta)
    if len(valid) != len(index):
        cache.set(
            _session_index_key(username, agent_name),
            json.dumps(valid, ensure_ascii=False),
            timeout=SESSION_TTL,
        )

    return jsonify({'status': 0, 'result': valid})


@chat_api_bp.route('/sessions/<session_id>', methods=['DELETE'])
@_require_auth
def delete_session(session_id):
    username = _get_user()
    agent_name = request.args.get('agent', '')

    if not agent_name:
        return jsonify({'status': 1, 'error': 'agent name required'}), 400

    _delete_session_cache(username, agent_name, session_id)
    _remove_from_index(username, agent_name, session_id)

    return jsonify({'status': 0, 'message': 'Session deleted'})


@chat_api_bp.route('/sessions/<session_id>', methods=['PATCH'])
@_require_auth
def update_session(session_id):
    username = _get_user()
    agent_name = request.args.get('agent', '')
    data = request.get_json(silent=True) or {}
    title = data.get('title', '')

    if not agent_name:
        return jsonify({'status': 1, 'error': 'agent name required'}), 400
    if not title:
        return jsonify({'status': 1, 'error': 'title required'}), 400

    sess = _load_session(username, agent_name, session_id)
    if not sess:
        return jsonify({'status': 1, 'error': 'session not found'}), 404

    sess['title'] = title
    _save_session(username, agent_name, session_id, sess)

    index = _get_index(username, agent_name)
    updated = False
    for item in index:
        if item.get('id') == session_id:
            item['title'] = title
            updated = True
            break
    if updated:
        from flask import current_app
        cache = current_app.config.get('CACHE')
        if cache:
            cache.set(
                _session_index_key(username, agent_name),
                json.dumps(index, ensure_ascii=False),
                timeout=SESSION_TTL,
            )

    return jsonify({'status': 0, 'result': {'id': session_id, 'title': title}})


# API - SSE 流式对话（CORE）
# ============================================================

@chat_api_bp.route('/chat/<agent_name>', methods=['POST'])
@_require_auth
def chat(agent_name):
    """
    POST /api/v2/chat/chat/<agent_name>

    核心 SSE 流式对话接口，完全复用现版 Chat_View_Base.chat() 的流式生成逻辑。

    Body: { "session_id": "...", "search_text": "你好", "stream": true }

    限制：
      - knowledge_base 类型不可对话（返回 400）
      - robot/agent 类型走 LLM 流式调用

    响应：text/event-stream
    """
    # 查找智能体
    agent = db.session.query(Chat).filter_by(name=agent_name).first()
    if not agent:
        return jsonify({"status": 1, "error": f"Agent '{agent_name}' not found"}), 404

    # 知识库：禁止对话
    if agent.agent_category == 'knowledge_base':
        return jsonify({
            "status": 1,
            "error": "知识库类不支持对话，请直接访问外链"
        }), 400

    # 检查是否已配置 API 地址和密钥
    import json as _json
    sc = _json.loads(agent.service_config) if agent.service_config else {}
    cred = _json.loads(agent.credentials) if agent.credentials else {}
    url = sc.get('llm_url', '') or cred.get('apiEndpoint', '')
    tokens = sc.get('llm_tokens', []) or ([cred['apiKey']] if cred.get('apiKey') else [])
    if not url or not tokens:
        return jsonify({
            "status": 1,
            "error": "该智能体尚未配置 API 地址和密钥，请先前往配置面板填写"
        }), 400

    # 解析请求体
    args = request.get_json(silent=True) or {}

    # ===== 复用现有 view_chat.Chat_View_Base.chat() 的完整逻辑 =====
    # 该方法的流式响应已经被充分测试和验证，包括：
    #   - token 池轮转（get_llm_url_header）
    #   - 提示词渲染（generate_prompt）
    #   - 会话历史缓存（cache.set）
    #   - before/after 文本处理
    #   - SSE 格式封装
    from myapp.views.view_chat import Chat_View_Base
    chat_view = Chat_View_Base()
    return chat_view.chat(agent_name, args=args)


# ============================================================
# API - 聊天历史
# ============================================================

@chat_api_bp.route('/history/<session_id>', methods=['GET'])
@_require_auth
def get_history(session_id):
    agent_name = request.args.get('agent', '')
    username = _get_user()
    limit = request.args.get('limit', 20, type=int)

    # 1. 从新缓存读取
    sess = _load_session(username, agent_name, session_id)
    messages = []
    new_history_empty = True

    if sess:
        history = sess.get('history', [])
        if history:
            new_history_empty = False
            for h in history:
                if isinstance(h, list) and len(h) >= 2:
                    messages.append({'role': 'user', 'content': h[0]})
                    messages.append({'role': 'assistant', 'content': h[1]})

    # 2. 兼容旧版 key: chat_{session_id}
    if new_history_empty:
        from flask import current_app
        cache = current_app.config.get('CACHE')
        if cache:
            old_history = cache.get('chat_' + session_id)
            if old_history:
                for entry in old_history:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        messages.append({'role': 'user', 'content': entry[0]})
                        messages.append({'role': 'assistant', 'content': entry[1]})

                # 迁移到新 key（延长 TTL 至 7 天并用户隔离）
                migrated = [[entry[0], entry[1]] for entry in old_history
                            if isinstance(entry, (list, tuple)) and len(entry) >= 2]
                if migrated and sess:
                    sess['history'] = migrated
                    _save_session(username, agent_name, session_id, sess)
                elif migrated:
                    meta = _session_meta(session_id, agent_name)
                    session_data = {**meta, 'username': username, 'history': migrated}
                    _save_session(username, agent_name, session_id, session_data)
                    _add_to_index(username, agent_name, session_id, meta)

    return jsonify({
        'status': 0,
        'result': {
            'sessionId': session_id,
            'messages': messages[-limit * 2:],
        }
    })
