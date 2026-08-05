import json
import logging

from flask import Blueprint, g, jsonify, request
from sqlalchemy.exc import OperationalError, ProgrammingError

from myapp import appbuilder, db
from myapp.models.model_notification import NotificationChannel, NotificationDelivery
from myapp.tools.dingtalk_notifier import send_dingtalk_message
from myapp.tools.notification_crypto import encrypt_secret

notification_api_bp = Blueprint('notification_api', __name__, url_prefix='/api/notification')
LOGGER = logging.getLogger(__name__)
DEFAULT_EVENTS = ['workflow.succeeded', 'workflow.failed']


def _is_admin():
    user = getattr(g, 'user', None)
    if not user or not getattr(user, 'is_authenticated', False):
        return False
    try:
        return bool(user.is_admin())
    except Exception:
        try:
            return appbuilder.sm.check_user_is_admin(user)
        except Exception:
            return False


def _admin_required(func):
    from functools import wraps
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not _is_admin():
            return jsonify({'status': 1, 'error': '仅平台管理员可以管理通知渠道'}), 403
        return func(*args, **kwargs)
    return wrapped


def _channel():
    return db.session.query(NotificationChannel).filter_by(channel_type='dingtalk').first()


def _public(channel):
    if not channel:
        return {
            'id': None, 'name': '钉钉任务通知', 'enabled': False,
            'webhookConfigured': False, 'secretConfigured': False,
            'projectScope': '', 'clusterScope': '', 'namespaceScope': '',
            'eventTypes': DEFAULT_EVENTS, 'pendingTimeout': 300, 'silenceSeconds': 1800,
            'resourceMonitorEnabled': False, 'cpuThreshold': 90, 'memoryThreshold': 90,
            'gpuThreshold': 95, 'gpuMemoryThreshold': 90, 'gpuTemperatureThreshold': 85,
            'thresholdDuration': 120, 'completionSummary': True,
        }
    try:
        event_types = json.loads(channel.event_types or '[]')
    except (TypeError, ValueError):
        event_types = DEFAULT_EVENTS
    return {
        'id': channel.id, 'name': channel.name, 'enabled': bool(channel.enabled),
        'webhookConfigured': bool(channel.webhook_encrypted),
        'secretConfigured': bool(channel.secret_encrypted),
        'projectScope': channel.project_scope or '', 'clusterScope': channel.cluster_scope or '',
        'namespaceScope': channel.namespace_scope or '', 'eventTypes': event_types,
        'pendingTimeout': channel.pending_timeout, 'silenceSeconds': channel.silence_seconds,
        'resourceMonitorEnabled': bool(channel.resource_monitor_enabled),
        'cpuThreshold': channel.cpu_threshold, 'memoryThreshold': channel.memory_threshold,
        'gpuThreshold': channel.gpu_threshold, 'gpuMemoryThreshold': channel.gpu_memory_threshold,
        'gpuTemperatureThreshold': channel.gpu_temperature_threshold,
        'thresholdDuration': channel.threshold_duration, 'completionSummary': bool(channel.completion_summary),
        'updatedAt': channel.updated_at.isoformat() if channel.updated_at else '',
    }


@notification_api_bp.route('/dingtalk', methods=['GET', 'PUT'])
@_admin_required
def dingtalk_config():
    try:
        channel = _channel()
        if request.method == 'GET':
            return jsonify({'status': 0, 'result': _public(channel)})
        data = request.get_json(silent=True) or {}
        webhook = str(data.get('webhook') or '').strip()
        secret = str(data.get('secret') or '').strip()
        if webhook and not webhook.startswith('https://oapi.dingtalk.com/robot/send'):
            return jsonify({'status': 1, 'error': 'Webhook 必须是钉钉群机器人 HTTPS 地址'}), 400
        if not channel:
            channel = NotificationChannel(channel_type='dingtalk', created_by=g.user.username)
            db.session.add(channel)
        channel.name = str(data.get('name') or channel.name or '钉钉任务通知').strip()
        channel.enabled = bool(data.get('enabled', False))
        channel.project_scope = str(data.get('projectScope') or '').strip()
        channel.cluster_scope = str(data.get('clusterScope') or '').strip()
        channel.namespace_scope = str(data.get('namespaceScope') or '').strip()
        channel.event_types = json.dumps(data.get('eventTypes') or DEFAULT_EVENTS, ensure_ascii=False)
        channel.pending_timeout = max(60, min(int(data.get('pendingTimeout') or 300), 86400))
        channel.silence_seconds = max(0, min(int(data.get('silenceSeconds') or 1800), 604800))
        channel.resource_monitor_enabled = bool(data.get('resourceMonitorEnabled', False))
        channel.cpu_threshold = max(1, min(int(data.get('cpuThreshold') or 90), 100))
        channel.memory_threshold = max(1, min(int(data.get('memoryThreshold') or 90), 100))
        channel.gpu_threshold = max(1, min(int(data.get('gpuThreshold') or 95), 100))
        channel.gpu_memory_threshold = max(1, min(int(data.get('gpuMemoryThreshold') or 90), 100))
        channel.gpu_temperature_threshold = max(30, min(int(data.get('gpuTemperatureThreshold') or 85), 120))
        channel.threshold_duration = max(10, min(int(data.get('thresholdDuration') or 120), 86400))
        channel.completion_summary = bool(data.get('completionSummary', True))
        if webhook:
            channel.webhook_encrypted = encrypt_secret(webhook)
        if secret:
            channel.secret_encrypted = encrypt_secret(secret)
        if channel.enabled and not channel.webhook_encrypted:
            return jsonify({'status': 1, 'error': '启用通知前必须配置 Webhook'}), 400
        db.session.commit()
        return jsonify({'status': 0, 'message': '钉钉通知配置已保存', 'result': _public(channel)})
    except (ProgrammingError, OperationalError):
        db.session.rollback()
        return jsonify({'status': 1, 'error': '通知表尚未初始化，请先执行数据库升级'}), 503
    except Exception as exc:
        db.session.rollback()
        LOGGER.exception('save dingtalk notification config failed')
        return jsonify({'status': 1, 'error': str(exc)}), 500


@notification_api_bp.route('/dingtalk/test', methods=['POST'])
@_admin_required
def test_dingtalk():
    channel = _channel()
    if not channel or not channel.webhook_encrypted:
        return jsonify({'status': 1, 'error': '请先保存钉钉 Webhook'}), 400
    ok = send_dingtalk_message(
        [g.user.username], '钉钉通知渠道测试成功', '/frontend/',
        event_type='channel.test', force=True, channel_id=channel.id,
    )
    if not ok:
        return jsonify({'status': 1, 'error': '发送失败，请查看通知记录或后端日志'}), 502
    return jsonify({'status': 0, 'message': '测试消息已发送到钉钉群'})


@notification_api_bp.route('/deliveries', methods=['GET'])
@_admin_required
def deliveries():
    rows = db.session.query(NotificationDelivery).order_by(NotificationDelivery.id.desc()).limit(50).all()
    return jsonify({'status': 0, 'result': [{
        'id': row.id, 'eventType': row.event_type, 'status': row.status,
        'summary': row.message_summary, 'error': row.error or '',
        'sentAt': row.sent_at.isoformat() if row.sent_at else '',
        'createdAt': row.created_at.isoformat() if row.created_at else '',
    } for row in rows]})
