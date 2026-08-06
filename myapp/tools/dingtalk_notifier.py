"""DingTalk group notification adapter for MLOps status events."""
import base64
import hashlib
import hmac
import logging
import time
from datetime import datetime
from urllib.parse import quote_plus

import requests

LOGGER = logging.getLogger(__name__)


def _signed_webhook(webhook, secret):
    if not secret:
        return webhook
    timestamp = str(round(time.time() * 1000))
    digest = hmac.new(secret.encode('utf-8'), f'{timestamp}\n{secret}'.encode('utf-8'), hashlib.sha256).digest()
    separator = '&' if '?' in webhook else '?'
    return f'{webhook}{separator}timestamp={timestamp}&sign={quote_plus(base64.b64encode(digest))}'


def _message_text(receivers, message, link):
    receiver_text = ', '.join(str(item) for item in (receivers or []) if item)
    lines = ['【MLOps 平台通知】', str(message)]
    if receiver_text:
        lines.append(f'相关用户：{receiver_text}')
    if isinstance(link, dict):
        lines.extend(f'{name}：{url}' for name, url in link.items())
    elif link:
        lines.append(f'查看详情：{link}')
    return '\n'.join(lines)


def _load_channel(channel_id=None):
    from myapp import app, db
    try:
        from myapp.models.model_notification import NotificationChannel
        query = db.session.query(NotificationChannel)
        channel = query.filter_by(id=channel_id).first() if channel_id else query.filter_by(channel_type='dingtalk').first()
        if channel:
            from myapp.tools.notification_crypto import decrypt_secret
            try:
                event_types = set(__import__('json').loads(channel.event_types or '[]'))
            except (TypeError, ValueError):
                event_types = set()
            return (
                channel, bool(channel.enabled),
                decrypt_secret(channel.webhook_encrypted),
                decrypt_secret(channel.secret_encrypted), event_types,
            )
    except Exception:
        db.session.rollback()
        LOGGER.debug('notification channel table unavailable; use config fallback', exc_info=True)
    return (
        None, bool(app.config.get('DINGTALK_NOTIFICATION_ENABLED', False)),
        str(app.config.get('DINGTALK_NOTIFICATION_WEBHOOK', '')).strip(),
        str(app.config.get('DINGTALK_NOTIFICATION_SECRET', '')).strip(),
        set(app.config.get('DINGTALK_NOTIFICATION_EVENT_TYPES', [])),
    )


def _already_sent(channel, dedup_key):
    if not channel or not dedup_key:
        return False
    try:
        from myapp import db
        from myapp.models.model_notification import NotificationDelivery
        return db.session.query(NotificationDelivery.id).filter_by(
            channel_id=channel.id, dedup_key=dedup_key, status='SUCCESS'
        ).first() is not None
    except Exception:
        return False

def _record(channel, event_type, dedup_key, status, message, response_code=None, error=''):
    try:
        from myapp import db
        from myapp.models.model_notification import NotificationDelivery
        row = NotificationDelivery(
            channel_id=channel.id if channel else None, event_type=event_type,
            dedup_key=dedup_key, status=status, response_code=response_code,
            message_summary=str(message)[:500], error=str(error)[:4000],
            sent_at=datetime.now() if status == 'SUCCESS' else None,
        )
        db.session.add(row)
        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
        LOGGER.debug('cannot persist notification delivery', exc_info=True)


def send_dingtalk_message(receivers, message, link=None, event_type='platform.message', dedup_key=None, force=False, channel_id=None):
    """Send an event; failures are logged and never interrupt workflow monitoring."""
    from myapp import app
    with app.app_context():
        channel, enabled, webhook, secret, event_types = _load_channel(channel_id)
        if (not force and (not enabled or event_type not in event_types)) or not webhook:
            return False
        if not force and _already_sent(channel, dedup_key):
            return True
        try:
            response = requests.post(
                _signed_webhook(webhook, secret),
                json={'msgtype': 'text', 'text': {'content': _message_text(receivers, message, link)}},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json() if response.content else {}
            if result.get('errcode', 0) != 0:
                raise RuntimeError(result.get('errmsg') or str(result))
            _record(channel, event_type, dedup_key, 'SUCCESS', message, response.status_code)
            return True
        except Exception as exc:
            LOGGER.exception('failed to send MLOps event to DingTalk')
            _record(channel, event_type, dedup_key, 'FAILED', message, getattr(getattr(exc, 'response', None), 'status_code', None), exc)
            return False
