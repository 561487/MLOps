import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


def _fernet():
    from myapp import app
    raw_key = str(app.config.get('NOTIFICATION_ENCRYPTION_KEY') or app.config.get('SECRET_KEY') or '')
    if not raw_key:
        raise RuntimeError('NOTIFICATION_ENCRYPTION_KEY/SECRET_KEY is not configured')
    key = base64.urlsafe_b64encode(hashlib.sha256(raw_key.encode('utf-8')).digest())
    return Fernet(key)


def encrypt_secret(value):
    value = (value or '').strip()
    return _fernet().encrypt(value.encode('utf-8')).decode('ascii') if value else ''


def decrypt_secret(value):
    if not value:
        return ''
    try:
        return _fernet().decrypt(value.encode('ascii')).decode('utf-8')
    except InvalidToken as exc:
        raise RuntimeError('notification secret cannot be decrypted; check encryption key') from exc
