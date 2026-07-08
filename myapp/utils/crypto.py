"""
加密工具模块
用于敏感参数（如 API Token）的加密存储。

使用 Fernet 对称加密，密钥从 app.config['SECRET_KEY'] 派生。
加密后的值以 ENC: 前缀标记，便于识别和兼容未加密的旧数据。
"""

import base64
import hashlib

from cryptography.fernet import Fernet

ENC_PREFIX = 'ENC:'


def _get_fernet():
    """从 Flask app secret key 派生 Fernet 密钥"""
    from myapp import app
    secret_key = app.config.get('SECRET_KEY', 'default-secret-key')
    # Fernet 需要 32 字节的 base64 编码密钥
    key_bytes = hashlib.sha256(secret_key.encode('utf-8')).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_value(plaintext: str) -> str:
    """
    加密明文字符串，返回 ENC: 前缀的密文。
    如果已经是 ENC: 前缀，直接返回（防止重复加密）。
    """
    if not plaintext:
        return plaintext
    if plaintext.startswith(ENC_PREFIX):
        return plaintext
    f = _get_fernet()
    encrypted = f.encrypt(plaintext.encode('utf-8'))
    return ENC_PREFIX + encrypted.decode('utf-8')


def decrypt_value(ciphertext: str) -> str:
    """
    解密 ENC: 前缀的密文，返回明文。
    如果不是 ENC: 前缀（旧数据或未加密），直接返回原值。
    """
    if not ciphertext:
        return ciphertext
    if not ciphertext.startswith(ENC_PREFIX):
        return ciphertext
    f = _get_fernet()
    encrypted = ciphertext[len(ENC_PREFIX):].encode('utf-8')
    return f.decrypt(encrypted).decode('utf-8')


def is_encrypted(value: str) -> bool:
    """检查值是否已加密"""
    return bool(value) and value.startswith(ENC_PREFIX)
