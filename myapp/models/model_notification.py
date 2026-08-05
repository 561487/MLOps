from datetime import datetime

from flask_appbuilder import Model
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text


class NotificationChannel(Model):
    __tablename__ = 'mlops_notification_channel'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, default='钉钉任务通知')
    channel_type = Column(String(32), nullable=False, default='dingtalk')
    enabled = Column(Boolean, nullable=False, default=False)
    webhook_encrypted = Column(Text, nullable=True)
    secret_encrypted = Column(Text, nullable=True)
    project_scope = Column(Text, nullable=True, default='')
    cluster_scope = Column(Text, nullable=True, default='')
    namespace_scope = Column(Text, nullable=True, default='')
    event_types = Column(Text, nullable=False, default='["workflow.succeeded","workflow.failed"]')
    pending_timeout = Column(Integer, nullable=False, default=300)
    silence_seconds = Column(Integer, nullable=False, default=1800)
    resource_monitor_enabled = Column(Boolean, nullable=False, default=False)
    cpu_threshold = Column(Integer, nullable=False, default=90)
    memory_threshold = Column(Integer, nullable=False, default=90)
    gpu_threshold = Column(Integer, nullable=False, default=95)
    gpu_memory_threshold = Column(Integer, nullable=False, default=90)
    gpu_temperature_threshold = Column(Integer, nullable=False, default=85)
    threshold_duration = Column(Integer, nullable=False, default=120)
    completion_summary = Column(Boolean, nullable=False, default=True)
    created_by = Column(String(100), nullable=True, default='')
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class NotificationDelivery(Model):
    __tablename__ = 'mlops_notification_delivery'

    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, nullable=True)
    event_type = Column(String(100), nullable=False, default='platform.message')
    dedup_key = Column(String(512), nullable=True, index=True)
    status = Column(String(32), nullable=False, default='PENDING')
    retry_count = Column(Integer, nullable=False, default=0)
    response_code = Column(Integer, nullable=True)
    message_summary = Column(String(500), nullable=True, default='')
    error = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
