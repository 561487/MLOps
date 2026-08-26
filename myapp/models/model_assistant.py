"""
浮窗 AI 助手 — 对话历史持久化模型
================================

表设计（2 张表）：
  - assistant_conversation: 会话表，按 username 维度存对话元信息
  - assistant_message:      消息表，按 session_id 关联会话，每轮对话拆 2 行（user + assistant）

保留策略：
  - 自动过期：30 天前的会话由 GET /sessions 懒清理（不引入 cron）
  - 数据库为唯一事实来源，后端不维护内存字典 _HISTORY

字段说明参考现有 ChatLog（model_chat.py）的 username 维度设计，不使用 user_id 外键。
"""
from datetime import datetime

from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, Text, DateTime

from myapp.models.base import MyappModelBase


class AssistantConversation(Model, MyappModelBase):
    """AI 助手对话会话"""
    __tablename__ = 'assistant_conversation'

    id = Column(Integer, primary_key=True, comment='id主键')
    username = Column(String(64), nullable=False, index=True, comment='用户名（账号）')
    session_id = Column(String(128), nullable=False, unique=True, index=True, comment='会话 ID')
    title = Column(String(256), nullable=True, default='', comment='对话标题（首条 user 消息前 30 字）')
    created_on = Column(DateTime, nullable=False, default=datetime.now, comment='创建时间')
    changed_on = Column(
        DateTime,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
        comment='最近更新时间',
    )

    def __repr__(self):
        return f'<AssistantConversation {self.session_id}>'


class AssistantMessage(Model, MyappModelBase):
    """AI 助手消息（每轮对话拆 2 行：user + assistant）"""
    __tablename__ = 'assistant_message'

    id = Column(Integer, primary_key=True, comment='id主键')
    session_id = Column(String(128), nullable=False, index=True, comment='关联会话 ID')
    role = Column(String(16), nullable=False, comment='user / assistant')
    content = Column(Text, nullable=True, comment='消息内容')
    created_on = Column(DateTime, nullable=False, default=datetime.now, comment='创建时间')

    def __repr__(self):
        return f'<AssistantMessage {self.session_id}:{self.role}>'
