"""add assistant conversation/message tables for floating AI assistant history

Revision ID: 7b3c4d5e6f7a
Revises: 6a2b3c4d5e6f
Create Date: 2026-08-26

新增 2 张表（V4 对话历史持久化）：
  - assistant_conversation: 按 username 维度存的会话元信息（标题、时间戳）
  - assistant_message:      按 session_id 关联会话的消息（user / assistant 各 1 行）

设计说明：
  - 不加 user_id 外键，跟现有 ChatLog 一致，直接用 username 字符串
  - 保留策略：30 天前的会话由 GET /sessions 懒清理（不引入 cron）
  - session_id 由前端生成（assistant_<user>_<uuid8>），DB 不主动生成
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7b3c4d5e6f7a'
down_revision = '6a2b3c4d5e6f'
branch_labels = None
depends_on = None


def upgrade():
    # 1. 会话表
    op.create_table(
        'assistant_conversation',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, comment='id主键'),
        sa.Column('username', sa.String(length=64), nullable=False, comment='用户名（账号）'),
        sa.Column('session_id', sa.String(length=128), nullable=False, comment='会话 ID'),
        sa.Column('title', sa.String(length=256), nullable=True, server_default='', comment='对话标题'),
        sa.Column('created_on', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('changed_on', sa.DateTime(), nullable=False, comment='最近更新时间'),
        sa.UniqueConstraint('session_id', name='uq_assistant_conversation_session_id'),
        comment='AI 浮窗助手 — 会话表',
    )
    op.create_index('idx_assistant_conv_username', 'assistant_conversation', ['username'])
    op.create_index('idx_assistant_conv_session_id', 'assistant_conversation', ['session_id'])

    # 2. 消息表
    op.create_table(
        'assistant_message',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, comment='id主键'),
        sa.Column('session_id', sa.String(length=128), nullable=False, comment='关联会话 ID'),
        sa.Column('role', sa.String(length=16), nullable=False, comment='user / assistant'),
        sa.Column('content', sa.Text(), nullable=True, comment='消息内容'),
        sa.Column('created_on', sa.DateTime(), nullable=False, comment='创建时间'),
        comment='AI 浮窗助手 — 消息表',
    )
    op.create_index('idx_assistant_msg_session_id', 'assistant_message', ['session_id'])


def downgrade():
    op.drop_index('idx_assistant_msg_session_id', table_name='assistant_message')
    op.drop_table('assistant_message')
    op.drop_index('idx_assistant_conv_session_id', table_name='assistant_conversation')
    op.drop_index('idx_assistant_conv_username', table_name='assistant_conversation')
    op.drop_table('assistant_conversation')
