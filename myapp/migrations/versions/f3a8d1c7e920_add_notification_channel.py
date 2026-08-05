"""add notification channel and delivery tables

Revision ID: f3a8d1c7e920
Revises: 2dac8971d60e
"""
from alembic import op
import sqlalchemy as sa

revision = 'f3a8d1c7e920'
down_revision = '2dac8971d60e'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('mlops_notification_channel',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('channel_type', sa.String(32), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('webhook_encrypted', sa.Text()), sa.Column('secret_encrypted', sa.Text()),
        sa.Column('project_scope', sa.Text()), sa.Column('cluster_scope', sa.Text()),
        sa.Column('namespace_scope', sa.Text()), sa.Column('event_types', sa.Text(), nullable=False),
        sa.Column('pending_timeout', sa.Integer(), nullable=False, server_default='300'),
        sa.Column('silence_seconds', sa.Integer(), nullable=False, server_default='1800'),
        sa.Column('created_by', sa.String(100)), sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('channel_type', name='uq_notification_channel_type'))
    op.create_table('mlops_notification_delivery',
        sa.Column('id', sa.Integer(), primary_key=True), sa.Column('channel_id', sa.Integer()),
        sa.Column('event_type', sa.String(100), nullable=False), sa.Column('dedup_key', sa.String(512)),
        sa.Column('status', sa.String(32), nullable=False), sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('response_code', sa.Integer()), sa.Column('message_summary', sa.String(500)),
        sa.Column('error', sa.Text()), sa.Column('sent_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), nullable=False))
    op.create_index('ix_notification_delivery_dedup_key', 'mlops_notification_delivery', ['dedup_key'])


def downgrade():
    op.drop_index('ix_notification_delivery_dedup_key', table_name='mlops_notification_delivery')
    op.drop_table('mlops_notification_delivery')
    op.drop_table('mlops_notification_channel')
