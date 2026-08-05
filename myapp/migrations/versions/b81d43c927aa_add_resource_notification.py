"""add workflow resource notification settings

Revision ID: b81d43c927aa
Revises: f3a8d1c7e920
"""
from alembic import op
import sqlalchemy as sa
revision='b81d43c927aa'
down_revision='f3a8d1c7e920'
branch_labels=None
depends_on=None

def upgrade():
    for name,default in [('cpu_threshold','90'),('memory_threshold','90'),('gpu_threshold','95'),('gpu_memory_threshold','90'),('gpu_temperature_threshold','85'),('threshold_duration','120')]:
        op.add_column('mlops_notification_channel',sa.Column(name,sa.Integer(),nullable=False,server_default=default))
    op.add_column('mlops_notification_channel',sa.Column('resource_monitor_enabled',sa.Boolean(),nullable=False,server_default=sa.false()))
    op.add_column('mlops_notification_channel',sa.Column('completion_summary',sa.Boolean(),nullable=False,server_default=sa.true()))

def downgrade():
    for name in ('completion_summary','resource_monitor_enabled','threshold_duration','gpu_temperature_threshold','gpu_memory_threshold','gpu_threshold','memory_threshold','cpu_threshold'):
        op.drop_column('mlops_notification_channel',name)