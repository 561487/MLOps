"""add etl task instance

Revision ID: b4e8a1c2d9f0
Revises: a7d9c3e5b8f2
Create Date: 2026-06-09 21:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b4e8a1c2d9f0'
down_revision = 'a7d9c3e5b8f2'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'etl_task_instance',
        sa.Column('created_on', sa.DateTime(), nullable=True),
        sa.Column('changed_on', sa.DateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('run_id', sa.String(length=100), nullable=False, comment='运行id'),
        sa.Column('etl_pipeline_id', sa.Integer(), nullable=False, comment='任务流id'),
        sa.Column('etl_task_id', sa.Integer(), nullable=True, comment='任务id'),
        sa.Column('workflow', sa.String(length=100), nullable=False, comment='调度引擎'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='状态'),
        sa.Column('scheduler_url', sa.String(length=500), nullable=True, comment='调度实例地址'),
        sa.Column('log_url', sa.String(length=500), nullable=True, comment='日志地址'),
        sa.Column('expand', sa.Text(), nullable=True, comment='扩展参数'),
        sa.Column('created_by_fk', sa.Integer(), nullable=True),
        sa.Column('changed_by_fk', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['etl_pipeline_id'], ['etl_pipeline.id']),
        sa.ForeignKeyConstraint(['etl_task_id'], ['etl_task.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_etl_task_instance_run_id'), 'etl_task_instance', ['run_id'], unique=False)
    op.create_index(op.f('ix_etl_task_instance_etl_pipeline_id'), 'etl_task_instance', ['etl_pipeline_id'], unique=False)
    op.create_index(op.f('ix_etl_task_instance_etl_task_id'), 'etl_task_instance', ['etl_task_id'], unique=False)
    op.create_index(op.f('ix_etl_task_instance_status'), 'etl_task_instance', ['status'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_etl_task_instance_status'), table_name='etl_task_instance')
    op.drop_index(op.f('ix_etl_task_instance_etl_task_id'), table_name='etl_task_instance')
    op.drop_index(op.f('ix_etl_task_instance_etl_pipeline_id'), table_name='etl_task_instance')
    op.drop_index(op.f('ix_etl_task_instance_run_id'), table_name='etl_task_instance')
    op.drop_table('etl_task_instance')
