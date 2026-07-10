"""add mlops_training_monitor table

Revision ID: 0727622e2d32
Revises: d6a4c8f2b917
Create Date: 2026-07-09 13:56:46.442854
"""
from alembic import op
import sqlalchemy as sa

revision = "0727622e2d32"
down_revision = "d6a4c8f2b917"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "mlops_training_monitor",
        sa.Column("created_on", sa.DateTime(), nullable=True, comment="创建时间"),
        sa.Column("changed_on", sa.DateTime(), nullable=True, comment="修改时间"),
        sa.Column("id", sa.Integer(), nullable=False, comment="自增主键"),
        sa.Column("pipeline_id", sa.String(64), nullable=True, comment="任务流 ID"),
        sa.Column("pipeline_name", sa.String(256), nullable=True, comment="任务流名称"),
        sa.Column("run_id", sa.String(128), nullable=False, comment="运行实例 run-id"),
        sa.Column("workflow_name", sa.String(256), nullable=True, comment="Argo Workflow 名称"),
        sa.Column("task_id", sa.String(64), nullable=True, comment="Task ID"),
        sa.Column("task_name", sa.String(256), nullable=True, comment="Task 名称"),
        sa.Column("node_name", sa.String(256), nullable=False, comment="DAG 节点名称"),
        sa.Column("pod_name", sa.String(256), nullable=True, comment="K8s Pod 名称"),
        sa.Column("job_template_name", sa.String(256), nullable=True, comment="算子模板名称"),
        sa.Column("monitor_type", sa.String(32), nullable=False, comment="监控类型"),
        sa.Column("monitor_url", sa.Text(), nullable=True, comment="监控平台实验详情页地址"),
        sa.Column("monitor_status", sa.String(32), nullable=False, comment="监控状态"),
        sa.Column("creator", sa.String(100), nullable=True, comment="创建者用户名"),
        sa.Column("created_by_fk", sa.Integer(), nullable=True, comment="创建者 FK"),
        sa.Column("changed_by_fk", sa.Integer(), nullable=True, comment="修改者 FK"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["created_by_fk"], ["ab_user.id"]),
        sa.ForeignKeyConstraint(["changed_by_fk"], ["ab_user.id"]),
    )


def downgrade():
    op.drop_table("mlops_training_monitor")

