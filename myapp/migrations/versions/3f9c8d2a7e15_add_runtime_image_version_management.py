"""add runtime_image_version / model_runtime_mapping tables and job_template.runtime_key

Revision ID: 3f9c8d2a7e15
Revises: b81d43c927aa

"""
from alembic import op
import sqlalchemy as sa

revision = "3f9c8d2a7e15"
down_revision = "b81d43c927aa"
branch_labels = None
depends_on = None


def upgrade():
    # Job_Template 增加 runtime_key（nullable，普通模板不填，行为完全不变）
    op.add_column(
        "job_template",
        sa.Column(
            "runtime_key",
            sa.String(100),
            nullable=True,
            comment="模型Runtime类型，如 msswift/gptqmodel（空=不纳入Runtime版本管理）",
        ),
    )

    # 平台登记的 Runtime 镜像版本
    op.create_table(
        "runtime_image_version",
        sa.Column("created_on", sa.DateTime(), nullable=True, comment="创建时间"),
        sa.Column("changed_on", sa.DateTime(), nullable=True, comment="修改时间"),
        sa.Column("id", sa.Integer(), nullable=False, comment="id主键"),
        sa.Column("runtime_key", sa.String(100), nullable=False, comment="Runtime类型，如 msswift/gptqmodel/vllm/sglang"),
        sa.Column("version", sa.String(100), nullable=False, comment="Runtime版本，如 4.0-r1"),
        sa.Column("image", sa.String(500), nullable=False, comment="Harbor完整镜像地址"),
        sa.Column("enabled", sa.Boolean(), nullable=False, comment="是否允许新任务使用（false仅停用，不删除）"),
        sa.Column("remark", sa.String(500), nullable=True, comment="备注"),
        sa.Column("created_by_fk", sa.Integer(), nullable=True, comment="创建者 FK"),
        sa.Column("changed_by_fk", sa.Integer(), nullable=True, comment="修改者 FK"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image", name="uq_runtime_image"),
        sa.UniqueConstraint("runtime_key", "version", name="uq_runtime_key_version"),
        sa.ForeignKeyConstraint(["created_by_fk"], ["ab_user.id"]),
        sa.ForeignKeyConstraint(["changed_by_fk"], ["ab_user.id"]),
    )
    op.create_index("ix_runtime_image_version_runtime_key", "runtime_image_version", ["runtime_key"])

    # 模型与 Runtime 的已验证兼容映射
    op.create_table(
        "model_runtime_mapping",
        sa.Column("created_on", sa.DateTime(), nullable=True, comment="创建时间"),
        sa.Column("changed_on", sa.DateTime(), nullable=True, comment="修改时间"),
        sa.Column("id", sa.Integer(), nullable=False, comment="id主键"),
        sa.Column("model", sa.String(500), nullable=False, comment="与--model参数匹配的模型标识（精确匹配）"),
        sa.Column("scene", sa.String(50), nullable=False, comment="场景：finetune/quantization/inference"),
        sa.Column("runtime_key", sa.String(100), nullable=False, comment="Runtime类型，如 msswift"),
        sa.Column("runtime_version_id", sa.Integer(), nullable=False, comment="Runtime镜像版本id"),
        sa.Column("is_default", sa.Boolean(), nullable=False, comment="是否该模型当前默认Runtime"),
        sa.Column("remark", sa.String(500), nullable=True, comment="验证说明"),
        sa.Column("created_by_fk", sa.Integer(), nullable=True, comment="创建者 FK"),
        sa.Column("changed_by_fk", sa.Integer(), nullable=True, comment="修改者 FK"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["runtime_version_id"], ["runtime_image_version.id"]),
        sa.ForeignKeyConstraint(["created_by_fk"], ["ab_user.id"]),
        sa.ForeignKeyConstraint(["changed_by_fk"], ["ab_user.id"]),
    )
    op.create_index("ix_model_runtime_mapping_model", "model_runtime_mapping", ["model"])

    # Task 保存任务创建时实际解析得到的镜像（历史任务重跑不重新解析）
    op.add_column(
        "task",
        sa.Column(
            "runtime_image",
            sa.String(500),
            nullable=True,
            comment="任务创建时解析得到的最终镜像（Runtime版本管理）",
        ),
    )


def downgrade():
    op.drop_column("task", "runtime_image")
    op.drop_index("ix_model_runtime_mapping_model", table_name="model_runtime_mapping")
    op.drop_table("model_runtime_mapping")
    op.drop_index("ix_runtime_image_version_runtime_key", table_name="runtime_image_version")
    op.drop_table("runtime_image_version")
    op.drop_column("job_template", "runtime_key")
