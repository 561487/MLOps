"""add storage model

Revision ID: 2f1b6a7c9d01
Revises: 593366be4eff
Create Date: 2026-05-28 14:54:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2f1b6a7c9d01'
down_revision = '593366be4eff'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'storage',
        sa.Column('created_on', sa.DateTime(), nullable=True),
        sa.Column('changed_on', sa.DateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('project_id', sa.Integer(), nullable=False, comment='项目组id'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='英文名，用于生成k8s pvc名称'),
        sa.Column('label', sa.String(length=200), nullable=True, comment='显示名称'),
        sa.Column('storage_type', sa.String(length=50), nullable=False, comment='存储类型，nfs/minio_juicefs'),
        sa.Column('cluster', sa.String(length=100), nullable=False, comment='所属集群'),
        sa.Column('namespace', sa.String(length=200), nullable=False, comment='PVC所在命名空间，多个用逗号分隔'),
        sa.Column('mount_path', sa.String(length=500), nullable=True, comment='推荐容器挂载路径'),
        sa.Column('capacity', sa.String(length=50), nullable=False, comment='存储容量'),
        sa.Column('access_modes', sa.String(length=200), nullable=False, comment='访问模式，多个用逗号分隔'),
        sa.Column('storage_class', sa.String(length=200), nullable=True, comment='StorageClass，静态NFS PV默认留空'),
        sa.Column('pv_name', sa.String(length=500), nullable=True, comment='K8s PV名称，多个用逗号分隔'),
        sa.Column('pvc_name', sa.String(length=200), nullable=True, comment='K8s PVC名称'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='状态'),
        sa.Column('config', sa.Text(length=65536), nullable=True, comment='非敏感配置JSON'),
        sa.Column('remark', sa.Text(), nullable=True, comment='备注'),
        sa.Column('expand', sa.Text(length=65536), nullable=True, comment='扩展参数'),
        sa.Column('created_by_fk', sa.Integer(), nullable=True),
        sa.Column('changed_by_fk', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['project_id'], ['project.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )


def downgrade():
    op.drop_table('storage')
