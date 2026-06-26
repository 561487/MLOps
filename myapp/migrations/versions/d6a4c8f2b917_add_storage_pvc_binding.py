"""add storage pvc binding

Revision ID: d6a4c8f2b917
Revises: c3d8e2f1a4b5
Create Date: 2026-06-25 11:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column


# revision identifiers, used by Alembic.
revision = 'd6a4c8f2b917'
down_revision = 'c3d8e2f1a4b5'
branch_labels = None
depends_on = None


def _split_names(value):
    value = value or ''
    for separator in [';', '\n', '\t']:
        value = value.replace(separator, ',')
    return [item.strip() for item in value.split(',') if item.strip()]


def upgrade():
    op.create_table(
        'storage_pvc_binding',
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('storage_id', sa.Integer(), nullable=False, comment='存储资源id'),
        sa.Column('cluster', sa.String(length=100), nullable=False, comment='所属集群'),
        sa.Column('namespace', sa.String(length=200), nullable=False, comment='PVC所在命名空间'),
        sa.Column('pvc_name', sa.String(length=200), nullable=False, comment='K8s PVC名称'),
        sa.Column('pv_name', sa.String(length=500), nullable=True, comment='K8s PV名称'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='Kubernetes PVC状态'),
        sa.Column('storage_class', sa.String(length=200), nullable=True, comment='StorageClass'),
        sa.Column('backend_identity', sa.Text(length=65536), nullable=True, comment='后端标识JSON'),
        sa.Column('backend_match_status', sa.String(length=50), nullable=False, comment='后端一致性状态'),
        sa.Column('backend_match_message', sa.Text(), nullable=True, comment='后端一致性说明'),
        sa.Column('last_checked_at', sa.DateTime(), nullable=True, comment='最后检查时间'),
        sa.ForeignKeyConstraint(['storage_id'], ['storage.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_storage_pvc_binding_storage_namespace',
        'storage_pvc_binding',
        ['storage_id', 'namespace'],
        unique=True,
    )

    bind = op.get_bind()
    storage_rows = bind.execute(sa.text(
        "select id, name, cluster, namespace, pvc_name, pv_name, storage_class, status from storage"
    )).fetchall()

    binding_table = table(
        'storage_pvc_binding',
        column('storage_id', sa.Integer),
        column('cluster', sa.String),
        column('namespace', sa.String),
        column('pvc_name', sa.String),
        column('pv_name', sa.String),
        column('status', sa.String),
        column('storage_class', sa.String),
        column('backend_identity', sa.Text),
        column('backend_match_status', sa.String),
        column('backend_match_message', sa.Text),
        column('last_checked_at', sa.DateTime),
    )

    for row in storage_rows:
        namespaces = _split_names(row.namespace)
        pv_names = _split_names(row.pv_name)
        pvc_name = row.pvc_name or row.name or ''
        for index, namespace in enumerate(namespaces):
            bind.execute(binding_table.insert().values(
                storage_id=row.id,
                cluster=row.cluster or '',
                namespace=namespace,
                pvc_name=pvc_name,
                pv_name=pv_names[index] if index < len(pv_names) else '',
                status='Missing',
                storage_class=row.storage_class or '',
                backend_identity='{}',
                backend_match_status='unknown',
                backend_match_message='',
                last_checked_at=None,
            ))


def downgrade():
    op.drop_index('ix_storage_pvc_binding_storage_namespace', table_name='storage_pvc_binding')
    op.drop_table('storage_pvc_binding')
