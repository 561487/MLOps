"""add billing models

Revision ID: 7c2a9e8d4f10
Revises: 2f1b6a7c9d01
Create Date: 2026-06-08 13:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7c2a9e8d4f10'
down_revision = '2f1b6a7c9d01'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'pod_charge_record',
        sa.Column('created_on', sa.DateTime(), nullable=True),
        sa.Column('changed_on', sa.DateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('username', sa.String(length=100), nullable=False, comment='用户名'),
        sa.Column('project', sa.String(length=200), nullable=True, comment='项目组'),
        sa.Column('cluster', sa.String(length=100), nullable=False, comment='集群'),
        sa.Column('resource_group', sa.String(length=100), nullable=True, comment='资源组'),
        sa.Column('namespace', sa.String(length=200), nullable=False, comment='命名空间'),
        sa.Column('pod_type', sa.String(length=100), nullable=True, comment='Pod类型'),
        sa.Column('node', sa.String(length=200), nullable=True, comment='机器'),
        sa.Column('pod_name', sa.String(length=300), nullable=False, comment='Pod名称'),
        sa.Column('cpu', sa.Float(), nullable=False, comment='CPU核数'),
        sa.Column('memory', sa.Float(), nullable=False, comment='内存GB'),
        sa.Column('gpu', sa.Float(), nullable=False, comment='GPU卡数'),
        sa.Column('vgpu', sa.Float(), nullable=False, comment='VGPU卡数'),
        sa.Column('start_time', sa.DateTime(), nullable=False, comment='开始时间'),
        sa.Column('end_time', sa.DateTime(), nullable=True, comment='截止时间'),
        sa.Column('duration_hours', sa.Float(), nullable=False, comment='耗时小时'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='状态'),
        sa.Column('price', sa.Float(), nullable=False, comment='价格'),
        sa.Column('labels', sa.Text(length=65536), nullable=True, comment='Labels'),
        sa.Column('annotations', sa.Text(length=65536), nullable=True, comment='Annotations'),
        sa.Column('events', sa.Text(length=65536), nullable=True, comment='Events'),
        sa.Column('raw_pod', sa.Text(length=16777216), nullable=True, comment='原始Pod信息'),
        sa.Column('created_by_fk', sa.Integer(), nullable=True),
        sa.Column('changed_by_fk', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cluster', 'namespace', 'pod_name', 'start_time', name='uq_pod_charge_identity')
    )
    op.create_index(op.f('ix_pod_charge_record_cluster'), 'pod_charge_record', ['cluster'], unique=False)
    op.create_index(op.f('ix_pod_charge_record_end_time'), 'pod_charge_record', ['end_time'], unique=False)
    op.create_index(op.f('ix_pod_charge_record_namespace'), 'pod_charge_record', ['namespace'], unique=False)
    op.create_index(op.f('ix_pod_charge_record_pod_name'), 'pod_charge_record', ['pod_name'], unique=False)
    op.create_index(op.f('ix_pod_charge_record_start_time'), 'pod_charge_record', ['start_time'], unique=False)
    op.create_index(op.f('ix_pod_charge_record_username'), 'pod_charge_record', ['username'], unique=False)

    op.create_table(
        'bill_record',
        sa.Column('created_on', sa.DateTime(), nullable=True),
        sa.Column('changed_on', sa.DateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('bill_type', sa.String(length=50), nullable=False, comment='账单类型'),
        sa.Column('bill_date', sa.Date(), nullable=False, comment='账单日期'),
        sa.Column('bill_id', sa.String(length=200), nullable=False, comment='账单ID'),
        sa.Column('amount', sa.Float(), nullable=False, comment='金额'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='状态'),
        sa.Column('discount_price', sa.Float(), nullable=False, comment='优惠价格'),
        sa.Column('balance_pay', sa.Float(), nullable=False, comment='余额支付'),
        sa.Column('username', sa.String(length=100), nullable=False, comment='用户名'),
        sa.Column('created_by_fk', sa.Integer(), nullable=True),
        sa.Column('changed_by_fk', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('bill_id')
    )
    op.create_index(op.f('ix_bill_record_bill_date'), 'bill_record', ['bill_date'], unique=False)
    op.create_index(op.f('ix_bill_record_bill_id'), 'bill_record', ['bill_id'], unique=True)
    op.create_index(op.f('ix_bill_record_username'), 'bill_record', ['username'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_bill_record_username'), table_name='bill_record')
    op.drop_index(op.f('ix_bill_record_bill_id'), table_name='bill_record')
    op.drop_index(op.f('ix_bill_record_bill_date'), table_name='bill_record')
    op.drop_table('bill_record')
    op.drop_index(op.f('ix_pod_charge_record_username'), table_name='pod_charge_record')
    op.drop_index(op.f('ix_pod_charge_record_start_time'), table_name='pod_charge_record')
    op.drop_index(op.f('ix_pod_charge_record_pod_name'), table_name='pod_charge_record')
    op.drop_index(op.f('ix_pod_charge_record_namespace'), table_name='pod_charge_record')
    op.drop_index(op.f('ix_pod_charge_record_end_time'), table_name='pod_charge_record')
    op.drop_index(op.f('ix_pod_charge_record_cluster'), table_name='pod_charge_record')
    op.drop_table('pod_charge_record')
