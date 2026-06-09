"""add llm gateway

Revision ID: 9b3c1d2e4f56
Revises: 7c2a9e8d4f10
Create Date: 2026-06-09 17:50:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9b3c1d2e4f56'
down_revision = '7c2a9e8d4f10'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'llm_gateway',
        sa.Column('created_on', sa.DateTime(), nullable=True),
        sa.Column('changed_on', sa.DateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='英文名'),
        sa.Column('label', sa.String(length=100), nullable=True, comment='显示名'),
        sa.Column('model_name', sa.String(length=200), nullable=False, comment='对外模型名'),
        sa.Column('project_id', sa.Integer(), nullable=False, comment='项目组id'),
        sa.Column('service_type', sa.String(length=50), nullable=False, comment='后端服务类型'),
        sa.Column('service_id', sa.Integer(), nullable=False, comment='推理服务id'),
        sa.Column('api_key', sa.String(length=128), nullable=False, comment='API Key'),
        sa.Column('token_quota', sa.String(length=50), nullable=False, comment='token额度'),
        sa.Column('expire_at', sa.DateTime(), nullable=True, comment='过期时间'),
        sa.Column('status', sa.String(length=50), nullable=False, comment='状态'),
        sa.Column('created_by_fk', sa.Integer(), nullable=True),
        sa.Column('changed_by_fk', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']),
        sa.ForeignKeyConstraint(['project_id'], ['project.id']),
        sa.ForeignKeyConstraint(['service_id'], ['inferenceservice.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_llm_gateway_api_key'), 'llm_gateway', ['api_key'], unique=True)
    op.create_index(op.f('ix_llm_gateway_model_name'), 'llm_gateway', ['model_name'], unique=False)

    op.create_table(
        'llm_gateway_log',
        sa.Column('id', sa.Integer(), nullable=False, comment='id主键'),
        sa.Column('gateway_id', sa.Integer(), nullable=True, comment='服务网关id'),
        sa.Column('request_id', sa.String(length=100), nullable=True, comment='请求id'),
        sa.Column('model_name', sa.String(length=200), nullable=True, comment='模型名'),
        sa.Column('api_key_prefix', sa.String(length=32), nullable=True, comment='API Key前缀'),
        sa.Column('client_ip', sa.String(length=100), nullable=True, comment='客户端IP'),
        sa.Column('path', sa.String(length=500), nullable=True, comment='请求路径'),
        sa.Column('method', sa.String(length=20), nullable=True, comment='请求方法'),
        sa.Column('status_code', sa.Integer(), nullable=True, comment='状态码'),
        sa.Column('success', sa.Boolean(), nullable=False, comment='是否成功'),
        sa.Column('latency_ms', sa.Integer(), nullable=True, comment='耗时毫秒'),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True, comment='输入token'),
        sa.Column('completion_tokens', sa.Integer(), nullable=True, comment='输出token'),
        sa.Column('total_tokens', sa.Integer(), nullable=True, comment='总token'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='错误信息'),
        sa.Column('created_on', sa.DateTime(), nullable=True, comment='创建时间'),
        sa.ForeignKeyConstraint(['gateway_id'], ['llm_gateway.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_llm_gateway_log_created_on'), 'llm_gateway_log', ['created_on'], unique=False)
    op.create_index(op.f('ix_llm_gateway_log_gateway_id'), 'llm_gateway_log', ['gateway_id'], unique=False)
    op.create_index(op.f('ix_llm_gateway_log_model_name'), 'llm_gateway_log', ['model_name'], unique=False)
    op.create_index(op.f('ix_llm_gateway_log_request_id'), 'llm_gateway_log', ['request_id'], unique=False)
    op.create_index(op.f('ix_llm_gateway_log_success'), 'llm_gateway_log', ['success'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_llm_gateway_log_success'), table_name='llm_gateway_log')
    op.drop_index(op.f('ix_llm_gateway_log_request_id'), table_name='llm_gateway_log')
    op.drop_index(op.f('ix_llm_gateway_log_model_name'), table_name='llm_gateway_log')
    op.drop_index(op.f('ix_llm_gateway_log_gateway_id'), table_name='llm_gateway_log')
    op.drop_index(op.f('ix_llm_gateway_log_created_on'), table_name='llm_gateway_log')
    op.drop_table('llm_gateway_log')
    op.drop_index(op.f('ix_llm_gateway_model_name'), table_name='llm_gateway')
    op.drop_index(op.f('ix_llm_gateway_api_key'), table_name='llm_gateway')
    op.drop_table('llm_gateway')
