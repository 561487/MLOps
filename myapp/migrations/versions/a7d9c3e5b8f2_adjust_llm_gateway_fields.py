"""adjust llm gateway fields

Revision ID: a7d9c3e5b8f2
Revises: 9b3c1d2e4f56
Create Date: 2026-06-09 19:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a7d9c3e5b8f2'
down_revision = '9b3c1d2e4f56'
branch_labels = None
depends_on = None


def _table_columns(table_name):
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column['name'] for column in inspector.get_columns(table_name)}


def _table_indexes(table_name):
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {index['name'] for index in inspector.get_indexes(table_name)}


def upgrade():
    columns = _table_columns('llm_gateway')
    indexes = _table_indexes('llm_gateway')

    if 'api_key' not in columns:
        op.add_column('llm_gateway', sa.Column('api_key', sa.String(length=128), nullable=True, comment='API Key'))
        bind = op.get_bind()
        if bind.dialect.name == 'mysql':
            op.execute("UPDATE llm_gateway SET api_key = CONCAT('sk-', REPLACE(UUID(), '-', '')) WHERE api_key IS NULL OR api_key = ''")
        else:
            op.execute("UPDATE llm_gateway SET api_key = 'sk-' || id WHERE api_key IS NULL OR api_key = ''")
        op.alter_column('llm_gateway', 'api_key', existing_type=sa.String(length=128), nullable=False)

    if 'token_quota' not in columns:
        op.add_column('llm_gateway', sa.Column('token_quota', sa.String(length=50), nullable=False, server_default='no-limit', comment='token额度'))
        op.alter_column('llm_gateway', 'token_quota', existing_type=sa.String(length=50), server_default=None)

    if 'ix_llm_gateway_api_key_hash' in indexes:
        op.drop_index(op.f('ix_llm_gateway_api_key_hash'), table_name='llm_gateway')

    indexes = _table_indexes('llm_gateway')
    if 'ix_llm_gateway_api_key' not in indexes:
        op.create_index(op.f('ix_llm_gateway_api_key'), 'llm_gateway', ['api_key'], unique=True)

    columns = _table_columns('llm_gateway')
    for column_name in ['api_key_hash', 'api_key_prefix', 'retry_count', 'timeout', 'config']:
        if column_name in columns:
            op.drop_column('llm_gateway', column_name)


def downgrade():
    columns = _table_columns('llm_gateway')
    indexes = _table_indexes('llm_gateway')

    if 'api_key_hash' not in columns:
        op.add_column('llm_gateway', sa.Column('api_key_hash', sa.String(length=128), nullable=True, comment='API Key哈希'))
        op.execute("UPDATE llm_gateway SET api_key_hash = api_key WHERE api_key_hash IS NULL")
        op.alter_column('llm_gateway', 'api_key_hash', existing_type=sa.String(length=128), nullable=False)
    if 'api_key_prefix' not in columns:
        op.add_column('llm_gateway', sa.Column('api_key_prefix', sa.String(length=32), nullable=True, comment='API Key前缀'))
        op.execute("UPDATE llm_gateway SET api_key_prefix = SUBSTR(api_key, 1, 10) WHERE api_key_prefix IS NULL")
        op.alter_column('llm_gateway', 'api_key_prefix', existing_type=sa.String(length=32), nullable=False)
    if 'retry_count' not in columns:
        op.add_column('llm_gateway', sa.Column('retry_count', sa.Integer(), nullable=False, server_default='1', comment='重试次数'))
        op.alter_column('llm_gateway', 'retry_count', existing_type=sa.Integer(), server_default=None)
    if 'timeout' not in columns:
        op.add_column('llm_gateway', sa.Column('timeout', sa.Integer(), nullable=False, server_default='60', comment='请求超时秒数'))
        op.alter_column('llm_gateway', 'timeout', existing_type=sa.Integer(), server_default=None)
    if 'config' not in columns:
        op.add_column('llm_gateway', sa.Column('config', sa.Text(length=65536), nullable=True, comment='扩展配置'))

    if 'ix_llm_gateway_api_key' in indexes:
        op.drop_index(op.f('ix_llm_gateway_api_key'), table_name='llm_gateway')
    indexes = _table_indexes('llm_gateway')
    if 'ix_llm_gateway_api_key_hash' not in indexes:
        op.create_index(op.f('ix_llm_gateway_api_key_hash'), 'llm_gateway', ['api_key_hash'], unique=True)

    columns = _table_columns('llm_gateway')
    for column_name in ['api_key', 'token_quota']:
        if column_name in columns:
            op.drop_column('llm_gateway', column_name)
