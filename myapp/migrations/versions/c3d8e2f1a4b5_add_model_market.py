"""add model market tables

Revision ID: c3d8e2f1a4b5
Revises: b4e8a1c2d9f0
Create Date: 2026-06-16 17:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d8e2f1a4b5'
down_revision = 'b4e8a1c2d9f0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'model_market_model',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=64), nullable=True),
        sa.Column('task_type', sa.String(length=64), nullable=True),
        sa.Column('framework', sa.String(length=64), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('cover_url', sa.String(length=1024), nullable=True),
        sa.Column('tags', sa.String(length=1024), nullable=True),
        sa.Column('support_experience', sa.Boolean(), nullable=True),
        sa.Column('support_develop', sa.Boolean(), nullable=True),
        sa.Column('support_finetune', sa.Boolean(), nullable=True),
        sa.Column('support_deploy', sa.Boolean(), nullable=True),
        sa.Column('model_path', sa.String(length=1024), nullable=True),
        sa.Column('default_version', sa.String(length=128), nullable=True),
        sa.Column('python_version', sa.String(length=32), nullable=True),
        sa.Column('cuda_version', sa.String(length=32), nullable=True),
        sa.Column('notebook_image', sa.String(length=1024), nullable=True),
        sa.Column('finetune_image', sa.String(length=1024), nullable=True),
        sa.Column('inference_image', sa.String(length=1024), nullable=True),
        sa.Column('default_cpu', sa.String(length=32), nullable=True),
        sa.Column('default_memory', sa.String(length=32), nullable=True),
        sa.Column('default_gpu', sa.String(length=32), nullable=True),
        sa.Column('default_ports', sa.String(length=64), nullable=True),
        sa.Column('volume_mount', sa.Text(), nullable=True),
        sa.Column('command', sa.Text(), nullable=True),
        sa.Column('env_json', sa.Text(), nullable=True),
        sa.Column('notebook_template', sa.Text(), nullable=True),
        sa.Column('finetune_template', sa.Text(), nullable=True),
        sa.Column('deploy_template', sa.Text(), nullable=True),
        sa.Column('demo_input_type', sa.String(length=64), nullable=True),
        sa.Column('demo_output_type', sa.String(length=64), nullable=True),
        sa.Column('demo_dataset_url', sa.String(length=1024), nullable=True),
        sa.Column('api_schema_json', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=True),
        sa.Column('created_by', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    op.create_table(
        'model_market_action',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.BigInteger(), nullable=False),
        sa.Column('action_type', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=True),
        sa.Column('target_type', sa.String(length=64), nullable=True),
        sa.Column('target_id', sa.BigInteger(), nullable=True),
        sa.Column('target_name', sa.String(length=255), nullable=True),
        sa.Column('target_url', sa.String(length=1024), nullable=True),
        sa.Column('project_id', sa.BigInteger(), nullable=True),
        sa.Column('namespace', sa.String(length=255), nullable=True),
        sa.Column('request_json', sa.Text(), nullable=True),
        sa.Column('response_json', sa.Text(), nullable=True),
        sa.Column('error_msg', sa.Text(), nullable=True),
        sa.Column('created_by', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'model_market_service',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.BigInteger(), nullable=False),
        sa.Column('action_id', sa.BigInteger(), nullable=True),
        sa.Column('service_id', sa.BigInteger(), nullable=False),
        sa.Column('service_name', sa.String(length=255), nullable=True),
        sa.Column('service_status', sa.String(length=64), nullable=True),
        sa.Column('project_id', sa.BigInteger(), nullable=True),
        sa.Column('namespace', sa.String(length=255), nullable=True),
        sa.Column('endpoint', sa.String(length=1024), nullable=True),
        sa.Column('pc_demo_url', sa.String(length=1024), nullable=True),
        sa.Column('mobile_demo_url', sa.String(length=1024), nullable=True),
        sa.Column('api_schema_json', sa.Text(), nullable=True),
        sa.Column('extra_json', sa.Text(), nullable=True),
        sa.Column('created_by', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('service_id')
    )


def downgrade():
    op.drop_table('model_market_service')
    op.drop_table('model_market_action')
    op.drop_table('model_market_model')
