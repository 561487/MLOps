"""add experiment_run_id to mlops_training_monitor

Revision ID: 2dac8971d60e
Revises: 0727622e2d32
Create Date: 2026-07-17 17:25:29.146463

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2dac8971d60e'
down_revision = '0727622e2d32'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('mlops_training_monitor', schema=None) as batch_op:
        batch_op.add_column(sa.Column('experiment_run_id', sa.String(length=256), nullable=True,
                                       comment='SwanLab run 目录名，例如 run-20260713_175627-4wm7y5d2'))


def downgrade():
    with op.batch_alter_table('mlops_training_monitor', schema=None) as batch_op:
        batch_op.drop_column('experiment_run_id')
