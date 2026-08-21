"""Runtime 版本合并入「镜像管理」Images 表（第三版优化）

背景：RuntimeImageVersion 与 Images 业务语义 1:1（一个镜像只属于一个 Runtime 框架），
真实库两张表（runtime_image_version / model_runtime_mapping）均为空表，合并零数据风险。

结构变化：
- images 新增 runtime_key / runtime_version / runtime_enabled，唯一约束 uq_runtime_key_version
  （MySQL 唯一约束对 NULL 不冲突，普通镜像 runtime_key=NULL 不受影响）
- model_runtime_mapping 由 runtime_version_id → runtime_image_version 改为 images_id → images
  （空表，直接结构改造；非空环境回填逻辑保留，mapping 不再经过中间层）
- runtime_image_version 表保留（不 DROP），仅停止注册 API 与菜单入口，兼容历史

安全顺序（公共 MySQL 库）：
1. images 加字段 → 2. runtime_image_version 数据回填 → 3. mapping 结构转换 →
4. 删旧 FK → 5. runtime_image_version 保留不动

Revision ID: 6a2b3c4d5e6f
Revises: 5a1b2c3d4e5f
Create Date: 2026-08-20
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '6a2b3c4d5e6f'
down_revision = '5a1b2c3d4e5f'
branch_labels = None
depends_on = None


def _has_column(inspector, table, column):
    return column in [c['name'] for c in inspector.get_columns(table)]


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # ---- 1. images 增加 Runtime 元数据字段 ----
    if not _has_column(inspector, 'images', 'runtime_key'):
        op.add_column('images', sa.Column('runtime_key', sa.String(100), nullable=True,
                                          comment='模型Runtime类型，如 msswift/gptqmodel（空=普通镜像，不纳入Runtime版本管理）'))
    if not _has_column(inspector, 'images', 'runtime_version'):
        op.add_column('images', sa.Column('runtime_version', sa.String(100), nullable=True,
                                          comment='Runtime版本，如 3.12.5-r1（普通镜像为空）'))
    if not _has_column(inspector, 'images', 'runtime_enabled'):
        op.add_column('images', sa.Column('runtime_enabled', sa.Boolean(), nullable=False,
                                          server_default=sa.text('1'),
                                          comment='是否允许新任务使用（false仅停用，镜像保留）'))
    inspector = sa.inspect(bind)
    idxs = [i['name'] for i in inspector.get_indexes('images')]
    if 'ix_images_runtime_key' not in idxs:
        op.create_index('ix_images_runtime_key', 'images', ['runtime_key'])

    # 唯一约束：同一 Runtime 类型同一版本只允许一个镜像。
    # 当前 images 表全部 runtime_key=NULL，MySQL 唯一约束对 NULL 不冲突，普通镜像不受影响。
    inspector = sa.inspect(bind)
    if not any(u['name'] == 'uq_runtime_key_version' for u in inspector.get_unique_constraints('images')):
        op.create_unique_constraint('uq_runtime_key_version', 'images', ['runtime_key', 'runtime_version'])

    # ---- 2. runtime_image_version → images 数据回填（当前空表无效果，保留通用逻辑） ----
    # 2a. 按 images_id 关联回填（第二版新格式）
    bind.execute(sa.text(
        "UPDATE images i JOIN runtime_image_version r ON r.images_id = i.id "
        "SET i.runtime_key = r.runtime_key, i.runtime_version = r.version, "
        "    i.runtime_enabled = r.enabled "
        "WHERE i.runtime_key IS NULL"
    ))
    # 2b. 按 image = images.name 匹配回填（第一版旧格式，无 images_id 的记录）
    bind.execute(sa.text(
        "UPDATE images i JOIN runtime_image_version r ON r.image = i.name "
        "SET i.runtime_key = r.runtime_key, i.runtime_version = r.version, "
        "    i.runtime_enabled = r.enabled, r.images_id = i.id "
        "WHERE i.runtime_key IS NULL"
    ))

    # ---- 3. model_runtime_mapping 结构转换：runtime_version_id → images_id ----
    inspector = sa.inspect(bind)
    if not _has_column(inspector, 'model_runtime_mapping', 'images_id'):
        op.add_column('model_runtime_mapping', sa.Column('images_id', sa.Integer(), nullable=True,
                                                         comment='Runtime镜像id（镜像管理Images表）'))
        # 3a. 数据转换（有数据时）：mapping → runtime_image_version → images
        bind.execute(sa.text(
            "UPDATE model_runtime_mapping m JOIN runtime_image_version r ON m.runtime_version_id = r.id "
            "SET m.images_id = r.images_id "
            "WHERE m.images_id IS NULL AND r.images_id IS NOT NULL"
        ))
    # 3b. 删除旧 FK 与旧列（空表直接成功；有数据时先转换后删除）
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys('model_runtime_mapping'):
        if fk.get('constrained_columns') == ['runtime_version_id']:
            with op.batch_alter_table('model_runtime_mapping') as batch_op:
                batch_op.drop_constraint(fk.get('name'), type_='foreignkey')
            break
    inspector = sa.inspect(bind)
    if _has_column(inspector, 'model_runtime_mapping', 'runtime_version_id'):
        op.drop_column('model_runtime_mapping', 'runtime_version_id')
    # 3c. images_id 非空 + 外键
    with op.batch_alter_table('model_runtime_mapping') as batch_op:
        batch_op.alter_column('images_id', existing_type=sa.Integer(), nullable=False)
    inspector = sa.inspect(bind)
    if not any(fk.get('constrained_columns') == ['images_id'] for fk in inspector.get_foreign_keys('model_runtime_mapping')):
        with op.batch_alter_table('model_runtime_mapping') as batch_op:
            batch_op.create_foreign_key('fk_model_runtime_mapping_images_id', 'images',
                                        ['images_id'], ['id'])

    # ---- 4. runtime_image_version 表保留（不 DROP），仅停止 API/菜单注册 ----
    print('[runtime_merge_into_images] images runtime 字段/mapping images_id 改造完成；'
          'runtime_image_version 表保留（历史兼容，不再使用）')


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # ---- mapping 回退为 runtime_version_id（有数据时按 runtime_image_version 已有记录关联） ----
    if not _has_column(inspector, 'model_runtime_mapping', 'runtime_version_id'):
        op.add_column('model_runtime_mapping', sa.Column('runtime_version_id', sa.Integer(), nullable=True,
                                                         comment='Runtime镜像版本id'))
        bind.execute(sa.text(
            "UPDATE model_runtime_mapping m JOIN runtime_image_version r "
            "ON r.images_id = m.images_id AND r.runtime_key = (SELECT i.runtime_key FROM images i WHERE i.id = m.images_id) "
            "SET m.runtime_version_id = r.id "
            "WHERE m.runtime_version_id IS NULL AND m.images_id IS NOT NULL"
        ))
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys('model_runtime_mapping'):
        if fk.get('constrained_columns') == ['images_id']:
            with op.batch_alter_table('model_runtime_mapping') as batch_op:
                batch_op.drop_constraint(fk.get('name'), type_='foreignkey')
            break
    inspector = sa.inspect(bind)
    if _has_column(inspector, 'model_runtime_mapping', 'images_id'):
        op.drop_column('model_runtime_mapping', 'images_id')
    with op.batch_alter_table('model_runtime_mapping') as batch_op:
        batch_op.alter_column('runtime_version_id', existing_type=sa.Integer(), nullable=False)
    inspector = sa.inspect(bind)
    if not any(fk.get('constrained_columns') == ['runtime_version_id'] for fk in inspector.get_foreign_keys('model_runtime_mapping')):
        with op.batch_alter_table('model_runtime_mapping') as batch_op:
            batch_op.create_foreign_key('fk_model_runtime_mapping_runtime_version', 'runtime_image_version',
                                        ['runtime_version_id'], ['id'])

    # ---- images Runtime 元数据回填到 runtime_image_version 后删除列（保留数据） ----
    bind.execute(sa.text(
        "INSERT IGNORE INTO runtime_image_version (runtime_key, version, images_id, enabled, image, created_on, changed_on) "
        "SELECT runtime_key, runtime_version, id, runtime_enabled, name, NOW(), NOW() "
        "FROM images WHERE runtime_key IS NOT NULL AND runtime_key <> ''"
    ))
    inspector = sa.inspect(bind)
    if any(u['name'] == 'uq_runtime_key_version' for u in inspector.get_unique_constraints('images')):
        with op.batch_alter_table('images') as batch_op:
            batch_op.drop_constraint('uq_runtime_key_version', type_='unique')
    inspector = sa.inspect(bind)
    if 'ix_images_runtime_key' in [i['name'] for i in inspector.get_indexes('images')]:
        op.drop_index('ix_images_runtime_key', table_name='images')
    for col in ('runtime_enabled', 'runtime_version', 'runtime_key'):
        inspector = sa.inspect(bind)
        if _has_column(inspector, 'images', col):
            op.drop_column('images', col)
    print('[runtime_merge_into_images] downgrade: mapping/images 已回退（Runtime 数据已回填 runtime_image_version）')
