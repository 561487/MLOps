"""Runtime 镜像版本关联 Images 镜像资产（第二版优化）

- runtime_image_version 新增 images_id 外键 → images.id，镜像管理（Images 表）成为镜像资产的唯一来源
- image 字段降级为兼容字段（nullable），新数据由 images 关联自动同步，后续可删除
- 移除 uq_runtime_image 唯一约束（image 为兼容字段后不再强制唯一；同一镜像可服务于不同 Runtime）
- 兼容回填：已有 image 字符串与 images.name 完全一致的记录自动填充 images_id（匹配不到的不动，不猜测）

Revision ID: 5a1b2c3d4e5f
Revises: 3f9c8d2a7e15
Create Date: 2026-08-19
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '5a1b2c3d4e5f'
down_revision = '3f9c8d2a7e15'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # 1. 新增 images_id 列（可空，兼容历史数据）
    if 'images_id' not in [c['name'] for c in inspector.get_columns('runtime_image_version')]:
        op.add_column('runtime_image_version', sa.Column('images_id', sa.Integer(), nullable=True, comment='镜像id（镜像管理Images表，镜像资产唯一来源）'))
    with op.batch_alter_table('runtime_image_version') as batch_op:
        batch_op.create_index('ix_runtime_image_version_images_id', ['images_id'])

    # 2. image 降级为兼容字段（nullable）
    with op.batch_alter_table('runtime_image_version') as batch_op:
        batch_op.alter_column('image', existing_type=sa.String(500), nullable=True,
                              existing_comment='Harbor完整镜像地址')

    # 3. 移除 uq_runtime_image 唯一约束（image 为兼容字段后不再强制唯一）
    inspector = sa.inspect(bind)
    for c in inspector.get_unique_constraints('runtime_image_version'):
        if c['name'] == 'uq_runtime_image':
            with op.batch_alter_table('runtime_image_version') as batch_op:
                batch_op.drop_constraint('uq_runtime_image', type_='unique')

    # 4. 添加外键 images_id → images.id（表为空时直接创建；有旧数据时回填后创建）
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    fks = inspector.get_foreign_keys('runtime_image_version')
    if not any(fk.get('constrained_columns') == ['images_id'] for fk in fks):
        with op.batch_alter_table('runtime_image_version') as batch_op:
            batch_op.create_foreign_key('fk_runtime_image_version_images_id', 'images',
                                        ['images_id'], ['id'])

    # 5. 兼容回填：image 与 images.name 完全一致的记录 → 填充 images_id（匹配不到不猜不动）
    bind = op.get_bind()
    result = bind.execute(sa.text(
        "UPDATE runtime_image_version SET images_id = ("
        "  SELECT i.id FROM images i WHERE i.name = runtime_image_version.image"
        ") WHERE images_id IS NULL AND image IS NOT NULL AND image <> '' AND EXISTS ("
        "  SELECT 1 FROM images i WHERE i.name = runtime_image_version.image"
        ")"
    ))
    print('[runtime_version_link_images] 兼容回填 images_id 记录数: %s' % (result.rowcount if result else 0))

    # 6. P0：同步平台已配置 runtime_key 的 LLM 模板到数据库。
    #    init-job-template.json 已配置 runtime_key，但开发库中历史模板未经重新导入而为空；
    #    此处仅对明确模板名且当前为空的情况赋值（幂等，不覆盖管理员后续手工修改）。
    runtime_key_map = {
        'msswift': 'msswift',
        'llama-factory': 'llama_factory',
        'deepspeed': 'deepspeed',
        'model-quantize': 'gptqmodel',
    }
    for tpl_name, rk in runtime_key_map.items():
        result = bind.execute(sa.text(
            "UPDATE job_template SET runtime_key = :rk "
            "WHERE name = :name AND (runtime_key IS NULL OR runtime_key = '')"
        ), {'rk': rk, 'name': tpl_name})
        if result.rowcount:
            print('[runtime_version_link_images] 模板 %s 设置 runtime_key=%s' % (tpl_name, rk))


def downgrade():
    bind = op.get_bind()

    # 回退结构前：image 为空但 images 关联存在的记录，用 images.name 无损回填，
    # 否则恢复 NOT NULL 约束会失败（无法回填的记录保持原样，ALTER 报错提示人工处理）
    bind.execute(sa.text(
        "UPDATE runtime_image_version SET image = ("
        "  SELECT i.name FROM images i WHERE i.id = runtime_image_version.images_id"
        ") WHERE (image IS NULL OR image = '') AND images_id IS NOT NULL AND EXISTS ("
        "  SELECT 1 FROM images i WHERE i.id = runtime_image_version.images_id"
        ")"
    ))
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys('runtime_image_version'):
        if fk.get('constrained_columns') == ['images_id']:
            with op.batch_alter_table('runtime_image_version') as batch_op:
                batch_op.drop_constraint(fk.get('name'), type_='foreignkey')
            break

    with op.batch_alter_table('runtime_image_version') as batch_op:
        batch_op.drop_index('ix_runtime_image_version_images_id')

    inspector = sa.inspect(bind)
    if 'images_id' in [c['name'] for c in inspector.get_columns('runtime_image_version')]:
        op.drop_column('runtime_image_version', 'images_id')

    # 恢复 image 非空约束
    with op.batch_alter_table('runtime_image_version') as batch_op:
        batch_op.alter_column('image', existing_type=sa.String(500), nullable=False)

    # 恢复 uq_runtime_image 唯一约束（若不存在同名约束才创建）
    # 新格式允许同一镜像服务多个 Runtime 版本，数据存在重复 image 时约束无法恢复：
    # 跳过并告警（不删除数据，由人工评估后决定），避免回滚被数据阻塞。
    inspector = sa.inspect(bind)
    if not any(c['name'] == 'uq_runtime_image' for c in inspector.get_unique_constraints('runtime_image_version')):
        dup = bind.execute(sa.text(
            "SELECT image, COUNT(*) FROM runtime_image_version "
            "WHERE image IS NOT NULL AND image <> '' GROUP BY image HAVING COUNT(*) > 1 LIMIT 1"
        )).fetchone()
        if dup:
            print('[WARN] downgrade: 镜像地址 "%s" 存在 %s 条重复记录（新格式允许同镜像多版本），'
                  '无法恢复 uq_runtime_image 唯一约束；数据已保留，请人工评估。' % (dup[0], dup[1]))
        else:
            with op.batch_alter_table('runtime_image_version') as batch_op:
                batch_op.create_unique_constraint('uq_runtime_image', ['image'])
