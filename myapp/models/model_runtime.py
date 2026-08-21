# -*- coding: utf-8 -*-
"""模型 Runtime 镜像管理（合并入「镜像管理」）

Runtime 元数据（runtime_key / runtime_version / runtime_enabled）直接挂在
「开发 → 镜像管理」的 Images 表上，镜像管理成为镜像资产的唯一来源：

- Images.runtime_key / runtime_version / runtime_enabled：大模型 Runtime 镜像的版本属性，
  普通镜像这三个字段为空/默认 true，不受 Runtime 管理影响
- model_runtime_mapping：模型 + 场景 + 镜像的人工验证兼容关系
  （is_default 决定新任务使用哪个镜像；images_id 直接关联 Images）

约定：
- 只有人工验证 PASS 的 Model + Runtime 组合才写入 mapping
- runtime_enabled=false 只阻止新任务选择，不删除镜像与历史任务
- 同一 model + scene + runtime_key 只允许一个 is_default=true（在 service/view 层事务保证）
- mapping.runtime_key 冗余保留（列表/搜索/默认分组键），保存时强制自动同步自 images.runtime_key，
  禁止人工输入两份数据
"""
from flask_appbuilder import Model
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable

# 平台真实存在的大模型 Runtime 框架。
# 与 job_template.runtime_key / Resolver RUNTIME_SCENE 保持一致。
RUNTIME_KEY_CHOICES = [
    'msswift',
    'llama_factory',
    'deepspeed',
    'litgpt',
    'gptqmodel',
    'vllm',
    'sglang',
]


class RuntimeImageVersion(Model, AuditMixinNullable, MyappModelBase):
    """（历史遗留表，并入 Images，不再注册任何 API/菜单，表结构保留兼容历史数据）"""
    __tablename__ = 'runtime_image_version'
    id = Column(Integer, primary_key=True, comment='id主键')
    runtime_key = Column(String(100), nullable=False, index=True, comment='Runtime类型，如 msswift/gptqmodel/vllm/sglang')
    version = Column(String(100), nullable=False, comment='Runtime版本，如 4.0-r1')
    images_id = Column(Integer, ForeignKey('images.id'), nullable=True, index=True, comment='镜像id（镜像管理Images表，镜像资产唯一来源）')
    images = relationship(
        "Images", foreign_keys=[images_id], lazy='selectin'
    )
    image = Column(String(500), nullable=True, comment='Harbor完整镜像地址（兼容旧数据；新数据由 images 关联自动同步，后续可删除）')
    enabled = Column(Boolean, nullable=False, default=True, comment='是否允许新任务使用（false仅停用，不删除）')
    remark = Column(String(500), comment='备注')

    def __repr__(self):
        return '%s / %s' % (self.runtime_key, self.version)


class ModelRuntimeMapping(Model, AuditMixinNullable, MyappModelBase):
    __tablename__ = 'model_runtime_mapping'
    id = Column(Integer, primary_key=True, comment='id主键')
    model = Column(String(500), nullable=False, index=True, comment='与--model参数匹配的模型标识（精确匹配）')
    scene = Column(String(50), nullable=False, comment='场景：finetune/pretrain/quantization/inference')
    # runtime_key 冗余保留（列表/搜索/默认分组键），保存时强制 = images.runtime_key（自动同步，不手工输入）
    runtime_key = Column(String(100), nullable=False, comment='Runtime类型，如 msswift（自动同步自所选镜像）')
    images_id = Column(Integer, ForeignKey('images.id'), nullable=False, comment='Runtime镜像id（镜像管理Images表）')
    images = relationship(
        "Images", foreign_keys=[images_id], lazy='selectin'
    )
    is_default = Column(Boolean, nullable=False, default=False, comment='是否该模型当前默认Runtime')
    remark = Column(String(500), comment='验证说明')

    @property
    def runtime_image(self):
        """表单联动下拉/列表显示串：Runtime类型 / 版本 / 镜像名（只读，schema dump-only）"""
        img = self.images if self.images_id else None
        if not img or not img.runtime_key:
            return ''
        version = img.runtime_version or '-'
        return '%s / %s / %s' % (img.runtime_key, version, img.name)

    def __repr__(self):
        return '%s/%s/%s' % (self.model, self.scene, self.runtime_key)


def ensure_single_default(mapping, dbsession=None):
    """同一 model + scene + runtime_key 只允许一个 is_default=true（事务保证）。

    MySQL 不方便使用 partial unique constraint，因此通过 model 层在事务中保证：
    - 设置新默认 Runtime 时，自动把该组其它映射的 is_default 置为 false
    - 在 FAB pre_add / pre_update 中调用，mapping 为已应用新值（is_default）的对象
    - 不主动 commit，由调用方（FAB datamodel add/edit）的同一事务提交
    """
    from myapp import db as _db

    dbsession = dbsession or _db.session
    if not mapping.is_default:
        return
    query = dbsession.query(ModelRuntimeMapping).filter(
        ModelRuntimeMapping.model == mapping.model,
        ModelRuntimeMapping.scene == mapping.scene,
        ModelRuntimeMapping.runtime_key == mapping.runtime_key,
        ModelRuntimeMapping.is_default == True,  # noqa: E712
    )
    if getattr(mapping, 'id', None):
        query = query.filter(ModelRuntimeMapping.id != mapping.id)
    query.update({ModelRuntimeMapping.is_default: False}, synchronize_session=False)
