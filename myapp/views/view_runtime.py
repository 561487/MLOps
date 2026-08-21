# -*- coding: utf-8 -*-
"""模型 Runtime 镜像管理 —— 后台管理页面（Runtime 版本并入「镜像管理」）

- 镜像的 Runtime 元数据（runtime_key / runtime_version / runtime_enabled）在
  「开发 → 镜像管理」的 Images 表上配置，镜像管理 = 镜像资产唯一来源
- 模型 Runtime 映射：model / scene / runtime_key(联动) / runtime_image(联动下拉) / is_default / remark
- 两级联动（复用前端 ADUGTemplate 的 column_related 机制，前端零改动）：
  1. scene → runtime_key：不同场景只出现允许的 Runtime 类型
  2. runtime_key → runtime_image：只显示该类型「已启用」的镜像
- 保存链路说明（真实 post()/put() 流程）：
  前端提交的是 runtime_image 显示串（通用组件联动 option value==label 的约束，
  无法直接提交 images_id）；baseApi.post() 的 add_columns 白名单过滤会丢弃不在
  add_columns 中的字段，因此不能依赖 pre_add_req 注入 images_id。
  正确做法：pre_add_req 暂存显示串，pre_add/pre_update 在 INSERT 前把显示串
  解析回 Images 并回填 item.images_id / item.runtime_key（后端唯一转换点）。
- 后端强校验（不依赖前端）：images_id 非空、镜像存在、runtime_enabled=true、
  runtime_key 与镜像一致、scene 与 Runtime 类型组合合法；runtime_key 自动同步自
  所选镜像（禁止手工输入两份数据）
"""
from flask import g
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from wtforms import BooleanField, SelectField, StringField
from wtforms.validators import DataRequired, Length

from myapp import app, appbuilder, db
from myapp.exceptions import MyappException
from myapp.forms import MySelect2Widget
from myapp.models.model_job import Images
from myapp.models.model_runtime import (
    ModelRuntimeMapping,
    RUNTIME_KEY_CHOICES,
    ensure_single_default,
)
from myapp.views.baseApi import MyappModelRestApi
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface

conf = app.config

# 平台真实存在的大模型操作场景（第三版保持）：
# - finetune：msswift / llama_factory / deepspeed
# - pretrain：litgpt（scene 预留，运行链路后续接入）
# - quantization：gptqmodel
# - inference：vllm / sglang（scene 预留，推理创建链路暂未接入 Resolver）
SCENE_CHOICES = [
    ['finetune', _('finetune(微调/训练)')],
    ['pretrain', _('pretrain(预训练)')],
    ['quantization', _('quantization(量化)')],
    ['inference', _('inference(推理)')],
]

# 场景 → 允许的 Runtime 类型（第一层联动 + 后端组合校验，页面与后端同时保证）
SCENE_RUNTIME_MAP = {
    'finetune': ['msswift', 'llama_factory', 'deepspeed'],
    'pretrain': ['litgpt'],
    'quantization': ['gptqmodel'],
    'inference': ['vllm', 'sglang'],
}

# 镜像显示串分隔符（' / ' 不会出现在 Harbor 镜像地址中，可安全按末段还原镜像名）
DISPLAY_SEP = ' / '


def _runtime_image_display(img):
    """镜像下拉/映射列表显示串：Runtime类型 / 版本 / 镜像名（末段为镜像名）"""
    version = img.runtime_version or '-'
    return '%s / %s / %s' % (img.runtime_key, version, img.name)


def _enabled_runtime_images():
    """已启用的大模型 Runtime 镜像（mapping 可选范围：普通镜像/停用镜像不可选）"""
    return db.session.query(Images).filter(
        Images.runtime_key.isnot(None),
        Images.runtime_key != '',
        Images.runtime_enabled == True,  # noqa: E712
    ).order_by(Images.runtime_key, Images.runtime_version.desc()).all()


def _calculate_id(str_list):
    """与前端 calculateId 一致：各字符串 charCode 之和（联动配置键）"""
    return sum(sum(ord(c) for c in (s or '')) for s in str_list)


def _build_column_related():
    """两级联动配置（column_related，前端 ADUGTemplate 级联刷新下拉）：

    1. scene → runtime_key：选择场景后 Runtime 类型只出现该场景允许的框架
    2. runtime_key → runtime_image：选择类型后镜像下拉只显示该类型已启用的镜像
    """
    imgs = _enabled_runtime_images()
    scene_related = [
        {'src_value': [scene], 'des_value': list(keys)}
        for scene, keys in SCENE_RUNTIME_MAP.items()
    ]
    runtime_related = [
        {'src_value': [key], 'des_value': [_runtime_image_display(i) for i in imgs if i.runtime_key == key]}
        for key in RUNTIME_KEY_CHOICES
    ]
    return {
        'scene_runtime_key': {
            'src_columns': ['scene'],
            'des_columns': ['runtime_key'],
            'related': scene_related,
        },
        'runtime_key_images': {
            'src_columns': ['runtime_key'],
            'des_columns': ['runtime_image'],
            'related': runtime_related,
        },
    }


class ModelRuntimeMapping_ModelView_Base():
    route_base = '/modelruntimemapping_modelview/api'
    datamodel = SQLAInterface(ModelRuntimeMapping)
    label_title = _('模型 Runtime 映射')

    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    list_columns = ['model', 'scene', 'runtime_key', 'runtime_image', 'is_default', 'remark', 'creator', 'modified']
    show_columns = list_columns
    # runtime_key / images_id 不人工输入：由所选镜像自动取得（数据上保证 runtime_key == images.runtime_key）
    add_columns = ['model', 'scene', 'runtime_key', 'runtime_image', 'is_default', 'remark']
    edit_columns = add_columns
    search_columns = ['model', 'scene', 'runtime_key', 'is_default']
    base_order = ('id', 'desc')
    order_columns = ['id']

    spec_label_columns = {
        "model": _('模型'),
        "scene": _('场景'),
        "runtime_key": _('Runtime类型'),
        "runtime_image": _('镜像版本'),
        "is_default": _('默认'),
        "remark": _('验证说明'),
    }
    cols_width = {
        "model": {"type": "ellip2", "width": 260},
        "runtime_image": {"type": "ellip2", "width": 460},
    }

    # 两级联动配置（column_related）：每次 _info 请求实时生成，管理员新增镜像后自动可见。
    # 注意：不能用 @property（_init_properties 的 dir/getattr 会对属性求值触发 import 期 DB 查询）
    def column_related(self):
        return _build_column_related()

    add_form_extra_fields = {
        "model": StringField(
            _('模型'),
            description=_('模型标识，ModelScope 模型建议填写完整 repo id，例如 Qwen/Qwen3-0.6B。历史模型名称仍兼容。'),
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Length(1, 500)]
        ),
        "scene": SelectField(
            _('场景'),
            default='finetune',
            choices=SCENE_CHOICES,
            widget=MySelect2Widget(),
            validators=[DataRequired()]
        ),
        "runtime_key": SelectField(
            _('Runtime类型'),
            description=_('用于筛选镜像版本'),
            choices=[[k, k] for k in RUNTIME_KEY_CHOICES],
            validators=[DataRequired()]
        ),
        "runtime_image": SelectField(
            _('镜像版本'),
            description=_('仅显示该 Runtime 类型已启用的镜像'),
            choices=[],
            validators=[DataRequired()]
        ),
        "is_default": BooleanField(
            _('默认'),
            description=_('true=该模型新任务默认使用此镜像；同一 模型+场景+Runtime 只允许一个默认'),
            default=False,
        ),
        "remark": StringField(
            _('验证说明'),
            description=_('请输入验证说明，例如：Qwen/Qwen3.5-2B SFT+LoRA 验证通过'),
            default='',
            widget=BS3TextFieldWidget(),
        ),
    }
    edit_form_extra_fields = add_form_extra_fields

    def check_edit_permission(self, item):
        if not g.user.is_admin():
            return False
        return True
    check_delete_permission = check_edit_permission

    # 打开表单时刷新镜像下拉初始项（未选 Runtime 类型时的全部已启用 Runtime 镜像）。
    # 注意 choices 必须用 list 而非 tuple（baseApi.make_ui_info 只认 type(choice)==list）
    def pre_add_web(self):
        self.add_form_extra_fields['runtime_image'].kwargs['choices'] = [
            [d, d] for d in (_runtime_image_display(i) for i in _enabled_runtime_images())]

    def pre_update_web(self, item):
        self.pre_add_web()

    # ---- 真实保存链路（post()/put()）：----
    # post(): pre_add_req(1345) → add_columns 白名单过滤(1352) → schema.load(1369) → pre_add(1378) → INSERT(1379)
    # put():  循环(1430) → pre_update_req(1445) → _merge_update_item(1452) → 过滤(1453) → schema.load(1454)
    #         → pre_update(1477) → UPDATE
    # 关键点：
    # 1. runtime_image 是 dump-only property，marshmallow load 会直接拒绝（Unknown field），
    #    必须在进入 schema 前把它从请求中取出（pop），暂存供 pre_add/pre_update 解析回填
    # 2. put() 的 _merge_update_item 会把已存记录的 dump（含 runtime_image）合并回 data，
    #    必须 override 一并移除，否则编辑也会 Unknown field
    def pre_add_req(self, req_json):
        self._pending_runtime_image = (req_json.pop('runtime_image', '') or '')
        return req_json

    def pre_update_req(self, req_json, src_item):
        self._pending_runtime_image = (req_json.pop('runtime_image', '') or '')
        return req_json

    def _merge_update_item(self, model_item, data):
        # 编辑路径：dump 合并会把 runtime_image（dump-only）重新带回 data → 移除，避免 load 拒绝
        data = super(ModelRuntimeMapping_ModelView_Base, self)._merge_update_item(model_item, data)
        data.pop('runtime_image', None)
        return data

    def _resolve_pending_image(self, item):
        """INSERT/UPDATE 前把前端提交的 runtime_image 解析回 Images 并回填 images_id/runtime_key。

        兼容两种提交值：
        - 镜像 id（纯数字，如 '123'）——后续前端若改为直接提交 id 也成立
        - 显示串（'Runtime类型 / 版本 / 镜像名'）——当前前端联动 value==label 的实际提交格式
        """
        display = (getattr(self, '_pending_runtime_image', None) or '')
        # 防御：antd 某些模式可能提交 {label, value} 对象
        if isinstance(display, dict):
            display = display.get('value') or display.get('label') or ''
        display = str(display).strip()
        if not display:
            # 编辑时未重新选择镜像 → 保留原关联；新增时必须选择
            if getattr(item, 'images_id', None):
                return item
            raise MyappException('请选择有效的 Runtime 镜像')
        if display.isdigit():
            img = db.session.query(Images).get(int(display))
        else:
            name = display.rsplit(DISPLAY_SEP, 1)[-1].strip()
            img = db.session.query(Images).filter_by(name=name).first()
        if not img or not img.runtime_key:
            raise MyappException('请选择有效的 Runtime 镜像（来自「开发 → 镜像管理」）')
        # 强校验 1：Runtime 类型与所选镜像必须一致（杜绝 msswift + gptqmodel 镜像组合）
        submitted_key = (item.runtime_key or '').strip()
        if submitted_key and submitted_key != img.runtime_key:
            raise MyappException(
                '所选镜像与 Runtime 类型不一致（%s ↔ %s），请修正' % (submitted_key, img.runtime_key))
        # 强校验 2：场景与 Runtime 类型组合必须合法（finetune + vllm 等非法组合拒绝）
        if img.runtime_key not in SCENE_RUNTIME_MAP.get(item.scene, []):
            raise MyappException('场景 %s 不允许 Runtime 类型 %s' % (item.scene, img.runtime_key))
        # 强校验 3：镜像必须已启用
        if not img.runtime_enabled:
            raise MyappException('镜像「%s」已停用（允许新任务使用=false），请选择其他镜像' % img.name)
        # 自动同步：runtime_key / images_id 均取自所选镜像（禁止手工输入两份数据）
        item.images_id = img.id
        item.runtime_key = img.runtime_key
        return item

    def _normalize_model(self, item):
        """模型标识轻量归一化：只去除首尾空白与必要的空值校验。

        不再强制 basename —— 完整 repo id（Qwen/Qwen3-0.6B）与历史模型名（Qwen3-0.6B）
        均原样保存，由 Resolver 对两种格式双兼容（primary 完整 repo id 优先，legacy 兜底）。
        """
        raw = (item.model or '').strip()
        if not raw:
            raise MyappException('模型标识无效，请输入有效的模型名称')
        item.model = raw

    def pre_add(self, item):
        self._normalize_model(item)
        self._resolve_pending_image(item)
        ensure_single_default(item)

    def pre_update(self, item):
        self._normalize_model(item)
        self._resolve_pending_image(item)
        ensure_single_default(item)


class ModelRuntimeMapping_ModelView_Api(ModelRuntimeMapping_ModelView_Base, MyappModelRestApi):
    pass


appbuilder.add_api(ModelRuntimeMapping_ModelView_Api)
