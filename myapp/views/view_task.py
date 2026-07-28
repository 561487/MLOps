import math
import random
import re

from flask_appbuilder.baseviews import expose_api

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import pysnooper
from myapp.models.model_job import Job_Template, Task, Pipeline
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp.utils.crypto import encrypt_value, decrypt_value, is_encrypted
from myapp import app, appbuilder, db, event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Environment, BaseLoader, DebugUndefined
import os
from wtforms.validators import DataRequired, Length, Regexp
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField, StringField, SelectField, FloatField, DateField, DateTimeField, SelectMultipleField

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, BS3PasswordFieldWidget, DatePickerWidget, DateTimePickerWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MyLineSeparatedListField, MyJSONField, MyBS3TextFieldWidget, MySelect2Widget, MySelectMultipleField
from myapp.utils.storage_volume import available_volume_choices, filter_selected_volume_mount, split_volume_mount
from flask_wtf.file import FileField
from .baseApi import (
    MyappModelRestApi
)
import logging
from flask import (
    flash,
    g,
    redirect, render_template
)

from flask_appbuilder import expose
import datetime, time, json

conf = app.config


# ---------------------------------------------------------------------------
# 敏感参数加解密辅助函数
# ---------------------------------------------------------------------------

def _find_encrypted_arg_names(job_args):
    """从 job_template_args 中找出所有标记了 encrypted: 1 的参数名"""
    encrypted_names = set()
    if not job_args:
        return encrypted_names
    for group in job_args.values():
        if not isinstance(group, dict):
            continue
        for arg_name, arg_attr in group.items():
            if isinstance(arg_attr, dict) and arg_attr.get('encrypted') == 1:
                encrypted_names.add(arg_name)
    return encrypted_names


def _arg_name_to_env(arg_name):
    """将 --ls_api_token 转为 LS_API_TOKEN 形式的环境变量名"""
    name = arg_name.lstrip('-')
    return name.upper().replace('-', '_')


def _encrypt_sensitive_args(task_args, job_args):
    """对 task_args 中标记了 encrypted 的字段进行加密，原地修改"""
    encrypted_names = _find_encrypted_arg_names(job_args)
    for name in encrypted_names:
        if name in task_args and task_args[name] and not is_encrypted(str(task_args[name])):
            task_args[name] = encrypt_value(str(task_args[name]))


def _decrypt_and_extract_sensitive_args(task_args, job_args):
    """
    解密 task_args 中的敏感字段，并返回 (cleaned_args, extra_env) 元组。
    - cleaned_args: 移除了敏感值的 args（敏感字段置空）
    - extra_env: 解密后的环境变量字典，如 {'LS_API_TOKEN': 'xxx'}
    """
    extra_env = {}
    cleaned = dict(task_args)
    encrypted_names = _find_encrypted_arg_names(job_args)
    for name in encrypted_names:
        if name in cleaned and cleaned[name]:
            value = str(cleaned[name])
            if is_encrypted(value):
                extra_env[_arg_name_to_env(name)] = decrypt_value(value)
            else:
                # 未加密的旧数据（兼容），也迁移到环境变量
                extra_env[_arg_name_to_env(name)] = value
            cleaned[name] = ''  # 敏感值不再出现在 CLI args 中
    return cleaned, extra_env


class Task_ModelView_Base():
    label_title = _('任务')
    datamodel = SQLAInterface(Task)
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    list_columns = ['name', 'label', 'pipeline', 'job_template', 'volume_mount', 'resource_memory', 'resource_cpu',
                    'resource_gpu', 'resource_rdma', 'timeout', 'retry', 'created_on', 'changed_on', 'monitoring', 'expand']
    cols_width = {
        "name": {"type": "ellip2", "width": 250},
        "label": {"type": "ellip2", "width": 200},
        "pipeline": {"type": "ellip2", "width": 200},
        "job_template": {"type": "ellip2", "width": 200},
        "volume_mount": {"type": "ellip2", "width": 600},
        "command": {"type": "ellip2", "width": 200},
        "args": {"type": "ellip2", "width": 400},
        "resource_memory": {"type": "ellip2", "width": 100},
        "resource_cpu": {"type": "ellip2", "width": 100},
        "resource_gpu": {"type": "ellip2", "width": 100},
        "resource_rdma": {"type": "ellip2", "width": 100},
        "timeout": {"type": "ellip2", "width": 100},
        "retry": {"type": "ellip2", "width": 100},
        "created_on": {"type": "ellip2", "width": 300},
        "changed_on": {"type": "ellip2", "width": 300},
        "monitoring": {"type": "ellip2", "width": 300},
        "node_selector": {"type": "ellip2", "width": 200},
        "expand": {"type": "ellip2", "width": 300},
    }
    show_columns = ['name', 'label', 'pipeline', 'job_template', 'volume_mount', 'command', 'overwrite_entrypoint',
                    'working_dir', 'args_html', 'resource_memory', 'resource_cpu', 'resource_gpu', 'resource_rdma', 'timeout', 'retry',
                    'skip', 'created_by', 'changed_by', 'created_on', 'changed_on', 'monitoring_html']
    add_columns = ['name', 'label', 'job_template', 'pipeline', 'working_dir', 'command', 'args', 'volume_mount', 'resource_memory', 'resource_cpu', 'resource_gpu', 'resource_rdma', 'timeout', 'retry', 'skip',
                   'expand']
    edit_columns = ['name', 'label', 'working_dir', 'command', 'args', 'volume_mount', 'resource_memory',
                    'resource_cpu', 'resource_gpu', 'resource_rdma', 'timeout', 'retry', 'skip', 'expand']
    base_order = ('id', 'desc')
    order_columns = ['id']
    search_columns = ['pipeline', 'name', 'label']

    conv = GeneralModelConverter(datamodel)

    add_form_extra_fields = {
        "args": StringField(
            _('启动参数'),
            widget=MyBS3TextAreaFieldWidget(rows=10,is_json=True),
        ),
        "pipeline": QuerySelectField(
            _('任务流'),
            query_factory=lambda: db.session.query(Pipeline),
            allow_blank=True,
            widget=Select2Widget(extra_classes="readonly"),
        ),
        "job_template": QuerySelectField(
            _('任务模板'),
            query_factory=lambda: db.session.query(Job_Template),
            allow_blank=True,
            widget=Select2Widget(),
        ),

        "name": StringField(
            label= _('名称'),
            description= _('英文名(小写字母、数字、- 组成)，最长50个字符'),
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54), DataRequired()]
        ),
        "label": StringField(
            label= _('标签'),
            description= _('中文名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "volume_mount": StringField(
            label= _('挂载'),
            description= _('外部挂载，格式:<br>$pvc_name1(pvc):/$container_path1,$pvc_name2(pvc-share):/$container_path2,$storage_pvc_name(storage):/$container_path3,$hostpath1(hostpath):/$container_path4,$configmap(configmap):/$container_path5,$secret(secret):/$container_path6,$memory_size(memory):/$container_path7<br>注意pvc会自动挂载对应目录下的个人username子目录'),
            widget=BS3TextFieldWidget(),
            default='kubeflow-user-workspace(pvc):/mnt',
            validators=[Regexp('^[\x00-\x7F]*$')]
        ),
        "working_dir": StringField(
            label= _('工作目录'),
            description= _('工作目录，容器启动的初始所在目录，不填默认使用Dockerfile内定义的工作目录'),
            widget=BS3TextFieldWidget()
        ),
        "command": StringField(
            label= _('启动命令'),
            description= _('启动命令'),
            widget=BS3TextFieldWidget()
        ),
        "overwrite_entrypoint": BooleanField(
            label= _('覆盖入口点'),
            description= _('启动命令是否覆盖Dockerfile中ENTRYPOINT，不覆盖则叠加。')
        ),
        "node_selector": StringField(
            label= _('机器选择'),
            description= _('运行当前task所在的机器'),
            widget=BS3TextFieldWidget(),
            default=Task.node_selector.default.arg,
            validators=[]
        ),
        'resource_memory': StringField(
            label= _('memory'),
            default=Task.resource_memory.default.arg,
            description= _('内存的资源使用配置，示例1G，10G， 最大100G，如需更多联系管理员'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^[0-9]*G$")]
        ),
        'resource_cpu': StringField(
            label= _('cpu'),
            default=Task.resource_cpu.default.arg,
            description= _('cpu的资源使用配置(单位核)，示例 0.4，10，最大50核，如需更多联系管理员'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[0-9]*$")]
        ),
        'timeout': IntegerField(
            label= _('超时'),
            default=Task.timeout.default.arg,
            description= _('task运行时长限制，为0表示不限制(单位s)'),
            widget=BS3TextFieldWidget()
        ),
        'retry': IntegerField(
            label= _('重试'),
            default=Task.retry.default.arg,
            description= _('task重试次数'),
            widget=BS3TextFieldWidget()
        ),
        'outputs': StringField(
            label= _('输出'),
            default=Task.outputs.default.arg,
            description= _('task输出文件，支持容器目录文件和minio存储路径'),
            widget=MyBS3TextAreaFieldWidget(rows=3)
        ),
    }

    add_form_extra_fields['resource_gpu'] = StringField('gpu', default='0', description= _('gpu的资源使用配置(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100)'),widget=BS3TextFieldWidget(),validators=[DataRequired(),Regexp('^[\-\.0-9,a-zA-Z\(\)]*$')])
    add_form_extra_fields['resource_rdma'] = StringField('rdma', default='0', description= _('RDMA的资源使用配置，示例 0，1，10，填写方式咨询管理员'), widget=BS3TextFieldWidget())

    edit_form_extra_fields = add_form_extra_fields

    # 处理form请求
    # @pysnooper.snoop(watch_explode=('form'))
    def process_form(self, form, is_created):
        # from flask_appbuilder.forms import DynamicForm
        if 'job_describe' in form._fields:
            del form._fields['job_describe']  # 不处理这个字段

    def _set_volume_mount_select_field(self, project=None, current_volume_mount=''):
        self.add_form_extra_fields['volume_mount'] = MySelectMultipleField(
            label=_('挂载卷'),
            description=_('选择项目默认挂载卷或已同步的存储资源，保存后会自动生成 volume_mount 表达式'),
            default=current_volume_mount,
            widget=MySelect2Widget(multiple=True),
            choices=available_volume_choices(
                g.user,
                project=project,
                namespace=conf.get('PIPELINE_NAMESPACE', 'pipeline'),
                current_volume_mount=current_volume_mount
            ),
        )
        self.edit_form_extra_fields = self.add_form_extra_fields

    def _set_volume_mount_text_field(self):
        self.add_form_extra_fields['volume_mount'] = StringField(
            label=_('挂载'),
            description=_('外部挂载，格式:<br>$pvc_name1(pvc):/$container_path1,$pvc_name2(pvc-share):/$container_path2,$storage_pvc_name(storage):/$container_path3,$hostpath1(hostpath):/$container_path4,$configmap(configmap):/$container_path5,$secret(secret):/$container_path6,$memory_size(memory):/$container_path7<br>注意pvc会自动挂载对应目录下的个人username子目录'),
            widget=BS3TextFieldWidget(),
            default='kubeflow-user-workspace(pvc):/mnt',
            validators=[Regexp('^[\x00-\x7F]*$')]
        )
        self.edit_form_extra_fields = self.add_form_extra_fields

    def set_columns_related(self, exist_add_args, response_add_columns):
        if 'volume_mount' not in response_add_columns:
            return
        pipeline_value = exist_add_args.get('pipeline') or exist_add_args.get('pipeline_id') or {}
        if isinstance(pipeline_value, dict):
            pipeline_value = pipeline_value.get('id') or pipeline_value.get('value')
        pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_value)).first() if str(pipeline_value).isdigit() else None
        project = pipeline.project if pipeline else None
        current_volume_mount = exist_add_args.get('volume_mount') or (project.volume_mount if project else '')
        choices = available_volume_choices(
            g.user,
            project=project,
            namespace=conf.get('PIPELINE_NAMESPACE', 'pipeline'),
            current_volume_mount=current_volume_mount
        )
        response_add_columns['volume_mount'].update({
            "label": _('挂载卷'),
            "description": _('选择项目默认挂载卷或已同步的存储资源，保存后会自动生成 volume_mount 表达式'),
            "type": "Select",
            "ui-type": "select2",
            "default": split_volume_mount(current_volume_mount),
            "choices": choices,
            "values": [{"id": choice[0], "value": choice[1]} for choice in choices],
        })

    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        if g.user and g.user.is_admin():
            return True
        if g.user and g.user.username and item.pipeline and hasattr(item.pipeline, 'created_by'):
            if g.user.username == item.pipeline.created_by.username:
                return True
        # flash('just creator can edit/delete ', 'warning')
        return False

    check_delete_permission = check_edit_permission


    def pre_add_web(self, task=None):

        if g.user.is_admin():
            project = task.pipeline.project if task and task.pipeline else None
            self._set_volume_mount_select_field(project, task.volume_mount if task else '')
            if 'volume_mount' not in self.add_columns:
                self.add_columns = self.add_columns + ['volume_mount']
        else:
            project = task.pipeline.project if task and task.pipeline else None
            self._set_volume_mount_select_field(project, task.volume_mount if task else '')
            if 'volume_mount' not in self.add_columns:
                self.add_columns = self.add_columns + ['volume_mount']

        self.edit_columns = self.add_columns.copy()
        self.edit_columns.remove('job_template')
        self.edit_columns.remove('pipeline')

    pre_update_web = pre_add_web

    # 验证args参数
    # @pysnooper.snoop(watch_explode=('item'))
    def task_args_check(self, item):
        core.validate_str(item.name, 'name')
        core.validate_json(item.args)
        task_args = json.loads(item.args)
        task_args = {
            key:(task_args[key].strip(' ') if type(task_args[key])==str else task_args[key]) for key in task_args
        }
        job_args = json.loads(item.job_template.args)

        # 先校验
        validated = core.validate_task_args(task_args, job_args)

        # 加密标记了 "encrypted": 1 的敏感参数（仅对非空值加密）
        _encrypt_sensitive_args(validated, job_args)

        item.args = json.dumps(validated, indent=4, ensure_ascii=False)

        if item.volume_mount and ":" not in item.volume_mount:
            raise MyappException('volume_mount is not valid, must contain : or null')

    # 在web界面上添加一个图标
    # @pysnooper.snoop()
    def post_add(self, task):
        pipeline = task.pipeline
        expand = json.loads(pipeline.expand) if pipeline.expand else []
        for ui_node in expand:
            if ui_node.get('id', 0) == task.id:
                return
        expand.append(
            {
                "id": str(task.id),
                "type": 'dataSet',
                "position": {
                    "x": random.randint(50, 1000),
                    "y": random.randint(50, 600),
                },
                "data": {
                    "name": task.name,
                    "label": task.label
                }

            }
        )
        pipeline.expand = json.dumps(expand, ensure_ascii=False, indent=4)
        db.session.commit()
        pass

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):


        parameter = json.loads(item.pipeline.parameter)
        if parameter.get("demo", 'false').lower() == 'true':
            raise MyappException(__("示例pipeline，不允许修改，请复制后编辑"))
        item.name = item.name.replace('_', '-')[0:54].lower()
        if item.job_template is None:
            raise MyappException(__("Job Template 为必选"))

        item.resource_memory=item.resource_memory.upper()
        item.resource_gpu = item.resource_gpu.upper() if item.resource_gpu else '0'
        if 'G' not in item.resource_memory and 'M' not in item.resource_memory:
            item.resource_memory = item.resource_memory+"G"
        selected_volume_mount = item.volume_mount
        try:
            pipeline_parameter = json.loads(item.pipeline.parameter or '{}')
        except Exception:
            pipeline_parameter = {}
        item.volume_mount = core.merge_volume_mount(
            item.pipeline.project.volume_mount,
            pipeline_parameter.get('volume_mount', '')
        )  # 默认使用项目和流水线的配置

        if item.job_template.volume_mount and item.job_template.volume_mount not in item.volume_mount:
            if item.volume_mount:
                item.volume_mount += "," + item.job_template.volume_mount
            else:
                item.volume_mount = item.job_template.volume_mount
        if selected_volume_mount:
            if not g.user.is_admin():
                selected_volume_mount = filter_selected_volume_mount(
                    g.user,
                    item.pipeline.project,
                    selected_volume_mount,
                    namespace=conf.get('PIPELINE_NAMESPACE', 'pipeline')
                )
            item.volume_mount = core.merge_volume_mount(item.volume_mount, selected_volume_mount)
        item.resource_memory = core.check_resource_memory(item.resource_memory)
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu)
        if not item.args:
            item.args = '{}'
        self.task_args_check(item)
        item.create_datetime = datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()
        gpu_num, _, _ = core.get_gpu(item.resource_gpu)
        gpu_num = math.ceil(float(str(gpu_num).split(',')[-1]))
        item.node_selector = item.node_selector.replace(';mps=true', '').replace(';mps=false', '').replace('mps=true;', '').replace('mps=false;', '').replace('mps=true', '').replace('mps=false', '')
        if gpu_num < 0:
            item.node_selector = item.node_selector.replace('cpu=true', 'gpu=true')
            if 'mps=true' not in item.node_selector:
                item.node_selector += ';mps=true'
        elif gpu_num==0:
            item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')
        else:
            item.node_selector = item.node_selector.replace('cpu=true', 'gpu=true')
            if 'mps=false' not in item.node_selector:
                item.node_selector += ';mps=false'

    def pre_update_req(self,req_json=None,src_item=None,*args,**kwargs):
        if src_item and src_item.pipeline and src_item.pipeline.parameter:
            parameter = json.loads(src_item.pipeline.parameter)
            if parameter.get("demo", 'false').lower() == 'true':
                raise MyappException(__("示例pipeline，不允许修改，请复制后编辑"))

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_update(self, item):

        item.name = item.name.replace('_', '-')[0:54].lower()
        if item.resource_gpu:
            item.resource_gpu = str(item.resource_gpu).upper()
        item.resource_memory=item.resource_memory.upper()
        item.resource_gpu = item.resource_gpu.upper() if item.resource_gpu else '0'
        if 'G' not in item.resource_memory and 'M' not in item.resource_memory:
            item.resource_memory = item.resource_memory+"G"
        if item.job_template is None:
            raise MyappException(__("Job Template 为必选"))

        selected_volume_mount = item.volume_mount
        if selected_volume_mount and not g.user.is_admin():
            selected_volume_mount = filter_selected_volume_mount(
                g.user,
                item.pipeline.project,
                selected_volume_mount,
                namespace=conf.get('PIPELINE_NAMESPACE', 'pipeline')
            )

        try:
            pipeline_parameter = json.loads(item.pipeline.parameter or '{}')
        except Exception:
            pipeline_parameter = {}
        item.volume_mount = core.merge_volume_mount(
            item.pipeline.project.volume_mount,
            pipeline_parameter.get('volume_mount', ''),
            selected_volume_mount
        )

        # # 切换了项目组，要把项目组的挂载加进去
        all_project_volumes = []
        if item.volume_mount:
            all_project_volumes = [x.strip() for x in item.volume_mount.replace('，',',').split(',') if x.strip()]
        if item.job_template.volume_mount:
            all_project_volumes += [x.strip() for x in item.job_template.volume_mount.replace('，',',').split(',') if x.strip()]
        for volume_mount in all_project_volumes:
            if ":" in volume_mount:
                volume, mount = volume_mount.split(":")[0], volume_mount.split(":")[1]
                if mount not in item.volume_mount:
                    item.volume_mount = item.volume_mount.strip(',') + "," + volume_mount

        # 修改失败，直接换为原来的
        if item.volume_mount and ':' not in item.volume_mount:
            item.volume_mount = self.src_item_json.get('volume_mount', '')

        if item.volume_mount:
            # 合并项目组的挂载
            item.volume_mount = core.merge_volume_mount(
                item.pipeline.project.volume_mount,
                pipeline_parameter.get('volume_mount', ''),
                item.volume_mount
            )


        if item.outputs:
            core.validate_json(item.outputs)
            item.outputs = json.dumps(json.loads(item.outputs), indent=4, ensure_ascii=False)
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)

        item.resource_memory = core.check_resource_memory(item.resource_memory, self.src_item_json.get('resource_memory', None))
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu, self.src_item_json.get('resource_cpu', None))
        # item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_resource_memory)
        # item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_resource_cpu)

        if not item.args:
            item.args = '{}'
        self.task_args_check(item)
        item.change_datetime = datetime.datetime.now()
        gpu_num, _, _ = core.get_gpu(item.resource_gpu)
        gpu_num = math.ceil(float(str(gpu_num).replace('，',',').split(',')[-1]))
        item.node_selector = item.node_selector.replace(';mps=true', '').replace(';mps=false', '').replace('mps=true;', '').replace('mps=false;', '').replace('mps=true', '').replace('mps=false', '')
        if gpu_num < 0:
            item.node_selector = item.node_selector.replace('cpu=true', 'gpu=true')
            if 'mps=true' not in item.node_selector:
                item.node_selector += ';mps=true'
        elif gpu_num==0:
            item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')
        else:
            item.node_selector = item.node_selector.replace('cpu=true', 'gpu=true')
            if 'mps=false' not in item.node_selector:
                item.node_selector += ';mps=false'

        # 修改了名称，要在pipeline的属性里面一起改了
        src_task_name = self.src_item_json.get('name', item.name)
        if item.name != src_task_name:
            pipeline = item.pipeline
            pipeline.dag_json = pipeline.dag_json.replace(f'"{src_task_name}"',f'"{item.name}"')
            pipeline.expand = pipeline.expand.replace(f'"{src_task_name}"',f'"{item.name}"')



    # 添加和删除task以后要更新pipeline的信息,会报错is not bound to a Session
    # # @pysnooper.snoop()
    # def post_add(self, item):
    #     item.pipeline.pipeline_file = dag_to_pipeline(item.pipeline, db.session)
    #     pipeline_argo_id, version_id = upload_pipeline(item.pipeline)
    #     if pipeline_argo_id:
    #         item.pipeline.pipeline_argo_id = pipeline_argo_id
    #     if version_id:
    #         item.pipeline.version_id = version_id
    #     db.session.commit()

    # # @pysnooper.snoop(watch_explode=('item'))
    # def post_update(self, item):
    #     # if type(item)==UnmarshalResult:
    #     #     pass
    #     item.pipeline.pipeline_file = dag_to_pipeline(item.pipeline, db.session)
    #     pipeline_argo_id, version_id = upload_pipeline(item.pipeline)
    #     if pipeline_argo_id:
    #         item.pipeline.pipeline_argo_id = pipeline_argo_id
    #     if version_id:
    #         item.pipeline.version_id = version_id
    #     # db.session.update(item)
    #     db.session.commit()

    # 因为删除就找不到pipeline了
    def pre_delete(self, item):
        parameter = json.loads(item.pipeline.parameter) if item.pipeline.parameter else {}
        if parameter.get("demo", 'false').lower() == 'true':
            raise MyappException(__("示例pipeline，不允许修改，请复制后编辑"))
        self.pipeline = item.pipeline

    def post_delete(self, item):
        # 删除task启动的所有实例，如果没有删掉，也不重新删除了。就不管了
        self.delete_task_run(item)

    def run_pod(self, task, k8s_client, run_id, namespace, pod_name, image, working_dir, command, args):

        # 模板中环境变量
        task_env = task.job_template.env + "\n" if task.job_template.env else ''

        # 注入解密后的敏感参数（如 API Token）
        if hasattr(task, '_extra_env') and task._extra_env:
            for env_key, env_val in task._extra_env.items():
                task_env += f'{env_key}={env_val}\n'

        HostNetwork = json.loads(task.job_template.expand).get("HostNetwork", False) if task.job_template.expand else False
        byte_string = run_id.encode('utf-8')
        import hashlib
        # 计算字节串的哈希值
        hash_object = hashlib.sha256(byte_string)
        hash_value = int(hash_object.hexdigest(), 16)
        # 将哈希值映射到指定范围
        hostPort = 40000 + 2*(hash_value % 1000)
        meet_ports = core.get_not_black_port(hostPort)

        if HostNetwork:
            task_env += 'PORT1=' + str(meet_ports[1])+ "\n"
            task_env += 'PORT2=' + str(meet_ports[2])+ "\n"

        gpu_num, _, resource_name = core.get_gpu(task.resource_gpu)
        if isinstance(gpu_num, (int, float)) and gpu_num < 0:
            _, _, resource_name = core.get_gpu_shared_resource(task.resource_gpu)

        # 系统环境变量
        task_env += 'KFJ_TASK_ID=' + str(task.id) + "\n"
        task_env += 'KFJ_TASK_NAME=' + str(task.name) + "\n"
        task_env += 'KFJ_TASK_NODE_SELECTOR=' + str(task.get_node_selector()) + "\n"
        task_env += 'KFJ_TASK_VOLUME_MOUNT=' + str(task.volume_mount) + "\n"
        task_env += 'KFJ_TASK_IMAGES=' + str(task.job_template.images) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_CPU=' + str(task.resource_cpu) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_MEMORY=' + str(task.resource_memory) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_GPU=' + str(task.resource_gpu.replace('+', '')) + "\n"
        task_env += 'KFJ_TASK_PROJECT_NAME=' + str(task.pipeline.project.name) + "\n"
        task_env += 'KFJ_PIPELINE_ID=' + str(task.pipeline_id) + "\n"
        task_env += 'KFJ_RUN_ID=' + run_id + "\n"
        task_env += 'KFJ_CREATOR=' + str(task.pipeline.created_by.username) + "\n"
        task_env += 'KFJ_RUNNER=' + str(g.user.username) + "\n"
        task_env += 'KFJ_PIPELINE_NAME=' + str(task.pipeline.name) + "\n"
        task_env += 'KFJ_NAMESPACE=pipeline' + "\n"
        task_env += f'GPU_RESOURCE_NAME={resource_name}' + "\n"
        task_env += f"GPU_SHARED_RESOURCE_NAME={conf.get('GPU_SHARED_RESOURCE_NAME', 'nvidia.com/gpu.shared')}" + "\n"

        # ---- SwanLab 训练监控环境变量（仅训练类模板） ----
        _job_template_name = (task.job_template.name or '') if task.job_template else ''
        _training_templates = conf.get('TRAINING_JOB_TEMPLATES', [])
        # swanlab_enabled 优先级：显式 false → 禁用；显式 true → 启用；未配置 → 走 TRAINING_JOB_TEMPLATES
        _swanlab_task_args = json.loads(task.args) if task.args else {}
        _swanlab_enabled = _swanlab_task_args.get('swanlab_enabled')
        if _swanlab_enabled is False:
            _inject_swanlab = False
        elif _swanlab_enabled is True:
            _inject_swanlab = True
        else:
            _inject_swanlab = _job_template_name in _training_templates
        if _inject_swanlab:
            _pipeline_name = task.pipeline.name if task.pipeline else ''
            _task_name = task.name or ''
            _task_label = task.label or ''

            # MLOps 侧变量
            _register_url = conf.get('MLOPS_MONITOR_REGISTER_URL',
                                     'http://10.121.177.20:18080/training_monitor/api/register')
            task_env += "MLOPS_TRAINING_MONITOR_ENABLE=true\n"
            task_env += "MLOPS_TRAINING_MONITOR_TYPE=swanlab\n"
            task_env += f"MLOPS_MONITOR_REGISTER_URL={_register_url}\n"
            task_env += f"MLOPS_PIPELINE_RUN_ID={run_id}\n"
            task_env += f"MLOPS_WORKFLOW_NAME={_pipeline_name}\n"
            task_env += f"MLOPS_TASK_ID={str(task.id)}\n"
            task_env += f"MLOPS_TASK_NAME={_task_label}\n"
            task_env += f"MLOPS_NODE_NAME={_task_name}\n"
            task_env += f"MLOPS_JOB_TEMPLATE_NAME={_job_template_name}\n"

            # SwanLab 侧变量
            _swanlab_api_host = conf.get('SWANLAB_API_HOST', 'http://10.121.177.227:8000')
            _swanlab_web_host = conf.get('SWANLAB_WEB_HOST', 'http://10.121.177.227:8000')
            _swanlab_logdir = conf.get('SWANLAB_LOGDIR', '') or ''
            _swanlab_proj_name = (_swanlab_task_args.get('swanlab_project') or
                                conf.get('SWANLAB_PROJ_NAME', 'mlops-training'))
            _swanlab_workspace = (_swanlab_task_args.get('swanlab_workspace') or
                                  conf.get('SWANLAB_WORKSPACE', 'mlops'))

            _user_mode = (_swanlab_task_args.get('swanlab_mode') or '').strip().lower()
            _cloud_defaults = ('hyperparam-search', 'hyperparam-search-nni', 'lightgbm', 'gbdt', 'model-distillation', 'llama-factory')
            if _user_mode:
                _swanlab_mode = 'cloud' if _user_mode == 'online' else _user_mode
            elif _job_template_name in _cloud_defaults:
                _swanlab_mode = 'cloud'
            else:
                _swanlab_mode = conf.get('SWANLAB_MODE', 'local') or 'local'

            task_env += f"SWANLAB_MODE={_swanlab_mode}\n"
            task_env += f"SWANLAB_API_HOST={_swanlab_api_host}\n"
            task_env += f"SWANLAB_WEB_HOST={_swanlab_web_host}\n"
            task_env += f"SWANLAB_EXP_NAME={_pipeline_name}-{_task_name}-{run_id[:8]}\n"
            task_env += f"SWANLAB_GROUP={run_id}\n"
            task_env += "SWANLAB_TAGS=mlops,training\n"
            task_env += "SWANLAB_PROBE_HARDWARE=true\n"
            task_env += "SWANLAB_PROBE_MONITOR=true\n"
            task_env += "SWANLAB_PROBE_MONITOR_INTERVAL=10\n"

            if _swanlab_mode == "cloud":
                # Cloud / Self-hosted Online 模式
                task_env += f"SWANLAB_PROJ_NAME={_swanlab_proj_name}\n"
                task_env += f"SWANLAB_WORKSPACE={_swanlab_workspace}\n"
                # SWANLAB_API_KEY 需从 K8s Secret 注入（debug pod 路径不支持 valueFrom）
            else:
                # Local / Watch 模式（保持现有逻辑不变）
                task_env += "SWANLAB_PROJ_NAME=mlops-training\n"
                if _swanlab_logdir:
                    task_env += f"SWANLAB_LOGDIR={_swanlab_logdir}\n"

        template_kwargs={}
        def template_str(src_str):
            rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
            des_str = rtemplate.render(creator=task.pipeline.created_by.username,
                                       datetime=datetime,
                                       runner=g.user.username if g and g.user and g.user.username else task.pipeline.created_by.username,
                                       uuid=uuid,
                                       pipeline_id=task.pipeline.id,
                                       pipeline_name=task.pipeline.name,
                                       cluster_name=task.pipeline.project.cluster['NAME'],
                                       **template_kwargs
                                       )
            return des_str

        # 全局环境变量
        pipeline_global_env = template_str(task.pipeline.global_env.strip()) if task.pipeline.global_env else ''  # 优先渲染，不然里面如果有date就可能存在不一致的问题
        pipeline_global_env = [env.strip() for env in pipeline_global_env.split('\n') if '=' in env.strip()]
        for env in pipeline_global_env:
            key, value = env[:env.index('=')], env[env.index('=') + 1:]
            if key not in task_env:
                task_env += key + '=' + value + "\n"

        # 全局环境变量可以在任务的参数中引用
        for global_env in pipeline_global_env:
            key, value = global_env.split('=')[0], global_env.split('=')[1]
            if key not in template_kwargs:
                template_kwargs[key] = value

        platform_global_envs = json.loads(template_str(json.dumps(conf.get('GLOBAL_ENV', {}), indent=4, ensure_ascii=False)))
        for global_env_key in platform_global_envs:
            if global_env_key not in task_env:
                task_env += global_env_key + '=' + platform_global_envs[global_env_key] + "\n"
        new_args = []
        if args:
            for arg in args:
                new_args.append(template_str(arg))

        # ---- SwanLab 大模型微调框架参数自动追加 ----
        _task_args = json.loads(task.args) if task.args else {}
        _framework_type = (_task_args.get('swanlab_framework_type') or '').strip().lower()
        _ML_OPERATORS = {'lightgbm', 'gbdt', 'xgb', 'hyperparam-search', 'hyperparam-search-nni',
                         'lr', 'knn', 'decision-tree', 'random-forest', 'random-forest-regression',
                         'kmean', 'bayesian', 'adaboost', 'arima', 'ar'}
        # 当前只正式支持 LLaMA-Factory；其他类型预留但输出 warning
        _SUPPORTED_FRAMEWORK_TYPES = {'llamafactory'}
        _RESERVED_FRAMEWORK_TYPES = {'modelscope_swift', 'transformers', 'trl'}
        _FRAMEWORK_ARGS_MAP = {
            'llamafactory': ['--report_to', 'swanlab'],
        }
        if _framework_type and _framework_type not in ('none', 'generic', ''):
            if _framework_type in _RESERVED_FRAMEWORK_TYPES:
                print(f"[swanlab] WARNING: swanlab_framework_type={_framework_type} is reserved but not yet validated, "
                      f"no args will be auto-appended and wrapper will not wrap")
            elif _framework_type not in _SUPPORTED_FRAMEWORK_TYPES:
                print(f"[swanlab] WARNING: unknown swanlab_framework_type={_framework_type}, "
                      f"supported: {sorted(_SUPPORTED_FRAMEWORK_TYPES)}")
        if (_framework_type in _SUPPORTED_FRAMEWORK_TYPES
                and _job_template_name not in _ML_OPERATORS
                and _inject_swanlab):
            _extra_args = _FRAMEWORK_ARGS_MAP.get(_framework_type, [])
            if _extra_args:
                _cmd_str = ' '.join(command) if command else ''
                _arg_str = ' '.join(new_args) if new_args else ''
                _combined = f"{_cmd_str} {_arg_str}"
                _skipped = []
                for _ea in _extra_args:
                    if _ea.lstrip('-') in _combined:
                        print(f"[swanlab] WARNING: {_ea} already in command, skipping")
                        _skipped.append(_ea)
                _extra_args = [a for a in _extra_args if a not in _skipped]
                if _extra_args:
                    print(f"[swanlab] auto-appending framework args: {_extra_args}")
                    new_args.extend(_extra_args)

            # Wrapper 包装：将原始命令包装进 swanlab_framework_wrapper，负责 monitor 注册和状态同步
            if (_framework_type in _SUPPORTED_FRAMEWORK_TYPES
                    and _job_template_name not in _ML_OPERATORS
                    and _inject_swanlab):
                _merged_cmd = list(command) if command else []
                if new_args:
                    _merged_cmd.extend(new_args)
                if _merged_cmd:
                    command = ['python3', '/app/common/swanlab_framework_wrapper.py', '--']
                    new_args = _merged_cmd
                    print(f"[swanlab] wrapped command with swanlab_framework_wrapper")

        if command:
            command = json.loads(template_str(json.dumps(command)))
        if working_dir:
            working_dir = template_str(working_dir)

        volume_mount = task.volume_mount

        resource_cpu = task.job_template.get_env('TASK_RESOURCE_CPU') if task.job_template.get_env('TASK_RESOURCE_CPU') and 'run-' in run_id else task.resource_cpu
        resource_gpu = task.job_template.get_env('TASK_RESOURCE_GPU') if task.job_template.get_env('TASK_RESOURCE_GPU') and 'run-' in run_id else task.resource_gpu
        resource_memory = task.job_template.get_env('TASK_RESOURCE_MEMORY') if task.job_template.get_env('TASK_RESOURCE_MEMORY') and 'run-' in run_id else task.resource_memory
        host_aliases=conf.get('HOSTALIASES')
        if task.job_template.host_aliases:
            host_aliases += "\n" + task.job_template.host_aliases

        image_pull_secrets = conf.get('HUBSECRET', [])
        from myapp.models.model_job import Repository
        user_repositorys = db.session.query(Repository).filter(Repository.created_by_fk == g.user.id).all()
        image_pull_secrets = list(set([task.job_template.images.repository.hubsecret]+image_pull_secrets + [rep.hubsecret for rep in user_repositorys]))
        if image_pull_secrets:
            task_env += 'HUBSECRET='+ ','.join(image_pull_secrets) + "\n"
        # print(resource_gpu)
        task.namespace=namespace
        db.session.commit()
        k8s_client.create_debug_pod(namespace,
                                    name=pod_name,
                                    labels={'app':task.pipeline.name,"pipeline": task.pipeline.name, 'task': task.name, 'user': g.user.username, 'run-id': run_id, 'pod-type': "task"},
                                    annotations={'project':task.pipeline.project.name},
                                    command=command,
                                    args=new_args,
                                    volume_mount=volume_mount,
                                    working_dir=working_dir,
                                    node_selector=task.get_node_selector(),
                                    resource_memory=resource_memory,
                                    resource_cpu=resource_cpu,
                                    resource_gpu=resource_gpu,
                                    resource_rdma = '0',
                                    image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
                                    image_pull_secrets=image_pull_secrets,
                                    image=image,
                                    hostAliases=host_aliases,
                                    env=task_env,
                                    privileged=task.job_template.privileged,
                                    accounts=task.job_template.accounts,
                                    username=task.pipeline.created_by.username, # 人员离职需要可以
                                    hostPort=[hostPort+1,hostPort+2] if HostNetwork else []
                                    )

    @event_logger.log_this
    @expose_api(description="单个任务debug",url="/debug/<task_id>", methods=["GET", "POST"])
    def debug(self, task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()

        # 只有管理员和创建者可以debug
        if task.pipeline.created_by_fk!=g.user.id and not g.user.is_admin():
            # 模板创建者可以调试模板
            message = __('仅管理员或创建者，可debug该任务')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        # 逻辑节点不能进行调试
        if task.job_template.name == conf.get('LOGICAL_JOB'):
            message = __('当前任务类型不允许进行调试')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        # 除了自定义节点其他节点不能单任务调试
        if task.job_template.name != conf.get('CUSTOMIZE_JOB'):
            # 模板创建者可以调试模板
            if not g.user.is_admin() and task.job_template.created_by.username != g.user.username:
                message = __('仅管理员或当前任务模板创建者，可启动debug模式')
                flash(message, 'warning')
                return self.response(400, **{"status": 1, "result": {}, "message": message})


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster.get('KUBECONFIG', ''))
        new_namespace = task.pipeline.project.pipeline_namespace
        old_namespace = task.namespace

        pod_name = "debug-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')

        if old_namespace!=new_namespace:
            k8s_client.delete_pods(namespace=old_namespace, pod_name=pod_name)

        pod = k8s_client.get_pods(namespace=new_namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史非运行态，直接删除
        # if pod and (pod['status']!='Running' and pod['status']!='Pending'):
        if pod and pod['status'] == 'Succeeded':
            k8s_client.delete_pods(namespace=new_namespace, pod_name=pod_name)
            time.sleep(2)
            pod = None
        # 没有历史或者没有运行态，直接创建
        image = task.job_template.images.name
        if json.loads(task.args).get('--work_images',''):
            image = json.loads(task.args)['--work_images']
        if json.loads(task.args).get('--work_image',''):
            image = json.loads(task.args)['--work_image']
        if json.loads(task.args).get('--images',''):
            image = json.loads(task.args)['--images']
        if json.loads(task.args).get('--image',''):
            image = json.loads(task.args)['--image']
        if json.loads(task.args).get('images',''):
            image = json.loads(task.args)['images']
        working_dir = None
        if json.loads(task.args).get('workdir', ''):
            working_dir = json.loads(task.args)['workdir']
        if json.loads(task.args).get('--workdir', ''):
            working_dir = json.loads(task.args)['--workdir']
        if json.loads(task.args).get('--working_dir', ''):
            working_dir = json.loads(task.args)['--working_dir']

        if not pod or pod['status'] != 'Running':
            run_id = "debug-" + str(uuid.uuid4().hex)
            command=['sh','-c','sleep 7200 && hour=`date +%H` && while [ $hour -ge 06 ];do sleep 3600;hour=`date +%H`;done']
            try:
                self.run_pod(
                    task=task,
                    k8s_client=k8s_client,
                    run_id=run_id,
                    namespace=new_namespace,
                    pod_name=pod_name,
                    image=image,
                    working_dir=working_dir,
                    command=command,
                    args=None
                )
            except Exception as e:
                return render_template('close.html', data=str(e).replace('<br>', '\n'))
                return self.response(400, **{"status": 1, "result": {}, "message": str(e)})

        try_num = 30
        message = __('启动时间过长，一分钟后刷新此页面')
        while (try_num > 0):
            pod = k8s_client.get_pods(namespace=new_namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                pod = pod[0]
            # 有历史非运行态，直接删除
            if pod:
                if pod['status'] == 'Running':
                    break
                else:
                    events = k8s_client.get_pod_event(namespace=new_namespace, pod_name=pod_name)
                    # try:
                    #     message = '启动时间过长，一分钟后刷新此页面'+", status:"+pod['status']+", message:"+json.dumps(pod['status_more']['conditions'],indent=4,ensure_ascii=False)
                    # except Exception as e:
                    #     print(e)
                    try:
                        # 有新消息要打印
                        for event in events:
                            message = f'"time: "{event["time"]} \ntype: {event["type"]} \nreason: {event["reason"]} \nmessage: {event["message"]}'
                            message = message.replace('\n','<br>')
                            # print(message, flush=True)
                            message += "<br><br>" + message
                    except Exception as e:
                        print(e)
            try_num = try_num - 1
            time.sleep(2)
        if try_num == 0:
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        return redirect("/k8s/web/debug/%s/%s/%s/%s" % (task.pipeline.project.cluster['NAME'], new_namespace, pod_name, pod_name))

    @event_logger.log_this
    @expose_api(description="单个任务run",url="/run/<task_id>", methods=["GET", "POST"])
    def run_task(self, task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()

        # 只有管理员和创建者可以debug
        if task.pipeline.created_by_fk!=g.user.id and not g.user.is_admin():
            # 模板创建者可以调试模板
            message = __('仅管理员或创建者，可运行该任务')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        # 逻辑节点和python节点不能进行单任务运行
        if task.job_template.name == conf.get('LOGICAL_JOB'):
            message = __('当前任务类型不允许进行运行')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        # 包含上游输出的不能进行单任务运行
        import re
        all_templates_vars = re.findall("(\{\{.*?\}\})",task.args)
        for var in all_templates_vars:
            if '.output' in var:
                message = __('包含接收上游输出，不允许单任务运行')
                flash(message, 'warning')
                return self.response(400, **{"status": 1, "result": {}, "message": message})


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster.get('KUBECONFIG', ''))
        new_namespace = task.pipeline.project.pipeline_namespace
        old_namespace = task.namespace
        pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=old_namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            run_id = pod['labels'].get("run-id", '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info=conf.get('CRD_INFO', {}), namespace=old_namespace, run_id=run_id)

            k8s_client.delete_pods(namespace=old_namespace, pod_name=pod_name)
            delete_time = datetime.datetime.now()
            while pod:
                time.sleep(2)
                pod = k8s_client.get_pods(namespace=old_namespace, pod_name=pod_name)
                check_date = datetime.datetime.now()
                if (check_date - delete_time).total_seconds() > 60:
                    message = __("超时，请稍后重试")
                    flash(message, category='warning')
                    return self.response(400, **{"status": 1, "result": {}, "message": message})

        # 没有历史或者没有运行态，直接创建
        if not pod:
            command = None
            if task.job_template.entrypoint and task.job_template.entrypoint.strip():
                command = task.job_template.entrypoint.strip()
            if task.command and task.command.strip():
                command = task.command.strip()
            if command:
                command = command.split(" ")
                command = [com for com in command if com]
            ops_args = []

            task_args = json.loads(task.args) if task.args else {}

            # 解密敏感参数，提取为环境变量（不在 CLI args 中暴露）
            job_args = json.loads(task.job_template.args) if task.job_template.args else {}
            task_args, extra_env = _decrypt_and_extract_sensitive_args(task_args, job_args)
            task._extra_env = extra_env  # 传递给 run_pod

            for task_attr_name in task_args:
                # 布尔型只添加参数名
                if type(task_args[task_attr_name]) == bool:
                    if task_args[task_attr_name]:
                        ops_args.append('%s' % str(task_attr_name))
                elif not task_args[task_attr_name]:  # 如果参数值为空，则都不添加
                    pass
                # json类型直接导入序列化以后的
                elif type(task_args[task_attr_name]) == dict or type(task_args[task_attr_name])==list:
                    ops_args.append('%s' % str(task_attr_name))
                    args_values = json.dumps(task_args[task_attr_name], ensure_ascii=False)
                    ops_args.append('%s' % args_values)
                # list类型逗号分隔就好了
                # # list类型，分多次导入
                # elif type(task_args[task_attr_name]) == list:
                #     for args_values in task_args[task_attr_name].split('\n'):
                #         ops_args.append('%s' % str(task_attr_name))
                #         # args_values = template_str(args_values) if re.match('\{\{.*\}\}',args_values) else args_values
                #         ops_args.append('%s' % args_values)

                elif task_attr_name not in ['images','workdir']:  # 如果参数名直接是这些，就不作为参数，而是替换模板的这两个配置
                    ops_args.append('%s' % str(task_attr_name))
                    ops_args.append('%s' % str(task_args[task_attr_name]))  # 这里应该对不同类型的参数名称做不同的参数处理，比如bool型，只有参数，没有值

            # print(ops_args)
            run_id = "run-" + str(task.pipeline.id) + "-" + str(task.id)
            if task.job_template.name == conf.get('CUSTOMIZE_JOB'):
                command = ['bash','-c',json.loads(task.args)['command']]
            if task.job_template.name == conf.get('PYTHON_JOB'):
                command = ['python', '-c', json.loads(task.args)['code']]
            can_customize_args = [conf.get('CUSTOMIZE_JOB'),conf.get('PYTHON_JOB')]
            args=None if task.job_template.name in can_customize_args else ops_args
            try:
                self.run_pod(
                    task=task,
                    k8s_client=k8s_client,
                    run_id=run_id,
                    namespace=new_namespace,
                    pod_name=pod_name,
                    image=json.loads(task.args).get('images',task.job_template.images.name),
                    working_dir=json.loads(task.args).get('workdir',task.job_template.workdir),
                    command=command,
                    args=args
                )
            except Exception as e:
                return render_template('close.html', data=str(e).replace('<br>','\n'))
                # return self.response(400, **{"status": 1, "result": {}, "message": str(e)})



        try_num = 5
        while (try_num > 0):
            pod = k8s_client.get_pods(namespace=new_namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                break
            try_num = try_num - 1
            time.sleep(2)
        if try_num == 0:
            message = __('启动时间过长，一分钟后重试')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        return redirect("/k8s/web/log/%s/%s/%s" % (task.pipeline.project.cluster['NAME'], new_namespace, pod_name))

    def delete_task_run(self, task):
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster.get('KUBECONFIG', ''))
        namespace = task.namespace
        # 删除运行时容器
        pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace, pod_name=pod['name'])
            run_id = pod['labels'].get('run-id', '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info=conf.get("CRD_INFO", {}), namespace=namespace, run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id": run_id})
                k8s_client.delete_service(namespace=namespace, labels={"run-id": run_id})
                time.sleep(2)

        # 删除debug容器
        pod_name = "debug-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace, pod_name=pod['name'])
            run_id = pod['labels'].get('run-id', '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info=conf.get("CRD_INFO", {}), namespace=namespace, run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id": run_id})
                time.sleep(2)

    @event_logger.log_this
    @expose_api(description="单个任务clear",url="/clear/<task_id>", methods=["GET", "POST"])
    def clear_task(self, task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()

        # 只有管理员和创建者可以debug
        if task.pipeline.created_by_fk!=g.user.id and not g.user.is_admin():
            # 模板创建者可以调试模板
            message = __('仅管理员或创建者，可清理该任务')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        self.delete_task_run(task)
        # flash(__("删除完成"), category='success')
        # self.update_redirect()
        return redirect('/pipeline_modelview/api/web/%s' % str(task.pipeline.id))

    @expose_api(description="单个任务查看日志",url="/log/<task_id>", methods=["GET", "POST"])
    def log_task(self, task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster.get('KUBECONFIG', ''))
        namespace = task.namespace
        running_pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = running_pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            return redirect("/k8s/web/log/%s/%s/%s" % (task.pipeline.project.cluster['NAME'], namespace, pod_name))

        flash(__("未检测到当前task正在运行的容器"), category='success')
        return redirect('/pipeline_modelview/api/web/%s' % str(task.pipeline.id))

# # 添加api
class Task_ModelView_Api(Task_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Task)
    route_base = '/task_modelview/api'
    list_columns = ['name', 'label', 'pipeline', 'job_template', 'volume_mount', 'node_selector', 'command',
                    'overwrite_entrypoint', 'working_dir', 'args', 'resource_memory', 'resource_cpu', 'resource_gpu', 'resource_rdma',
                    'timeout', 'retry', 'created_by', 'changed_by', 'created_on', 'changed_on', 'monitoring', 'expand']
    add_columns = ['name', 'label', 'job_template', 'pipeline', 'working_dir', 'command', 'args', 'volume_mount',
                   'node_selector', 'resource_memory', 'resource_cpu', 'resource_gpu', 'resource_rdma', 'timeout', 'retry', 'skip',
                   'expand']
    edit_columns = ['name', 'label', 'working_dir', 'command', 'args', 'volume_mount', 'resource_memory',
                    'resource_cpu', 'resource_gpu', 'resource_rdma', 'timeout', 'retry', 'skip', 'outputs', 'expand']
    show_columns = ['name', 'label', 'pipeline', 'job_template', 'volume_mount', 'node_selector', 'command',
                    'overwrite_entrypoint', 'working_dir', 'args', 'resource_memory', 'resource_cpu', 'resource_gpu', 'resource_rdma',
                    'timeout', 'retry', 'skip', 'created_by', 'changed_by', 'created_on', 'changed_on', 'monitoring',
                    'expand']


appbuilder.add_api(Task_ModelView_Api)
