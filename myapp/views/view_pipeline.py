import base64
import math
import traceback

from flask_appbuilder.baseviews import expose_api

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import logging
import urllib.parse
from sqlalchemy.exc import InvalidRequestError
from myapp.models.model_job import Job_Template
from myapp.models.model_job import Task, Pipeline, Workflow, RunHistory
from myapp.models.model_job import TaskTemplateType, LogicalNodeType
from myapp.models.model_team import Project
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import jsonify, Response, request, render_template
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder, db, event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Environment, BaseLoader, DebugUndefined,Undefined
import os
from wtforms.validators import DataRequired, Length, Regexp
from myapp.views.view_task import Task_ModelView_Api
from sqlalchemy import or_
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField, StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2ManyWidget, Select2Widget, BS3TextAreaFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget, MySelectMultipleField
from myapp.models.model_job import Repository
from myapp.utils.storage_volume import available_volume_choices, filter_selected_volume_mount, split_volume_mount
from myapp.utils.pipeline_priority import get_pipeline_priority_config
from myapp.utils.py import py_k8s
import re, copy
from kubernetes.client.models import (
    V1EnvVar, V1SecurityContext
)
from .baseApi import (
    MyappModelRestApi,
    send_file
)
from flask import (
    flash,
    g,
    make_response,
    redirect,
    request
)
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project
import pysnooper
from kubernetes import client
from .base import MyappFilter,json_response

from flask_appbuilder import expose
import datetime, time, json

conf = app.config


class Pipeline_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query.filter(or_(Pipeline.type==None,Pipeline.type==''))

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # logging.info(join_projects_id)
        return query.filter(or_(Pipeline.type==None,Pipeline.type=='')).filter(
            or_(
                self.model.project_id.in_(join_projects_id),
                # self.model.project.name.in_(['public'])
            )
        )



def make_workflow_yaml(pipeline,workflow_label,hubsecret_list,dag_templates,containers_templates,dbsession=db.session):
    name = pipeline.name+"-"+uuid.uuid4().hex[:4]
    workflow_label['workflow-name']=name
    priority_cfg = get_pipeline_priority_config(pipeline.priority)
    workflow_crd_json={
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            # "generateName": pipeline.name+"-",
            "annotations": {
                "name": pipeline.name,
                "description": pipeline.describe.encode("unicode_escape").decode('utf-8')
            },
            "name": name,
            "labels": workflow_label,
            "namespace": pipeline.project.pipeline_namespace
        },
        "spec": {
            "ttlStrategy": {
                "secondsAfterCompletion": 10800,  # 3个小时候自动删除
                "ttlSecondsAfterFinished": 10800,  # 3个小时候自动删除
            },
            "archiveLogs": True,  # 打包日志
            "entrypoint": pipeline.name,
            "priority": priority_cfg.get("argo_priority", 0),
            "templates": [
                             {
                                 "name": pipeline.name,
                                 "dag": {
                                     "tasks": dag_templates
                                 }
                             }
                         ] + containers_templates,
            "arguments": {
                "parameters": []
            },
            "serviceAccountName": "pipeline-runner",
            "parallelism": int(pipeline.parallelism),
            "imagePullSecrets": [
                {
                    "name": hubsecret
                } for hubsecret in hubsecret_list
            ]
        }
    }
    return workflow_crd_json


# 转化为worfklow的yaml
# @pysnooper.snoop()
def dag_to_pipeline(pipeline, dbsession, workflow_label=None, **kwargs):
    dag_json = pipeline.fix_dag_json(dbsession)
    pipeline.dag_json=dag_json
    dbsession.commit()
    dag = json.loads(dag_json)

    # 如果dag为空，就直接退出
    if not dag:
        return None, None

    all_tasks = {}
    for task_name in dag:
        # 使用临时连接，避免连接中断的问题
        # try:

        task = dbsession.query(Task).filter_by(name=task_name, pipeline_id=pipeline.id).first()
        if not task:
            raise MyappException('task %s not exist ' % task_name)
        all_tasks[task_name] = task

    try:
        pipeline_parameter = json.loads(pipeline.parameter or '{}')
    except Exception:
        pipeline_parameter = {}
    pipeline_volume_mount = filter_selected_volume_mount(
        pipeline.created_by,
        pipeline.project,
        pipeline_parameter.get('volume_mount', ''),
        namespace=conf.get('PIPELINE_NAMESPACE', 'pipeline')
    )

    template_kwargs=kwargs
    if 'execution_date' not in template_kwargs:
        template_kwargs['execution_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    priority_cfg = get_pipeline_priority_config(pipeline.priority)
    priority_class = priority_cfg.get("priority_class_name")

    # 渲染字符串模板变量
    # @pysnooper.snoop()
    def template_str(src_str):
        import re
        # 保护 Argo 模板变量 (以 tasks./workflow./steps./inputs./outputs./item./pod. 开头)，
        # 避免被 Jinja2 误渲染，但不影响 Flask 变量如 {{creator}}/{{uuid.uuid4().hex}}
        argo_pattern = re.compile(r'\{\{(?:tasks|workflow|steps|inputs|outputs|item|pod)\.[^}]*\}\}')
        argo_vars = argo_pattern.findall(src_str)
        for i, var in enumerate(argo_vars):
            src_str = src_str.replace(var, '__ARGOVAR{}__'.format(i))
        rtemplate = Environment(loader=BaseLoader, undefined=Undefined).from_string(src_str)
        des_str = rtemplate.render(creator=pipeline.created_by.username,
                                   datetime=datetime,
                                   runner=g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
                                   uuid=uuid,
                                   pipeline_id=pipeline.id,
                                   pipeline_name=pipeline.name,
                                   cluster_name=pipeline.project.cluster['NAME'],
                                   **template_kwargs
                                   )
        # 恢复 Argo 模板变量
        for i, var in enumerate(argo_vars):
            des_str = des_str.replace('__ARGOVAR{}__'.format(i), var)
        return des_str

    pipeline_global_env = template_str(pipeline.global_env.strip()) if pipeline.global_env else ''  # 优先渲染，不然里面如果有date就可能存在不一致的问题
    pipeline_global_env = [env.strip() for env in pipeline_global_env.split('\n') if '=' in env.strip()]

    # 系统级别环境变量
    global_envs = json.loads(template_str(json.dumps(conf.get('GLOBAL_ENV', {}), indent=4, ensure_ascii=False)))
    for env in pipeline_global_env:
        key, value = env[:env.index('=')], env[env.index('=') + 1:]
        global_envs[key] = value
    # 全局环境变量可以在任务的参数中引用
    for global_env in pipeline_global_env:
        key,value = global_env.split('=')[0],global_env.split('=')[1]
        if key not in kwargs:
            template_kwargs[key]=value

    def make_dag_template():
        # 先找到所有 branch 逻辑节点，记录其下游任务名
        branch_downstreams = set()
        for task_name in dag:
            task = all_tasks[task_name]
            if task.template_type == TaskTemplateType.LOGICAL and task.logical_type == LogicalNodeType.BRANCH:
                # 找所有直接下游任务
                for other_name in dag:
                    if task_name in dag[other_name].get('upstream', []):
                        branch_downstreams.add(other_name)

        dag_template = []
        for task_name in dag:
            task = all_tasks[task_name]
            template_temp = {
                "name": task_name,
                "template": task_name,
                "dependencies": dag[task_name].get('upstream', [])
            }
            # branch 节点的直接下游: 带上 when 条件
            if task_name in branch_downstreams:
                # 找到是哪个 branch 上游
                for up_name in dag[task_name].get('upstream', []):
                    up_task = all_tasks.get(up_name)
                    if up_task and up_task.template_type == TaskTemplateType.LOGICAL and up_task.logical_type == LogicalNodeType.BRANCH:
                        template_temp['when'] = '{{{{tasks.{}.outputs.parameters.branch_result}}}} == true'.format(up_name)
                        break

            # 逻辑节点: 根据 logical_type 添加 Argo 控制字段
            if task.template_type == TaskTemplateType.LOGICAL:
                logical_type = task.logical_type
                condition = task.logical_condition
                if logical_type == LogicalNodeType.LOOP and condition:
                    try:
                        parsed = json.loads(condition)
                        if isinstance(parsed, list):
                            template_temp['withParam'] = json.dumps(parsed)
                        else:
                            template_temp['withParam'] = condition
                    except (json.JSONDecodeError, TypeError):
                        template_temp['withParam'] = condition
                elif logical_type == LogicalNodeType.SUBPIPELINE:
                    subpipeline_name = task.logical_subpipeline_name
                    if subpipeline_name:
                        del template_temp['template']
                        template_temp['templateRef'] = {
                            'name': subpipeline_name,
                            'template': subpipeline_name
                        }
                # parallel / merge / end 不需要额外字段
            # 设置了跳过的话，在argo中设置跳过
            if task.skip:
                template_temp['when']='false'
            dag_template.append(template_temp)
        return dag_template

    # @pysnooper.snoop()
    def make_container_template(task_name,hubsecret_list=None):
        task = all_tasks[task_name]

        # 计算 volume mounts（逻辑节点也需要）
        k8s_volumes = []
        k8s_volume_mounts = []
        runtime_volume_mount = core.merge_volume_mount(task.volume_mount, pipeline_volume_mount)
        runtime_volume_mount = runtime_volume_mount.strip() if runtime_volume_mount else ''
        if runtime_volume_mount:
            try:
                k8s_volumes, k8s_volume_mounts = py_k8s.K8s.get_volume_mounts(runtime_volume_mount, pipeline.created_by.username)
            except Exception as e:
                print(e)

        # 逻辑节点: 生成轻量控制容器 + 跳过标准容器逻辑
        if task.template_type == TaskTemplateType.LOGICAL:
            logical_type = task.logical_type
            # subpipeline 使用 templateRef 引用外部工作流，无需生成容器模板
            if logical_type == LogicalNodeType.SUBPIPELINE and task.logical_subpipeline_name:
                logging.info(
                    "subpipeline node '%s': using templateRef to pipeline '%s'",
                    task.name, task.logical_subpipeline_name)
                return None

            # branch: Python 容器读取 JSON 文件 + 条件判断，输出结果参数
            if logical_type == LogicalNodeType.BRANCH:
                input_path = task.logical_input_path or '/tmp/result.json'
                condition = task.logical_condition or ''
                return _build_branch_template(task.name, input_path, condition, k8s_volume_mounts, k8s_volumes)

            template = {
                "name": task.name,
                "container": {
                    "name": task.name + "-" + uuid.uuid4().hex[:4],
                    "command": ["sh", "-c"],
                    "args": [f'echo "logic node: {logical_type}"'],
                    "image": conf.get('LOGIC_NODE_IMAGE', 'alpine:3.20'),
                    "imagePullPolicy": conf.get('IMAGE_PULL_POLICY', 'Always')
                }
            }
            if priority_class:
                template["priorityClassName"] = priority_class
            return template
        ops_args = []
        task_args = json.loads(task.args)
        for task_attr_name in task_args:
            # 布尔型只添加参数名
            if type(task_args[task_attr_name]) == bool:
                if task_args[task_attr_name]:
                    ops_args.append('%s' % str(task_attr_name))
            # 控制不添加
            elif not task_args[task_attr_name]:  # 如果参数值为空，则都不添加
                pass
            # json类型直接导入序列化以后的
            elif type(task_args[task_attr_name]) == dict or type(task_args[task_attr_name]) == list:
                ops_args.append('%s' % str(task_attr_name))
                args_values = json.dumps(task_args[task_attr_name], ensure_ascii=False)
                # args_values = template_str(args_values) if re.match('\{\{.*\}\}',args_values) else args_values
                ops_args.append('%s' % args_values)
            # # list类型，分多次导入,# list类型逗号分隔就好了
            # elif type(task_args[task_attr_name]) == list:
            #     for args_values in task_args[task_attr_name].split('\n'):
            #         ops_args.append('%s' % str(task_attr_name))
            #         # args_values = template_str(args_values) if re.match('\{\{.*\}\}',args_values) else args_values
            #         ops_args.append('%s' % args_values)
            # 其他的直接添加
            elif task_attr_name not in ['images','workdir']:
                ops_args.append('%s' % str(task_attr_name))
                args_values = task_args[task_attr_name]
                # args_values = template_str(args_values) if re.match('\{\{.*\}\}',args_values) else args_values
                ops_args.append('%s' % str(args_values))  # 这里应该对不同类型的参数名称做不同的参数处理，比如bool型，只有参数，没有值

        # 设置环境变量
        container_envs = []
        if task.job_template.env:
            envs = re.split('\r|\n', task.job_template.env)
            envs = [env.strip() for env in envs if env.strip()]
            for env in envs:
                env_key, env_value = env.split('=')[0], env.split('=')[1]
                container_envs.append((env_key, env_value))

        # 设置全局环境变量
        for global_env_key in global_envs:
            container_envs.append((global_env_key, global_envs[global_env_key]))

        # 设置task的默认环境变量
        gpu_num, _, gpu_resource_name = core.get_gpu(task.resource_gpu)
        container_envs.append(("KFJ_TASK_ID", str(task.id)))
        container_envs.append(("KFJ_TASK_NAME", str(task.name)))
        container_envs.append(("KFJ_TASK_NODE_SELECTOR", str(task.get_node_selector())))
        runtime_volume_mount = core.merge_volume_mount(task.volume_mount, pipeline_volume_mount)
        container_envs.append(("KFJ_TASK_VOLUME_MOUNT", str(runtime_volume_mount)))
        container_envs.append(("KFJ_TASK_IMAGES", str(task.job_template.images)))
        container_envs.append(("KFJ_TASK_RESOURCE_CPU", str(task.resource_cpu)))
        container_envs.append(("KFJ_TASK_RESOURCE_MEMORY", str(task.resource_memory)))
        container_envs.append(("KFJ_TASK_RESOURCE_GPU", str(task.resource_gpu)))
        container_envs.append(("KFJ_TASK_PROJECT_NAME", str(pipeline.project.name)))
        container_envs.append(("GPU_RESOURCE_NAME", gpu_resource_name))
        container_envs.append(("USERNAME", pipeline.created_by.username))
        container_envs.append(("IMAGE_PULL_POLICY", conf.get('IMAGE_PULL_POLICY','IfNotPresent')))
        if hubsecret_list:
            container_envs.append(("HUBSECRET", ','.join(hubsecret_list)))

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
            _run_id = global_envs.get('KFJ_RUN_ID', workflow_label.get('run-id', ''))
            _pipeline_name = pipeline.name or ''
            _task_name = task.name or ''
            _task_label = task.label or ''

            # MLOps 侧变量
            container_envs.append(("MLOPS_TRAINING_MONITOR_ENABLE", "true"))
            container_envs.append(("MLOPS_TRAINING_MONITOR_TYPE", "swanlab"))
            # 使用 K8s Pod 可访问的地址，不能用 Docker Compose 内部 DNS 名 myapp
            _register_url = conf.get('MLOPS_MONITOR_REGISTER_URL',
                                     'http://10.121.177.20:30080/training_monitor/api/register')
            container_envs.append(("MLOPS_MONITOR_REGISTER_URL", _register_url))
            container_envs.append(("MLOPS_PIPELINE_RUN_ID", _run_id))
            container_envs.append(("MLOPS_WORKFLOW_NAME", _pipeline_name))
            container_envs.append(("MLOPS_TASK_ID", str(task.id)))
            container_envs.append(("MLOPS_TASK_NAME", _task_label))
            container_envs.append(("MLOPS_NODE_NAME", _task_name))
            container_envs.append(("MLOPS_JOB_TEMPLATE_NAME", _job_template_name))

            # SwanLab 侧变量
            _swanlab_api_host = conf.get('SWANLAB_API_HOST', 'http://10.121.177.227:8000')
            _swanlab_web_host = conf.get('SWANLAB_WEB_HOST', 'http://10.121.177.227:8000')
            _swanlab_logdir = conf.get('SWANLAB_LOGDIR', '') or ''
            _swanlab_proj_name = (_swanlab_task_args.get('swanlab_project') or
                                conf.get('SWANLAB_PROJ_NAME', 'mlops-training'))
            _swanlab_workspace = (_swanlab_task_args.get('swanlab_workspace') or
                                  conf.get('SWANLAB_WORKSPACE', 'haimian_baobao'))

            # swanlab_mode 优先级（修复：不能用 or-chain，因为 conf SWANLAB_MODE=local 是 truthy）:
            #   1. Task arg 显式指定
            #   2. lightgbm / hyperparam-search / hyperparam-search-nni → cloud
            #   3. config.py SWANLAB_MODE（全局默认，当前为 local）
            #   4. 兜底 local
            _cloud_default_templates = ('hyperparam-search', 'hyperparam-search-nni', 'lightgbm', 'gbdt', 'model-distillation', 'msswift', 'llama-factory')
            _user_mode = (_swanlab_task_args.get('swanlab_mode') or '').strip().lower()
            if _user_mode:
                # Normalize "online" to "cloud" (official SDK name vs MLOps convention)
                _swanlab_mode = 'cloud' if _user_mode == 'online' else _user_mode
            elif _job_template_name in _cloud_default_templates:
                _swanlab_mode = 'cloud'
            else:
                _swanlab_mode = conf.get('SWANLAB_MODE', 'local') or 'local'

            container_envs.append(("SWANLAB_MODE", _swanlab_mode))
            container_envs.append(("SWANLAB_API_HOST", _swanlab_api_host))
            container_envs.append(("SWANLAB_WEB_HOST", _swanlab_web_host))
            container_envs.append(("SWANLAB_EXP_NAME", f"{_pipeline_name}-{_task_name}-{_run_id[:8]}"))
            container_envs.append(("SWANLAB_GROUP", _run_id))
            container_envs.append(("SWANLAB_TAGS", "mlops,training"))
            container_envs.append(("SWANLAB_PROBE_HARDWARE", "true"))
            container_envs.append(("SWANLAB_PROBE_MONITOR", "true"))
            container_envs.append(("SWANLAB_PROBE_MONITOR_INTERVAL", "10"))

            if _swanlab_mode == "cloud":
                # ---- Cloud / Self-hosted Online 模式 ----
                container_envs.append(("SWANLAB_PROJ_NAME", _swanlab_proj_name))
                container_envs.append(("SWANLAB_WORKSPACE", _swanlab_workspace))
                # API Key 通过 Kubernetes Secret 注入
                container_envs.append({
                    "name": "SWANLAB_API_KEY",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "swanlab-secret",
                            "key": "SWANLAB_API_KEY"
                        }
                    }
                })
                # Cloud 模式不挂载 swanlab PVC，不设 SWANLAB_LOGDIR
                print(f"[swanlab inject] task={_task_name}, template={_job_template_name}, "
                      f"mode=cloud, project={_swanlab_proj_name}, workspace={_swanlab_workspace}, "
                      f"secret=yes, logdir=no, pvc=no, register_url={_register_url}")
            else:
                # ---- Local / Watch 模式（保持现有逻辑不变） ----
                container_envs.append(("SWANLAB_PROJ_NAME", "mlops-training"))
                if _swanlab_logdir:
                    container_envs.append(("SWANLAB_LOGDIR", _swanlab_logdir))

                # 注入 SwanLab 监控共享卷（仅 local 模式）
                k8s_volume_mounts.append({
                    "name": "swanlab-storage",
                    "mountPath": "/mnt/storage/swanlab",
                })
                k8s_volumes.append({
                    "name": "swanlab-storage",
                    "persistentVolumeClaim": {"claimName": "swanlab"},
                })
                print(f"[swanlab inject] task={_task_name}, template={_job_template_name}, "
                      f"mode=local, project=mlops-training, secret=no, logdir=yes, pvc=yes, "
                      f"register_url={_register_url}")


        # 创建工作目录
        working_dir = None
        if task.job_template.workdir and task.job_template.workdir.strip():
            working_dir = task.job_template.workdir.strip()
        if task.working_dir and task.working_dir.strip():
            working_dir = task.working_dir.strip()

        # 配置启动命令
        task_command = ''

        if task.command:
            commands = re.split('\r|\n', task.command)
            commands = [command.strip() for command in commands if command.strip()]
            if task_command:
                task_command += " && " + " && ".join(commands)
            else:
                task_command += " && ".join(commands)

        job_template_entrypoint = task.job_template.entrypoint.strip() if task.job_template.entrypoint else ''

        command = None
        if job_template_entrypoint:
            command = job_template_entrypoint

        if task_command:
            command = task_command

        images = task.job_template.images.name
        command = command.split(' ') if command else []
        command = [com for com in command if com]
        arguments = ops_args
        file_outputs = json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None
        # 构建 Argo output parameters（逻辑节点跳过）
        output_parameters = []
        if file_outputs and task.template_type not in (TaskTemplateType.LOGICAL,):
            for param_name, file_path in file_outputs.items():
                if file_path:
                    output_parameters.append({
                        "name": param_name,
                        "valueFrom": {"path": file_path}
                    })

        # 如果模板配置了images参数，那直接用模板的这个参数
        if json.loads(task.args).get('images',''):
            images = json.loads(task.args).get('images')

        # 自定义节点 (CUSTOMIZE_JOB)
        if task.template_type == TaskTemplateType.CUSTOMIZE:
            working_dir = json.loads(task.args).get('workdir')
            command = ['bash', '-c', json.loads(task.args).get('command')]
            arguments = []

        # Python 节点 (PYTHON_JOB): 简单 python -c 执行
        if task.template_type == TaskTemplateType.PYTHON:
            command = ['python', '-c', json.loads(task.args).get('code', '')]
            arguments = None

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
                if arguments is None:
                    arguments = []
                _cmd_str = ' '.join(command) if command else ''
                _arg_str = ' '.join(arguments) if arguments else ''
                _combined = f"{_cmd_str} {_arg_str}"
                _skipped = []
                for _ea in _extra_args:
                    if _ea.lstrip('-') in _combined:
                        print(f"[swanlab] WARNING: {_ea} already in command, skipping")
                        _skipped.append(_ea)
                _extra_args = [a for a in _extra_args if a not in _skipped]
                if _extra_args:
                    print(f"[swanlab] auto-appending framework args: {_extra_args}")
                    arguments.extend(_extra_args)

            # Wrapper 包装：将原始命令包装进 swanlab_framework_wrapper，负责 monitor 注册和状态同步
            if (_framework_type in _SUPPORTED_FRAMEWORK_TYPES
                    and _job_template_name not in _ML_OPERATORS
                    and _inject_swanlab):
                # 合并 command + arguments 作为 wrapper 的子命令
                _merged_cmd = list(command) if command else []
                if arguments:
                    _merged_cmd.extend(arguments)
                if _merged_cmd:
                    command = ['python3', '/app/common/swanlab_framework_wrapper.py', '--']
                    arguments = _merged_cmd
                    print(f"[swanlab] wrapped command with swanlab_framework_wrapper")

        # 添加用户自定义挂载（逻辑节点已在前面处理）

        # 添加node selector
        nodeSelector, nodeAffinity = core.get_node_selector(task.get_node_selector())

        # 添加pod label
        pod_label = {
            "pipeline-id": str(pipeline.id),
            "pipeline-name": str(pipeline.name),
            "app":str(pipeline.name),
            "task-id": str(task.id),
            "task-name": str(task.name),
            "run-id": global_envs.get('KFJ_RUN_ID', ''),
            'run-username': g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
            'pipeline-username': pipeline.created_by.username

        }
        pod_annotations = {
            'project': pipeline.project.name,
            'pipeline': pipeline.describe,
            "task": task.label,
            'job-template': task.job_template.describe
        }

        # 设置资源限制
        resource_cpu = task.job_template.get_env('TASK_RESOURCE_CPU') if task.job_template.get_env('TASK_RESOURCE_CPU') else task.resource_cpu
        resource_gpu = task.job_template.get_env('TASK_RESOURCE_GPU') if task.job_template.get_env('TASK_RESOURCE_GPU') else task.resource_gpu

        resource_memory = task.job_template.get_env('TASK_RESOURCE_MEMORY') if task.job_template.get_env('TASK_RESOURCE_MEMORY') else task.resource_memory

        resources_requests = resources_limits = {}

        if resource_memory:
            if not '~' in resource_memory:
                resources_requests['memory'] = resource_memory
                resources_limits['memory'] = resource_memory
            else:
                resources_requests['memory'] = resource_memory.split("~")[0]
                resources_limits['memory'] = resource_memory.split("~")[1]

        if resource_cpu:
            if not '~' in resource_cpu:
                resources_requests['cpu'] = resource_cpu
                resources_limits['cpu'] = resource_cpu

            else:
                resources_requests['cpu'] = resource_cpu.split("~")[0]
                resources_limits['cpu'] = resource_cpu.split("~")[1]

        if resource_gpu:

            hami_gpu = core.get_hami_gpu(resource_gpu)
            gpu_num, gpu_type, gpu_resource_name = core.get_gpu(resource_gpu)
            if gpu_type and gpu_type.strip():
                nodeSelector['gpu-type'] = gpu_type.strip().upper()

            if hami_gpu.get('enabled'):
                nodeSelector.pop('cpu', None)
                for selector_key, selector_value in conf.get('HAMI_NODE_SELECTOR', {}).items():
                    nodeSelector[selector_key] = selector_value
                resources_requests[hami_gpu['resource_name']] = str(hami_gpu['gpu'])
                resources_limits[hami_gpu['resource_name']] = str(hami_gpu['gpu'])
                if hami_gpu.get('gpumem'):
                    resources_requests[hami_gpu['memory_resource_name']] = str(hami_gpu['gpumem'])
                    resources_limits[hami_gpu['memory_resource_name']] = str(hami_gpu['gpumem'])
                resources_requests[hami_gpu['core_resource_name']] = str(hami_gpu['gpucores'])
                resources_limits[hami_gpu['core_resource_name']] = str(hami_gpu['gpucores'])

            # 整卡占用
            if isinstance(gpu_num, (int, float)) and gpu_num >= 1 and not hami_gpu.get('enabled'):
                nodeSelector.pop('cpu', None)
                for selector_key, selector_value in conf.get('NVIDIA_GPU_NODE_SELECTOR', {'gpu': 'true'}).items():
                    nodeSelector[selector_key] = selector_value
                resources_requests[gpu_resource_name] = str(int(gpu_num))
                resources_limits[gpu_resource_name] = str(int(gpu_num))

            if 0 == gpu_num:
                # 没要gpu的容器，就要加上可视gpu为空，不然gpu镜像能看到和使用所有gpu
                for gpu_alias in conf.get('GPU_NONE', {}):
                    container_envs.append((conf.get('GPU_NONE',{})[gpu_alias][0], conf.get('GPU_NONE',{})[gpu_alias][1]))
        # 配置host
        host_aliases = {}

        global_host_aliases = conf.get('HOSTALIASES', '')
        # global_host_aliases = ''
        if task_temp.job_template.host_aliases:
            global_host_aliases += "\n" + task_temp.job_template.host_aliases
        if global_host_aliases:
            host_aliases_list = re.split('\r|\n', global_host_aliases)
            host_aliases_list = [host.strip() for host in host_aliases_list if host.strip()]
            for row in host_aliases_list:
                hosts = row.strip().split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    host_aliases[hosts[1]] = hosts[0]

        if task.skip:
            command = ["echo", "skip"]
            arguments = None
            resources_requests = None
            resources_limits = None

        # 构建 outputs.artifacts：将 task.outputs 中定义的输出文件声明为 Argo artifact
        # PVC 挂载路径 Argo 不会捕获，统一复制到 /tmp/cube_outputs/ 再让 Argo 抓
        artifacts = []
        if file_outputs:
            copy_cmds = ['mkdir -p /tmp/cube_outputs']
            for artifact_name, file_path in file_outputs.items():
                tmp_path = f'/tmp/cube_outputs/{artifact_name}'
                copy_cmds.append(f'cp -r {file_path} {tmp_path}')
                artifacts.append({
                    "name": artifact_name,
                    "path": tmp_path,
                })
            # 把原命令和复制命令拼接成 bash -c 单行脚本
            primary = ' '.join(command) if command else ''
            if arguments:
                primary += ' ' + ' '.join(arguments)
            if primary:
                primary += ' && '
            command = ['bash', '-c', primary + ' && '.join(copy_cmds)]
            arguments = []

        task_template = {
            "name": task.name,
            "outputs": {
                "parameters": output_parameters,
                "artifacts": []
            },
            "container": {
                "name": task.name + "-" + uuid.uuid4().hex[:4],
                "ports": [],

                "command": command,
                "args": arguments,
                "env": [
                    item if isinstance(item, dict) else {"name": item[0], "value": item[1]}
                    for item in container_envs
                ],
                "image": images,
                "resources": {
                    "limits": resources_limits,
                    "requests": resources_requests
                },
                "volumeMounts": k8s_volume_mounts,
                "workingDir": working_dir,
                "imagePullPolicy": conf.get('IMAGE_PULL_POLICY', 'IfNotPresent')
            },
            "nodeSelector": nodeSelector,
            "securityContext": {
                "privileged": True if task.job_template.privileged else False
            },
            "affinity": {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "podAffinityTerm": {
                                "labelSelector": {
                                    "matchLabels": {
                                        "pipeline-id": str(pipeline.id)
                                    }
                                },
                                "topologyKey": "kubernetes.io/hostname"
                            },
                            "weight": 80
                        }
                    ]
                }
            },
            "metadata": {
                "labels": pod_label,
                "annotations": pod_annotations
            },
            "retryStrategy": {
                "limit": int(task.retry)
            } if task.retry else None,
            "volumes": k8s_volumes,
            "hostAliases": [
                {
                    "hostnames": [hostname],
                    "ip": host_aliases[hostname]
                } for hostname in host_aliases
            ],
            "activeDeadlineSeconds": task.timeout if task.timeout else None
        }
        if core.get_hami_gpu(resource_gpu).get('enabled'):
            task_template["schedulerName"] = conf.get('GPU_SCHEDULERNAME', 'hami-scheduler')
        if priority_class:
            task_template["priorityClassName"] = priority_class

        # 统一添加一些固定环境变量，比如hostip，podip等
        task_template['container']['env'].append({
            "name":"K8S_NODE_NAME",
            "valueFrom":{
                "fieldRef":{
                    "apiVersion":"v1",
                    "fieldPath":"spec.nodeName"
                }
            }
        })
        task_template['container']['env'].append({
            "name": "K8S_POD_IP",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "status.podIP"
                }
            }
        })
        task_template['container']['env'].append({
            "name": "K8S_HOST_IP",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "status.hostIP"
                }
            }
        })
        task_template['container']['env'].append({
            "name": "K8S_POD_NAME",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "metadata.name"
                }
            }
        })


        return task_template

    def _build_branch_template(task_name, input_path, condition, volume_mounts=None, volumes=None):
        """构建分支逻辑节点的容器模板：读取JSON文件 → 条件判断 → 输出Argo参数"""
        import re
        # 解析 condition: "accuracy > 0.9" → field=accuracy, op=>, value=0.9
        _cond_match = re.match(r'^\s*(\S+)\s*(>=|<=|!=|==|>|<)\s*(.+)\s*$', condition)
        if not _cond_match:
            # 无法解析，回退为 alpine echo
            template = {
                "name": task_name,
                "container": {
                    "name": task_name + "-" + uuid.uuid4().hex[:4],
                    "command": ["sh", "-c"],
                    "args": ['echo "branch condition parse error: {}"'.format(condition)],
                    "image": "alpine:3.20",
                    "imagePullPolicy": conf.get('IMAGE_PULL_POLICY', 'Always')
                }
            }
            if priority_class:
                template["priorityClassName"] = priority_class
            return template

        field, op, value = _cond_match.group(1), _cond_match.group(2), _cond_match.group(3).strip()

        # 构造 Python 判断脚本：全流程异常保护，始终写入 branch_result
        script_lines = [
            '#!/bin/sh',
            "python << 'PYEOF'",
            'import json, sys, os, traceback',
            '',
            'def write_result(value):',
            '    """无论如何都写入 Argo 输出参数"""',
            '    argo_dir = "/var/run/argo/outputs/parameters"',
            '    try:',
            '        if not os.path.exists(argo_dir):',
            '            os.makedirs(argo_dir)',
            '    except:',
            '        pass',
            '    try:',
            '        with open(os.path.join(argo_dir, "branch_result"), "w") as f:',
            '            f.write(value)',
            '    except Exception as e:',
            '        print("FATAL: cannot write branch_result: " + str(e))',
            '',
            'try:',
            "    with open('{}') as f:".format(input_path),
            "        data = json.load(f)",
            '',
            "    # 提取字段值",
            "    val = data",
            "    for key in '{}'.split('.'):".format(field),
            "        if isinstance(val, dict):",
            "            val = val.get(key)",
            "        else:",
            "            val = None",
            "            break",
            "        if val is None:",
            "            break",
            '',
            "    if val is None:",
            "        print('ERROR: field {} not found in JSON')".format(field),
            "        write_result('error')",
            "        sys.exit(0)",
            '',
            "    # 比较判断",
            "    result = False",
            "    try:",
            "        result = float(val) {} {}".format(op, value),
            "    except Exception:",
            "        result = str(val) {} str({})".format(op, value),
            '',
            '    print("branch: " + str(val) + " ' + op + ' ' + value + ' -> " + str(result))',
            "    write_result('true' if result else 'false')",
            '',
            'except Exception as e:',
            "    print('ERROR: ' + str(e))",
            '    traceback.print_exc()',
            "    write_result('error')",
            'PYEOF'
        ]
        script = '\n'.join(script_lines)

        template = {
            "name": task_name,
            "outputs": {
                "parameters": [
                    {"name": "branch_result", "valueFrom": {"path": "branch_result"}}
                ]
            },
            "container": {
                "name": task_name + "-" + uuid.uuid4().hex[:4],
                "command": ["sh", "-c"],
                "args": [script],
                "image": conf.get('LOGIC_NODE_IMAGE', 'python:3.9-alpine'),
                "imagePullPolicy": conf.get('IMAGE_PULL_POLICY', 'Always'),
                "volumeMounts": volume_mounts or []
            },
            "volumes": volumes or []
        }
        if priority_class:
            template["priorityClassName"] = priority_class
        return template

    # 添加个人创建的所有仓库秘钥
    image_pull_secrets = conf.get('HUBSECRET', [])
    user_repositorys = dbsession.query(Repository).filter(Repository.created_by_fk == pipeline.created_by.id).all()
    hubsecret_list = list(set(image_pull_secrets + [rep.hubsecret for rep in user_repositorys]))

    # 配置拉取秘钥
    for task_name in all_tasks:
        # 配置拉取秘钥。本来在contain里面，workflow在外面
        task_temp = all_tasks[task_name]
        if task_temp.job_template.images.repository.hubsecret:
            hubsecret = task_temp.job_template.images.repository.hubsecret
            if hubsecret not in hubsecret_list:
                hubsecret_list.append(hubsecret)

    hubsecret_list = list(set(hubsecret_list))

    # 设置workflow标签
    if not workflow_label:
        workflow_label = {}

    workflow_label['run-username'] = g.user.username if g and g.user and g.user.username else pipeline.created_by.username
    workflow_label['pipeline-username'] = pipeline.created_by.username
    workflow_label['save-time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    workflow_label['pipeline-id'] = str(pipeline.id)
    workflow_label['pipeline-name'] = str(pipeline.name)
    workflow_label['pipeline-priority'] = (pipeline.priority or conf.get("PIPELINE_PRIORITY_DEFAULT", "high")).lower()
    workflow_label['app'] = str(pipeline.name)
    workflow_label['run-id'] = global_envs.get('KFJ_RUN_ID', '')  # 以此来绑定运行时id，不能用kfp的run—id。那个是传到kfp以后才产生的。
    workflow_label['cluster'] = pipeline.project.cluster['NAME']

    containers_template = []
    for task_name in dag:
        ct = make_container_template(task_name=task_name, hubsecret_list=hubsecret_list)
        if ct is not None:
            containers_template.append(ct)

    workflow_json = make_workflow_yaml(pipeline=pipeline, workflow_label=workflow_label, hubsecret_list=hubsecret_list, dag_templates=make_dag_template(), containers_templates=containers_template,dbsession=dbsession)
    # 先这是某个模板变量不进行渲染，一直向后传递到argo
    pipeline_file = json.dumps(workflow_json,ensure_ascii=False,indent=4)
    # print(pipeline_file)
    pipeline_file = template_str(pipeline_file)

    return pipeline_file, workflow_label['run-id']


# @pysnooper.snoop(watch_explode=())
def run_pipeline(pipeline, workflow_json):
    cluster = pipeline.project.cluster
    crd_name = workflow_json.get('metadata', {}).get('name', '')
    from myapp.utils.py.py_k8s import K8s
    k8s_client = K8s(cluster.get('KUBECONFIG', ''))
    namespace = workflow_json.get('metadata', {}).get("namespace", pipeline.project.pipeline_namespace)
    crd_info = conf.get('CRD_INFO', {}).get('workflow', {})
    try:
        workflow_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
        if workflow_obj:
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
            time.sleep(1)

        crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=workflow_json)
        pipeline.namespace=namespace
        db.session.commit()
    except Exception as e:
        print(e)

    return crd_name


class Pipeline_ModelView_Base():
    label_title = _('任务流')
    datamodel = SQLAInterface(Pipeline)

    base_permissions = ['can_show', 'can_edit', 'can_list', 'can_delete', 'can_add']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']

    list_columns = ['id', 'project', 'pipeline_url', 'creator', 'modified']
    cols_width = {
        "id": {"type": "ellip2", "width": 100},
        "project": {"type": "ellip2", "width": 200},
        "pipeline_url": {"type": "ellip2", "width": 400},
        "modified": {"type": "ellip2", "width": 150}
    }
    spec_label_columns={
        "parameter":_("后端扩展"),
        "expand":_("前端扩展")
    }
    add_columns = ['project', 'name', 'describe', 'parameter']
    edit_columns = ['project', 'name', 'describe', 'schedule_type', 'cron_time', 'depends_on_past', 'max_active_runs',
                    'expired_limit', 'parallelism', 'priority', 'global_env', 'parameter', 'alert_status', 'alert_user',
                    'cronjob_start_time']
    show_columns = ['project', 'name', 'describe', 'schedule_type', 'cron_time', 'depends_on_past', 'max_active_runs',
                    'expired_limit', 'parallelism', 'priority', 'global_env', 'dag_json', 'pipeline_file', 'pipeline_argo_id',
                    'run_id', 'created_by', 'changed_by', 'created_on', 'changed_on', 'expand',
                    'parameter', 'alert_status', 'alert_user', 'cronjob_start_time']
    # show_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    search_columns = ['id', 'created_by', 'name', 'describe', 'schedule_type', 'project']

    base_filters = [["id", Pipeline_Filter, lambda: []]]
    conv = GeneralModelConverter(datamodel)

    add_form_extra_fields = {

        "name": StringField(
            _('名称'),
            description= _("英文名(小写字母、数字、- 组成)，最长50个字符"),
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54), DataRequired()]
        ),
        "describe": StringField(
            _("描述"),
            description="",
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "project": QuerySelectField(
            _('项目组'),
            query_factory=filter_join_org_project,
            allow_blank=True,
            widget=Select2Widget()
        ),
        "dag_json": StringField(
            _('上下游关系'),
            default='{}',
            description=_("任务的上下游关系，目前不需要手动修改"),
            widget=MyBS3TextAreaFieldWidget(rows=10,is_json=True),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "namespace": StringField(
            _('命名空间'),
            description= _("部署task所在的命名空间(目前无需填写)"),
            default='pipeline',
            widget=BS3TextFieldWidget()
        ),
        "node_selector": StringField(
            _('机器选择'),
            description= _("部署task所在的机器(目前无需填写)"),
            widget=BS3TextFieldWidget(),
            default=datamodel.obj.node_selector.default.arg
        ),
        "image_pull_policy": SelectField(
            _('拉取策略'),
            description= _("镜像拉取策略(always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)"),
            widget=Select2Widget(),
            default='Always',
            choices=[['Always', 'Always'], ['IfNotPresent', 'IfNotPresent']]
        ),

        "depends_on_past": BooleanField(
            _('过往依赖'),
            description= _("任务运行是否依赖上一次的示例状态"),
            default=True
        ),
        "max_active_runs": IntegerField(
            _('最大激活数'),
            description= _("当前pipeline可同时运行的任务流实例数目"),
            widget=BS3TextFieldWidget(),
            default=1,
            validators=[DataRequired()]
        ),
        "expired_limit": IntegerField(
            _('过期保留数'),
            description= _("定时调度最新实例限制数目，0表示不限制"),
            widget=BS3TextFieldWidget(),
            default=1,
            validators=[DataRequired()]
        ),
        "parallelism": IntegerField(
            _('并发数'),
            description= _("一个任务流实例中可同时运行的task数目"),
            widget=BS3TextFieldWidget(),
            default=3,
            validators=[DataRequired()]
        ),
        "priority": SelectField(
            _('调度优先级'),
            widget=Select2Widget(),
            default=conf.get('PIPELINE_PRIORITY_DEFAULT', 'high'),
            description=_('资源紧张时优先调度高优先级任务流'),
            choices=[['high', 'high'], ['low', 'low']],
            validators=[DataRequired()]
        ),
        "volume_mount": MySelectMultipleField(
            label=_('挂载卷'),
            default='',
            description=_('选择项目默认挂载卷或已同步的存储资源，保存后会自动挂载到流水线内每个task'),
            widget=MySelect2Widget(multiple=True),
            choices=[],
        ),
        "global_env": StringField(
            _('全局环境变量'),
            description= _("公共环境变量会以环境变量的形式传递给每个task，可以配置多个公共环境变量，每行一个，支持datetime/creator/runner/uuid/pipeline_id等变量 例如：USERNAME={{creator}}"),
            widget=BS3TextAreaFieldWidget()
        ),
        "schedule_type": SelectField(
            _('调度类型'),
            default='once',
            description= _("调度类型，once仅运行一次，crontab周期运行，crontab配置保存一个小时候后才生效"),
            widget=Select2Widget(),
            choices=[['once', 'once'], ['crontab', 'crontab']]
        ),
        "cron_time": StringField(
            _('调度周期'),
            description= _("周期任务的时间设定 * * * * * 表示为 minute hour day month week"),
            widget=BS3TextFieldWidget()
        ),
        "alert_status": MySelectMultipleField(
            label= _('监听状态'),
            widget=Select2ManyWidget(),
            choices=[[x, x] for x in
                     ['Created', 'Pending', 'Running', 'Succeeded', 'Failed', 'Unknown', 'Waiting', 'Terminated']],
            description= _("选择通知状态"),
            validators=[Length(0, 400), ]
        ),
        "alert_user": StringField(
            label= _('报警用户'),
            widget=BS3TextFieldWidget(),
            description= _("选择通知用户，每个用户使用逗号分隔")
        ),
        "parameter": StringField(
            _('后端扩展'),
            default='{}',
            description=_('后端扩展参数，用于配置是否为demo或固化任务流'),
            widget=MyBS3TextAreaFieldWidget(rows=10, is_json=True),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "expand": StringField(
            _('前端扩展'),
            default='{}',
            description=_('前端扩展参数，前端用于记录任务流节点位置和连线关系'),
            widget=MyBS3TextAreaFieldWidget(rows=10, is_json=True),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )

    }

    edit_form_extra_fields = add_form_extra_fields

    expand_columns = {
        "parameter": {
            "volume_mount": MySelectMultipleField(
                label=_('挂载卷'),
                default='',
                description=_('选择项目默认挂载卷或已同步的存储资源，保存后会自动挂载到流水线内每个 task'),
                widget=MySelect2Widget(multiple=True),
                choices=[],
            )
        }
    }


    related_views = [Task_ModelView_Api, ]

    def _pipeline_parameter(self, pipeline=None):
        if pipeline and pipeline.parameter:
            try:
                return json.loads(pipeline.parameter)
            except Exception:
                return {}
        return {}

    def _pipeline_volume_mount(self, pipeline=None):
        # 先尝试从 form 提交的 transient 属性读取，再尝试从 parameter JSON 读取
        if pipeline and hasattr(pipeline, 'volume_mount') and getattr(pipeline, 'volume_mount'):
            return getattr(pipeline, 'volume_mount')
        return ','.join(split_volume_mount(self._pipeline_parameter(pipeline).get('volume_mount', '')))

    def _pipeline_storage_namespace(self):
        return conf.get('PIPELINE_NAMESPACE', 'pipeline')

    def _save_pipeline_volume_mount(self, pipeline, selected_volume_mount):
        parameter = self._pipeline_parameter(pipeline)
        selected_volume_mount = filter_selected_volume_mount(
            g.user,
            pipeline.project,
            selected_volume_mount,
            namespace=self._pipeline_storage_namespace()
        )
        if selected_volume_mount:
            parameter['volume_mount'] = selected_volume_mount
        else:
            parameter.pop('volume_mount', None)
        pipeline.parameter = json.dumps(parameter, indent=4, ensure_ascii=False)
        pipeline_volume_mount = core.merge_volume_mount(pipeline.project.volume_mount, selected_volume_mount)
        for task in pipeline.get_tasks(db.session):
            task.volume_mount = core.merge_volume_mount(pipeline_volume_mount, task.job_template.volume_mount if task.job_template else '')

    def _merge_volume_mount_req(self, req_json, src_item=None):
        if not req_json:
            return req_json
        if 'volume_mount' not in req_json:
            return req_json
        req_json['volume_mount'] = ','.join(split_volume_mount(req_json.get('volume_mount')))
        return req_json

    def _build_volume_mount_field(self, current_volume_mount='', project=None):
        return MySelectMultipleField(
            label=_('挂载卷'),
            default=current_volume_mount,
            description=_('选择项目默认挂载卷或已同步的存储资源，保存后会自动挂载到流水线内每个 task'),
            widget=MySelect2Widget(multiple=True),
            choices=available_volume_choices(
                g.user,
                project=project,
                namespace=self._pipeline_storage_namespace(),
                current_volume_mount=current_volume_mount
            ),
        )

    def _set_pipeline_volume_mount_field(self, pipeline=None, project=None):
        project = project or (pipeline.project if pipeline else None)
        current_volume_mount = self._pipeline_volume_mount(pipeline)
        if not current_volume_mount and project:
            current_volume_mount = project.volume_mount
        field = self._build_volume_mount_field(current_volume_mount, project)
        self.expand_columns['parameter']['volume_mount'] = field
        self.edit_form_extra_fields['volume_mount'] = field
        self.add_form_extra_fields['volume_mount'] = field

    def set_columns_related(self, exist_add_args, response_add_columns):
        if 'volume_mount' not in response_add_columns:
            return
        project_value = exist_add_args.get('project') or exist_add_args.get('project_id') or {}
        if isinstance(project_value, dict):
            project_value = project_value.get('id') or project_value.get('value')
        project = db.session.query(Project).filter_by(id=int(project_value)).first() if str(project_value).isdigit() else None
        parameter = exist_add_args.get('parameter') or {}
        if isinstance(parameter, str):
            try:
                parameter = json.loads(parameter or '{}')
            except Exception:
                parameter = {}
        current_volume_mount = exist_add_args.get('volume_mount') or parameter.get('volume_mount') or (project.volume_mount if project else '')
        choices = available_volume_choices(
            g.user,
            project=project,
            namespace=self._pipeline_storage_namespace(),
            current_volume_mount=current_volume_mount
        )
        response_add_columns['volume_mount'].update({
            "label": _('挂载卷'),
            "description": _('选择项目默认挂载卷或已同步的存储资源，保存后会自动挂载到流水线内每个 task'),
            "type": "Select",
            "ui-type": "select2",
            "default": split_volume_mount(current_volume_mount),
            "choices": choices,
            "values": [{"id": choice[0], "value": choice[1]} for choice in choices],
        })

    def delete_task_run(self, task):
        try:
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
        except Exception as e:
            print(e)

    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        if g.user and g.user.is_admin():
            return True
        if g.user and g.user.username and hasattr(item, 'created_by'):
            if g.user.username == item.created_by.username:
                return True
        # flash('just creator can edit/delete ', 'warning')
        return False

    check_delete_permission = check_edit_permission

    # 验证args参数,并自动排版dag_json
    # @pysnooper.snoop(watch_explode=('item'))
    def pipeline_args_check(self, item):
        core.validate_str(item.name, 'name')
        if not item.dag_json:
            item.dag_json = '{}'
        core.validate_json(item.dag_json)

        # 校验task的关系，没有闭环，并且顺序要对。没有配置的，自动没有上游，独立
        # @pysnooper.snoop()
        def order_by_upstream(dag_json):
            order_dag = {}
            tasks_name = list(dag_json.keys())  # 如果没有配全的话，可能只有局部的task
            i = 0
            while tasks_name:
                i += 1
                if i > 100:  # 不会有100个依赖关系
                    break
                for task_name in tasks_name:
                    # 没有上游的情况
                    if not dag_json[task_name]:
                        order_dag[task_name] = {}
                        tasks_name.remove(task_name)
                        continue
                    # 没有上游的情况
                    elif 'upstream' not in dag_json[task_name] or not dag_json[task_name]['upstream']:
                        order_dag[task_name] = {}
                        tasks_name.remove(task_name)
                        continue
                    # 如果有上游依赖的话，先看上游任务是否已经加到里面了。
                    upstream_all_ready = True
                    for upstream_task_name in dag_json[task_name]['upstream']:
                        if upstream_task_name not in order_dag:
                            upstream_all_ready = False
                    if upstream_all_ready:
                        order_dag[task_name] = dag_json[task_name]
                        tasks_name.remove(task_name)
            if list(dag_json.keys()).sort() != list(order_dag.keys()).sort():
                message = __('dag pipeline 存在循环或未知上游')
                flash(message, category='warning')
                raise MyappException(message)
            return order_dag

        # 配置上缺少的默认上游
        dag_json = json.loads(item.dag_json)
        tasks = item.get_tasks(db.session)
        if tasks and dag_json:
            for task in tasks:
                if task.name not in dag_json:
                    dag_json[task.name] = {
                        "upstream": []
                    }
        item.dag_json = json.dumps(order_by_upstream(copy.deepcopy(dag_json)), ensure_ascii=False, indent=4)

        # # 生成workflow，如果有id， 校验的时候，先不生成file
        # if item.id and item.get_tasks():
        #     item.pipeline_file,item.run_id = dag_to_pipeline(item,db.session,workflow_label={"schedule_type":"once"})
        # else:
        #     item.pipeline_file = None



    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):
        selected_volume_mount = self._pipeline_volume_mount(item)
        if not item.project or item.project.type != 'org':
            project = db.session.query(Project).filter_by(name='public').filter_by(type='org').first()
            if project:
                item.project = project
        # 环境变量不能包含空格
        if item.global_env:
            pipeline_global_env = [env.strip() for env in item.global_env.split('\n') if '=' in env.strip()]
            for index,env in enumerate(pipeline_global_env):
                env = env.split('=')
                env = [x.strip() for x in env]
                pipeline_global_env[index]='='.join(env)
            item.global_env = '\n'.join(pipeline_global_env)

        item.name = item.name.replace('_', '-')[0:54].lower().strip('-')
        if item.priority not in ('high', 'low'):
            item.priority = conf.get('PIPELINE_PRIORITY_DEFAULT', 'high')
        item.namespace = item.project.pipeline_namespace
        # item.alert_status = ','.join(item.alert_status)
        self.pipeline_args_check(item)
        item.create_datetime = datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()
        item.cronjob_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        item.parameter = json.dumps({}, indent=4, ensure_ascii=False)
        self._save_pipeline_volume_mount(item, selected_volume_mount)
        # 检测crontab格式
        if item.schedule_type == 'crontab':
            if not re.match("^[0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+", item.cron_time):
                raise MyappException(__("crontab 格式错误"))
                item.cron_time = ''

    def pre_update_req(self,req_json=None,src_item=None,*args,**kwargs):
        req_json = self._merge_volume_mount_req(req_json, src_item)
        if src_item and src_item.parameter:
            parameter = json.loads(src_item.parameter)
            if parameter.get("demo", 'false').lower() == 'true':
                raise MyappException(__("示例pipeline，不允许修改，请复制后编辑"))

        core.validate_json(req_json.get('expand','{}'))
        return req_json

    pre_add_req = pre_update_req

    @pysnooper.snoop()
    def pre_update(self, item):
        selected_volume_mount = self._pipeline_volume_mount(item)
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)
        else:
            item.expand = '{}'

        # 环境变量不能包含空格
        if item.global_env:
            pipeline_global_env = [env.strip() for env in item.global_env.split('\n') if '=' in env.strip()]
            for index, env in enumerate(pipeline_global_env):
                env = env.split('=')
                env = [x.strip() for x in env]
                pipeline_global_env[index] = '='.join(env)
            item.global_env = '\n'.join(pipeline_global_env)

        item.name = item.name.replace('_', '-')[0:54].lower()
        if item.priority not in ('high', 'low'):
            item.priority = conf.get('PIPELINE_PRIORITY_DEFAULT', 'high')
        # item.alert_status = ','.join(item.alert_status)
        self.pipeline_args_check(item)
        item.change_datetime = datetime.datetime.now()
        if item.parameter:
            item.parameter = json.dumps(json.loads(item.parameter), indent=4, ensure_ascii=False)
        else:
            item.parameter = '{}'
        self._save_pipeline_volume_mount(item, selected_volume_mount)

        if (item.schedule_type=='crontab' and self.src_item_json.get("schedule_type")=='once') or (item.cron_time!=self.src_item_json.get("cron_time",'')):
            item.cronjob_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 把没必要的存储去掉
        expand = json.loads(item.expand)
        for node in expand:
            if 'data' in node and 'args' in node['data'].get("info",{}):
                del node['data']['info']['args']
        item.expand = json.dumps(expand)

        # 限制提醒
        if item.schedule_type == 'crontab':
            if not item.cron_time or not re.match("^[0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+", item.cron_time.strip().replace('  ', ' ')):
                item.cron_time = ''
                raise MyappException(__("crontab 格式错误"))

            org = item.project.user_org
            if not org or org == 'public':
                flash(__('无法保障公共集群的稳定性，定时任务请选择专门的日更集群项目组'), 'warning')


    def pre_add_web(self, item=None):
        self._set_pipeline_volume_mount_field(item)

    def pre_update_web(self, item):
        self._set_pipeline_volume_mount_field(item)
        item.dag_json = item.fix_dag_json()
        item.expand = json.dumps(item.fix_expand(), indent=4, ensure_ascii=False)
        db.session.commit()


    def merge_add_field_info(self, response, **kwargs):
        self._set_pipeline_volume_mount_field()
        super().merge_add_field_info(response, **kwargs)

    def merge_edit_field_info(self, response, **kwargs):
        self._set_pipeline_volume_mount_field()
        super().merge_edit_field_info(response, **kwargs)

    # 删除前先把下面的task删除了，把里面的运行实例也删除了，把定时调度删除了
    @pysnooper.snoop()
    def pre_delete(self, pipeline):
        db.session.commit()
        if __("(废弃)") not in pipeline.describe:
            pipeline.describe += __("(废弃)")

        pipeline.schedule_type = 'once'
        pipeline.expand = ""
        pipeline.dag_json = "{}"
        db.session.commit()
        try:
            # 删除所有相关的运行中workflow
            back_crds = pipeline.get_workflow()
            self.delete_bind_crd(back_crds)
        except Exception as e:
            print(e)

        # 删除所有的任务
        try:
            tasks = pipeline.get_tasks()
            # 删除task启动的所有实例
            for task in tasks:
                self.delete_task_run(task)
        except Exception as e:
            print(e)


        # 删除所有的workflow
        # 只是删除了数据库记录，但是实例并没有删除，会重新监听更新的。
        try:
            db.session.query(Task).filter_by(pipeline_id=pipeline.id).delete()
            db.session.commit()
        except Exception as e:
            pass
        try:
            db.session.query(Workflow).filter_by(foreign_key=str(pipeline.id)).delete(synchronize_session=False)
            db.session.commit()
            db.session.query(Workflow).filter(Workflow.labels.contains(f'"pipeline-id": "{str(pipeline.id)}"')).delete(synchronize_session=False)
            db.session.commit()
        except Exception as e:
            print(e)
        try:
            db.session.query(RunHistory).filter_by(pipeline_id=pipeline.id).delete()
            db.session.commit()
        except Exception as e:
            pass
    @expose_api(description="我的pipeline列表",url="/my/list/")
    def my(self):
        try:
            user_id = g.user.id
            if user_id:
                pipelines = db.session.query(Pipeline).filter_by(created_by_fk=user_id).order_by(Pipeline.id.desc()).all()
                back = []
                for pipeline in pipelines:
                    back.append(pipeline.to_json())
                return json_response(message='success', status=0, result=back)
        except Exception as e:
            print(e)
            return json_response(message=str(e), status=-1, result={})

    @expose_api(description="示例pipeline列表",url="/demo/list/")
    def demo(self):
        try:
            pipelines = db.session.query(Pipeline).filter(Pipeline.parameter.contains('"demo": "true"')).all()
            back = []
            for pipeline in pipelines:
                back.append(pipeline.to_json())
            return json_response(message='success', status=0, result=back)
        except Exception as e:
            print(e)
            return json_response(message=str(e), status=-1, result={})

    # 删除手动发起的workflow，不删除定时任务发起的workflow
    def delete_bind_crd(self, crds):

        for crd in crds:
            try:
                run_id = json.loads(crd['labels']).get("run-id", '')
                if run_id:
                    # 定时任务发起的不能清理
                    run_history = db.session.query(RunHistory).filter_by(run_id=run_id).first()
                    if run_history:
                        continue

                    db_crd = db.session.query(Workflow).filter_by(name=crd['name']).first()
                    if db_crd and db_crd.pipeline:
                        k8s_client = py_k8s.K8s(db_crd.pipeline.project.cluster.get('KUBECONFIG', ''))
                    else:
                        k8s_client = py_k8s.K8s()

                    k8s_client.delete_workflow(
                        all_crd_info=conf.get("CRD_INFO", {}),
                        namespace=crd['namespace'],
                        run_id=run_id
                    )
                    # push_message(conf.get('ADMIN_USER', '').split(','),'%s手动运行新的pipeline %s，进而删除旧的pipeline run-id: %s' % (pipeline.created_by.username,pipeline.describe,run_id,))
                    if db_crd:
                        db_crd.status = 'Deleted'
                        db_crd.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        db.session.commit()
            except Exception as e:
                pass
                # print(e)

    def check_pipeline_perms(user_fun):
        # @pysnooper.snoop()
        def wraps(*args, **kwargs):
            pipeline_id = int(kwargs.get('pipeline_id', '0'))
            if not pipeline_id:
                response = make_response("pipeline_id not exist")
                response.status_code = 404
                return response

            if g.user.is_admin():
                return user_fun(*args, **kwargs)

            join_projects_id = security_manager.get_join_projects_id(db.session)
            pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
            if pipeline.project.id in join_projects_id:
                return user_fun(*args, **kwargs)

            response = make_response("no perms to run pipeline %s" % pipeline_id)
            response.status_code = 403
            return response

        return wraps

    # 保存pipeline正在运行的workflow信息
    def save_workflow(self, back_crds):
        # 把消息加入到源数据库
        for crd in back_crds:
            try:
                workflow = db.session.query(Workflow).filter_by(name=crd['name']).first()
                if not workflow:
                    username = ''
                    labels = json.loads(crd['labels'])
                    if 'run-rtx' in labels:
                        username = labels['run-rtx']
                    elif 'pipeline-rtx' in labels:
                        username = labels['pipeline-rtx']
                    elif 'run-username' in labels:
                        username = labels['run-username']
                    elif 'pipeline-username' in labels:
                        username = labels['pipeline-username']

                    workflow = Workflow(name=crd['name'], namespace=crd['namespace'], create_time=crd['create_time'],
                                        cluster=labels.get("cluster", ''),
                                        status=crd['status'],
                                        annotations=crd['annotations'],
                                        labels=crd['labels'],
                                        spec=crd['spec'],
                                        status_more=crd['status_more'],
                                        username=username
                                        )
                    db.session.add(workflow)
                    db.session.commit()
            except Exception as e:
                print(e)

    @event_logger.log_this
    @expose_api(description="运行指定pipeline",url="/run_pipeline/<pipeline_id>", methods=["GET", "POST"])
    @check_pipeline_perms
    # @pysnooper.snoop()
    def run_pipeline(self, pipeline_id):
        # print(pipeline_id)
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()

        # 只有管理员和创建者可以debug
        if pipeline.created_by_fk!=g.user.id and not g.user.is_admin():
            # 模板创建者可以调试模板
            message = __('仅管理员或创建者，可运行该任务流')
            flash(message, 'warning')
            return self.response(400, **{"status": 1, "result": {}, "message": message})

        pipeline.delete_old_task()
        tasks = db.session.query(Task).filter_by(pipeline_id=pipeline_id).all()
        if not tasks:
            flash('no task', 'warning')
            return redirect('/pipeline_modelview/api/web/%s' % pipeline.id)

        time.sleep(1)

        back_crds = pipeline.get_workflow()
        # 添加会和watch中的重复
        # if back_crds:
        #     self.save_workflow(back_crds)
        # 这里直接删除所有的历史任务流，正在运行的也删除掉
        # not_running_crds = back_crds  # [crd for crd in back_crds if 'running' not in crd['status'].lower()]
        self.delete_bind_crd(back_crds)

        # 删除task启动的所有实例
        for task in tasks:
            self.delete_task_run(task)

        # self.delete_workflow(pipeline)
        pipeline.pipeline_file,pipeline.run_id = dag_to_pipeline(pipeline, db.session,workflow_label={"schedule_type":"once"})  # 合成workflow
        # print('make pipeline file %s' % pipeline.pipeline_file)
        # return
        print('begin upload and run pipeline %s' % pipeline.name)
        pipeline.version_id = ''
        if not pipeline.pipeline_file:
            flash(__("请先编排任务，并进行保存后再运行整个任务流"),'warning')
            return redirect('/pipeline_modelview/api/web/%s' % pipeline.id)
        try:
            crd_name = run_pipeline(pipeline, json.loads(pipeline.pipeline_file))  # 会根据版本号是否为空决定是否上传
            pipeline.pipeline_argo_id = crd_name
            db.session.commit()  # 更新
        except Exception as e:
            return render_template('close.html', data=str(e).replace('<br>', '\n'))
            return redirect('/pipeline_modelview/api/web/%s' % pipeline.id)

        # back_crds = pipeline.get_workflow()
        # 添加会和watch中的重复
        # if back_crds:
        #     self.save_workflow(back_crds)

        return redirect("/pipeline_modelview/api/web/log/%s" % pipeline_id)
        # return redirect(run_url)



    # # @event_logger.log_this
    @expose_api(description="打开任务流编排界面",url="/web/<pipeline_id>", methods=["GET"])
    # @pysnooper.snoop()
    def web(self, pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()

        pipeline.dag_json = pipeline.fix_dag_json()  # 修正 dag_json
        pipeline.expand = json.dumps(pipeline.fix_expand(), indent=4, ensure_ascii=False)  # 修正 前端expand字段缺失
        pipeline.expand = json.dumps(pipeline.fix_position(), indent=4, ensure_ascii=False)  # 修正 节点中心位置到视图中间

        # # 自动排版
        # db_tasks = pipeline.get_tasks(db.session)
        # if db_tasks:
        #     try:
        #         tasks={}
        #         for task in db_tasks:
        #             tasks[task.name]=task.to_json()
        #         expand = core.fix_task_position(pipeline.to_json(),tasks,json.loads(pipeline.expand))
        #         pipeline.expand=json.dumps(expand,indent=4,ensure_ascii=False)
        #         db.session.commit()
        #     except Exception as e:
        #         print(e)

        db.session.commit()
        # print(pipeline_id)
        url = '/static/appbuilder/vison/index.html?pipeline_id=%s' % pipeline_id  # 前后端集成完毕，这里需要修改掉
        return redirect('/frontend/showOutLink?url=%s' % urllib.parse.quote(url, safe=""))
        # 返回模板
        # return self.render_template('link.html', data=data)

    # # @event_logger.log_this
    @expose_api(description="打开任务流调试跟踪界面",url="/web/log/<pipeline_id>", methods=["GET"])
    def web_log(self, pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        namespace = pipeline.namespace
        workflow_name = pipeline.pipeline_argo_id
        if workflow_name:
            cluster = pipeline.project.cluster["NAME"]
            url = f'/frontend/commonRelation?backurl=/workflow_modelview/api/web/dag/{cluster}/{namespace}/{workflow_name}'
            return redirect(url)
        else:
            message = __('未发现之前启动的任务流实例，请先运行该实例')
            return render_template('close.html', data=str(message).replace('<br>', '\n'))

            url = '/frontend/showOutLink?url=%2Fstatic%2Fappbuilder%2Fvison%2Findex.html%3Fpipeline_id%3D'+str(pipeline_id)
            return redirect(url)



    # # @event_logger.log_this
    @expose_api(description="打开任务流资源监控界面",url="/web/monitoring/<pipeline_id>", methods=["GET"])
    def web_monitoring(self, pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()

        url = "//"+pipeline.project.cluster.get('HOST', request.host).split('|')[-1]+conf.get('GRAFANA_TASK_PATH')+ pipeline.name
        return redirect(url)
        # else:
        #     flash('no running instance', 'warning')
        #     return redirect('/pipeline_modelview/api/web/%s' % pipeline.id)

    # # @event_logger.log_this
    @expose_api(description="打开任务流的pod界面",url="/web/pod/<pipeline_id>", methods=["GET"])
    def web_pod(self, pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        namespace = pipeline.namespace
        return redirect(f'/k8s/web/search/{pipeline.project.cluster["NAME"]}/{namespace}/{pipeline.name.replace("_", "-").lower()}')

    @expose_api(description="打开任务流的定时记录界面",url="/web/runhistory/<pipeline_id>", methods=["GET"])
    def web_runhistory(self,pipeline_id):
        url = conf.get('MODEL_URLS', {}).get('runhistory', '') + '?filter=' + urllib.parse.quote(json.dumps([{"key": "pipeline", "value": int(pipeline_id)}], ensure_ascii=False))
        # print(url)
        return redirect(url)

    @expose_api(description="打开任务流的调试跟踪界面",url="/web/workflow/<pipeline_id>", methods=["GET"])
    def web_workflow(self,pipeline_id):
        url = conf.get('MODEL_URLS', {}).get('workflow', '') + '?filter=' + urllib.parse.quote(json.dumps([{"key": "labels", "value": '"pipeline-id": "%s"'%pipeline_id}], ensure_ascii=False))
        # print(url)
        return redirect(url)


    # @pysnooper.snoop(watch_explode=('expand'))
    def copy_db(self, pipeline):
        new_pipeline = pipeline.clone()
        expand = json.loads(pipeline.expand) if pipeline.expand else {}
        new_pipeline.name = new_pipeline.name.replace('_', '-') + "-" + uuid.uuid4().hex[:4]
        if 'copy' not in new_pipeline.describe:
            new_pipeline.describe = new_pipeline.describe+"(copy)"
        new_pipeline.created_on = datetime.datetime.now()
        new_pipeline.changed_on = datetime.datetime.now()
        db.session.add(new_pipeline)
        db.session.commit()

        def change_node(src_task_id, des_task_id):
            for node in expand:
                if 'source' not in node:
                    # 位置信息换成新task的id
                    if int(node['id']) == int(src_task_id):
                        node['id'] = str(des_task_id)
                else:
                    if int(node['source']) == int(src_task_id):
                        node['source'] = str(des_task_id)
                    if int(node['target']) == int(src_task_id):
                        node['target'] = str(des_task_id)

        # 复制绑定的task，并绑定新的pipeline
        for task in pipeline.get_tasks():
            new_task = task.clone()
            new_task.pipeline_id = new_pipeline.id
            new_task.create_datetime = datetime.datetime.now()
            new_task.change_datetime = datetime.datetime.now()
            db.session.add(new_task)
            db.session.commit()
            change_node(task.id, new_task.id)

        new_pipeline.expand = json.dumps(expand)
        new_pipeline.parameter="{}" # 扩展参数不进行复制，这样demo的pipeline不会被复制一遍
        db.session.commit()
        return new_pipeline

    # # @event_logger.log_this
    @expose_api(description="复制任务流",url="/copy_pipeline/<pipeline_id>", methods=["GET", "POST"])
    def copy_pipeline(self, pipeline_id):
        # print(pipeline_id)
        message = ''
        try:
            pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
            new_pipeline = self.copy_db(pipeline)
            # return jsonify(new_pipeline.to_json())
            return redirect('/pipeline_modelview/api/web/%s'%new_pipeline.id)
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            message = str(e)
        response = make_response("copy pipeline %s error: %s" % (pipeline_id, message))
        response.status_code = 500
        return response

    @action("copy", "复制", confirmation= '复制所选记录?', icon="fa-copy", multiple=True, single=False)
    def copy(self, pipelines):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        try:
            for pipeline in pipelines:
                self.copy_db(pipeline)
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            raise e

        return redirect(request.referrer)


    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        return self._muldelete(items)


# 添加api
class Pipeline_ModelView_Api(Pipeline_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Pipeline)
    route_base = '/pipeline_modelview/api'
    # show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    list_columns = ['id', 'project', 'pipeline_url', 'creator', 'modified']
    add_columns = ['project', 'name', 'describe', 'parameter']
    edit_columns = ['project', 'name', 'describe', 'schedule_type', 'cron_time', 'depends_on_past', 'max_active_runs',
                    'expired_limit', 'parallelism', 'priority', 'parameter', 'dag_json', 'global_env', 'alert_status', 'alert_user', 'expand',
                    'cronjob_start_time']

    related_views = [Task_ModelView_Api, ]

    def pre_add_web(self):
        self.default_filter = {
            "created_by": g.user.id
        }
        self._set_pipeline_volume_mount_field()

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields


appbuilder.add_api(Pipeline_ModelView_Api)

class Pipeline_ModelView_Home_Api(Pipeline_ModelView_Api):
    datamodel = SQLAInterface(Pipeline)
    route_base = '/pipeline_modelview/home/api'
    list_columns = ['id', 'project', 'pipeline_url', 'creator', 'modified', 'changed_on', 'describe']


appbuilder.add_api(Pipeline_ModelView_Home_Api)
