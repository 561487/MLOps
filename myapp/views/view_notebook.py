import os
import re
import shlex
import traceback
from types import SimpleNamespace

from flask_appbuilder.baseviews import expose_api

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import pysnooper
import uuid
from myapp.models.model_notebook import Notebook
from myapp.models.model_job import Repository
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder, db, event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, Length, Regexp
from wtforms import SelectField, StringField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MySelect2Widget, MyBS3TextFieldWidget, MySelectMultipleField
from myapp.utils.storage_volume import available_volume_choices, filter_selected_volume_mount, split_volume_mount
from flask import Markup
from myapp.utils.py.py_k8s import K8s
from flask import (
    abort,
    flash,
    g,
    redirect,
    request, make_response,
)
from .baseApi import (
    MyappModelRestApi
)
from .base import MyappFilter
from flask_appbuilder import expose
import datetime, time, json
from myapp.views.view_team import Project_Join_Filter, filter_join_org_project
from myapp.models.model_team import Project

conf = app.config


def _safe_image_component(value, fallback='notebook', max_length=80):
    value = re.sub(r'[^a-z0-9._-]+', '-', (value or '').lower()).strip('._-')
    return (value or fallback)[:max_length].rstrip('._-') or fallback


def _notebook_save_image_prefix():
    return conf.get(
        'NOTEBOOK_SAVE_IMAGE_PREFIX',
        '10.121.177.20:8082/notebook/'
    ).strip().rstrip('/') + '/'


def _valid_notebook_target_image(target_image):
    prefix = _notebook_save_image_prefix()
    if not target_image or not target_image.startswith(prefix):
        return False
    remainder = target_image[len(prefix):]
    return bool(re.fullmatch(
        r'[a-z0-9._-]+(?:/[a-z0-9._-]+)+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}',
        remainder
    )) and '..' not in remainder


class Notebook_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        return query.filter(self.model.created_by_fk == g.user.id)


class Notebook_ModelView_Base():
    datamodel = SQLAInterface(Notebook)
    label_title = _('notebook')
    crd_name = 'notebook'
    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete', 'can_edit', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    base_filters = [["id", Notebook_Filter, lambda: []]]
    order_columns = ['id']
    search_columns = ['created_by', 'name']

    add_columns = ['project', 'name', 'describe', 'images', 'working_dir', 'volume_mount', 'resource_memory','resource_cpu', 'resource_gpu']
    list_columns = ['project', 'ide_type_html', 'name_url', 'status', 'describe','reset', 'resource', 'renew', 'save']
    show_columns = ['project', 'name', 'namespace', 'describe', 'images', 'working_dir', 'env', 'volume_mount','resource_memory', 'resource_cpu', 'resource_gpu', 'expand']
    cols_width = {
        "project": {"type": "ellip2", "width": 120},
        "ide_type_html": {"type": "ellip2", "width": 200},
        "name_url": {"type": "ellip2", "width": 150},
        "describe": {"type": "ellip2", "width": 180},
        "resource": {"type": "ellip2", "width": 270},
        "status": {"type": "ellip2", "width": 140},
        "renew": {"type": "ellip2", "width": 150},
        "save": {"type": "ellip2", "width": 200},
        "ops_html": {"type": "ellip2", "width": 130}
    }
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields

    # @pysnooper.snoop()
    def set_column(self, notebook=None):
        # 对编辑进行处理
        self.add_form_extra_fields['name'] = StringField(
            _('名称'),
            default="%s-" % g.user.username + uuid.uuid4().hex[:4],
            description= _('英文名(小写字母、数字、-组成)，最长50个字符'),
            widget=MyBS3TextFieldWidget(readonly=True if notebook else False),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]  # 注意不能以-开头和结尾
        )
        self.add_form_extra_fields['describe'] = StringField(
            _('描述'),
            default='%s-notebook' % g.user.username,
            description= _('中文描述'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        self.add_form_extra_fields['project'] = QuerySelectField(
            _('项目组'),
            default='',
            description= _('部署项目组，在切换项目组前注意先停止当前notebook'),
            query_factory=filter_join_org_project,
            widget=MySelect2Widget(new_web=False),
        )

        NOTEBOOK_IMAGES = conf.get('NOTEBOOK_IMAGES', [])
        if type(NOTEBOOK_IMAGES)==dict:
            self.add_form_extra_fields['images'] = SelectField(
                _('镜像'),
                description=_('notebook基础环境镜像，如果显示不准确，请删除新建notebook'),
                widget=MySelect2Widget(new_web=False, can_input=True),
                choices=core.notebook_cascade_demo(NOTEBOOK_IMAGES),
                validators=[DataRequired()]
            )
        elif type(NOTEBOOK_IMAGES)==list:
            self.add_form_extra_fields['images'] = SelectField(
                _('镜像'),
                description=_('notebook基础环境镜像，如果显示不准确，请删除新建notebook'),
                widget=MySelect2Widget(new_web=False, can_input=True),
                choices=[[x[0], x[1]] for x in NOTEBOOK_IMAGES],
                validators=[DataRequired()]
            )



        self.add_form_extra_fields['node_selector'] = StringField(
            _('机器选择'),
            default='cpu=true;notebook=true',
            description= _("部署task所在的机器"),
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['image_pull_policy'] = SelectField(
            _('拉取策略'),
            description= _("镜像拉取策略(Always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)"),
            widget=Select2Widget(),
            choices=[['Always', 'Always'], ['IfNotPresent', 'IfNotPresent']]
        )
        current_volume_mount = notebook.volume_mount if notebook else ''
        project = notebook.project if notebook else None
        self.add_form_extra_fields['volume_mount'] = MySelectMultipleField(
            _('挂载卷'),
            default=current_volume_mount,
            description=_('选择项目默认挂载卷或已同步的存储资源，保存后会自动生成 volume_mount 表达式；reset notebook 后生效'),
            widget=MySelect2Widget(multiple=True),
            choices=available_volume_choices(g.user, project=project, current_volume_mount=current_volume_mount),
        )
        self.add_form_extra_fields['working_dir'] = StringField(
            _('工作目录'),
            default='/mnt',
            description= _("工作目录，如果为空，则使用Dockerfile中定义的workingdir"),
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['resource_memory'] = StringField(
            _('内存'),
            default=Notebook.resource_memory.default.arg,
            description= _('内存的资源使用配置，示例：1G，20G'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^.*G$")]
        )
        self.add_form_extra_fields['resource_cpu'] = StringField(
            _('cpu'),
            default=Notebook.resource_cpu.default.arg,
            description= _('cpu的资源使用配置(单位：核)，示例：2'), widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[0-9]*$")]
        )

        self.add_form_extra_fields['resource_gpu'] = StringField(
            _('gpu'),
            default='0',
            description= _('申请的gpu资源，示例:2为独占整卡；0.5为 HAMI 半卡；10G,50 为 HAMI 显存10G、算力50%；申请具体卡型号可写 1(V100) 或 10G,50(A100)。共享 GPU 使用 HAMI 格式'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp('^[\-\.0-9,a-zA-Z\(\)]*$')]
        )

        columns = ['name', 'describe', 'images', 'resource_memory', 'resource_cpu', 'resource_gpu']

        self.add_columns = ['project'] + columns + ['volume_mount']
        self.edit_columns = ['project'] + columns + ['volume_mount']

        self.edit_form_extra_fields = self.add_form_extra_fields
        self.default_filter = {
            "created_by": g.user.id
        }

    def set_columns_related(self, exist_add_args, response_add_columns):
        if 'volume_mount' not in response_add_columns:
            return
        project_value = exist_add_args.get('project') or exist_add_args.get('project_id') or {}
        if isinstance(project_value, dict):
            project_value = project_value.get('id') or project_value.get('value')
        project = db.session.query(Project).filter_by(id=int(project_value)).first() if str(project_value).isdigit() else None
        current_volume_mount = exist_add_args.get('volume_mount') or (project.volume_mount if project else '')
        choices = available_volume_choices(g.user, project=project, current_volume_mount=current_volume_mount)
        response_add_columns['volume_mount'].update({
            "label": _('挂载卷'),
            "description": _('选择项目默认挂载卷或已同步的存储资源，保存后会自动生成 volume_mount 表达式；reset notebook 后生效'),
            "type": "Select",
            "ui-type": "select2",
            "default": split_volume_mount(current_volume_mount),
            "choices": choices,
            "values": [{"id": choice[0], "value": choice[1]} for choice in choices],
        })

    def pre_add(self, item):
        item.name = item.name.replace("_", "-")[0:54].lower()
        item.resource_gpu = item.resource_gpu.upper() if item.resource_gpu else '0'
        selected_volume_mount = item.volume_mount

        # 不需要用户自己填写node selector
        # if core.get_gpu(item.resource_gpu)[0]:
        #     item.node_selector = item.node_selector.replace('cpu=true','gpu=true')
        # else:
        #     item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')

        item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_item_json.get('resource_memory',None))
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_item_json.get('resource_cpu',None))
        item.resource_gpu = core.check_resource_gpu(item.resource_gpu, self.src_item_json.get('resource_gpu', None))
        if not item.namespace:
            item.namespace = item.project.notebook_namespace

        if 'theia' in item.images or 'vscode' in item.images:
            item.ide_type = 'theia'
        elif 'matlab' in item.images:
            item.ide_type = 'matlab'
        elif 'rstudio' in item.images.lower() or 'rserver' in item.images.lower():
            item.ide_type = 'rstudio'
        else:
            item.ide_type = 'jupyter'

        if selected_volume_mount and not g.user.is_admin():
            selected_volume_mount = filter_selected_volume_mount(g.user, item.project, selected_volume_mount)
        item.volume_mount = core.merge_volume_mount(item.project.volume_mount, selected_volume_mount)



        all_images={x[1]:x[0] for x in conf.get('NOTEBOOK_IMAGES', []) }
        if item.images in all_images:
            item.images = all_images[item.images]


    # @pysnooper.snoop(watch_explode=('item'))
    def pre_update(self, item):

        # if item.changed_by_fk:
        #     item.changed_by=db.session.query(MyUser).filter_by(id=item.changed_by_fk).first()
        # if item.created_by_fk:
        #     item.created_by=db.session.query(MyUser).filter_by(id=item.created_by_fk).first()

        self.pre_add(item)




        # 如果修改了基础镜像，就把debug中的任务删除掉
        if self.src_item_json:
            # k8s集群更换了，要删除原来的
            if str(self.src_item_json.get('project_id', '1')) != str(item.project.id):
                src_project = db.session.query(Project).filter_by(id=int(self.src_item_json.get('project_id', '1'))).first()
                if src_project and src_project.cluster['NAME'] != item.project.cluster['NAME']:
                    self.base_stop([item],src_project.cluster['NAME'])
                    flash(__('发现集群更换，已帮你删除之前启动的notebook'), 'success')

    def post_add(self, item):

        try:
            self.reset_notebook(item)
        except Exception as e:
            print(e)
            flash(__('start fail, please reset notebook: ')+str(e), 'warning')
            return

        flash(__('自动reset 一分钟后生效'), 'info')

    # @pysnooper.snoop(watch_explode=('item'))
    def post_update(self, item):
        flash(__('reset以后配置方可生效'), 'info')

        # item.changed_on = datetime.datetime.now()
        # db.session.commit()
        # self.reset_notebook(item)

        # flash('自动reset 一分钟后生效', 'warning')
        if self.src_item_json:
            changed_by_fk = self.src_item_json.get('changed_by_fk','')
            if changed_by_fk:
                item.changed_by_fk = int(self.src_item_json.get('changed_by_fk'))
        if self.src_item_json:
            created_by_fk = self.src_item_json.get('created_by_fk','')
            if created_by_fk:
                item.created_by_fk = int(self.src_item_json.get('created_by_fk'))

        db.session.commit()

    def post_list(self,items):
        flash(__('注意：个人重要文件本地git保存，notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。<br>个人持久化目录在/mnt/')+g.user.username,category='info')
        return items

    pre_update_web = set_column
    pre_add_web = set_column

    @expose_api(description="创建打开jupyter",url='/entry/jupyter', methods=['GET', 'DELETE'])
    def entry_jupyter(self):
        data=request.args
        name = data.get('name', g.user.username + '-pipeline').replace("_", '')[:56]
        notebook = db.session.query(Notebook).filter(Notebook.name == name).first()
        expand = json.loads(notebook.expand) if notebook and notebook.expand else {}

        project_name=data.get('project_name',notebook.project.name if notebook else 'public')
        label = data.get('label',notebook.describe if notebook else '打开目录')
        resource_memory = data.get('resource_memory',notebook.resource_memory if notebook else "10G")
        resource_cpu = data.get('resource_cpu',notebook.resource_cpu if notebook else "10")
        volume_mount = data.get('volume_mount',notebook.volume_mount if notebook else 'kubeflow-user-workspace(pvc):/mnt')
        file_path = data.get('file_path', expand.get('root',''))   # 一定要是文件在 notebook的容器目录
        if 'http://' in file_path or 'https://' in file_path:
            file_path = '/mnt/{{creator}}'

        def template_str(src_str):
            from jinja2 import Environment, BaseLoader, DebugUndefined
            rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
            des_str = rtemplate.render(creator=g.user.username,
                                       datetime=datetime,
                                       runner=g.user.username,
                                       uuid=uuid
                                       )
            return des_str

        file_path = template_str(file_path)
        if f'/mnt/{g.user.username}' in file_path:
            file_path = file_path.replace(f'/mnt/{g.user.username}','')
        # 如果是打不开的文本文件，就自动变为目录
        file_name = file_path.split('/')[-1]
        text_file_extensions = {'ipynb', 'py', 'R', 'txt', 'csv', 'json', 'xml', 'py', 'html', 'css', 'js', 'md', 'yaml', 'yml', 'html', 'jpg', 'jpeg', 'png', 'sh'}
        if '.' in file_name:
            if file_name.split('.')[-1] not in text_file_extensions:
                file_path = file_path.replace(file_name,'')

        images = data.get('images',f'{conf.get("REPOSITORY_ORG","ccr.ccs.tencentyun.com/cube-studio/")}notebook:jupyter-ubuntu22.04')
        project = db.session.query(Project).filter(Project.name==project_name).filter(Project.type=='org').first()
        if not project:
            res = make_response("项目组%s不存在"%project_name)
            res.status_code = 405
            return res


        if not notebook:
            notebook = Notebook()
        notebook.project = project
        notebook.project_id = project.id
        notebook.name = name
        notebook.describe = label
        notebook.images = images
        notebook.ide_type = 'jupyter'
        notebook.working_dir = ''
        notebook.volume_mount = volume_mount
        notebook.resource_memory = resource_memory
        notebook.created_by=g.user
        notebook.changed_by=g.user
        notebook.resource_cpu = resource_cpu
        expand['status']='online'
        if file_path.strip('/'):
            expand['root'] = file_path
        notebook.expand = json.dumps(expand)

        if not notebook.id:
            notebook.created_on=datetime.datetime.now()
            db.session.add(notebook)

        db.session.commit()

        notebook_id = notebook.id
        k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG', ''))
        namespace = notebook.project.notebook_namespace
        del_namespace = notebook.namespace
        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})

        # 删除
        if request.method=='DELETE':
            try:
                k8s_client.delete_pods(namespace=del_namespace,pod_name=name)
            except Exception as e:
                print(e)
            try:
                k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=del_namespace, name=name)
            except Exception as e:
                print(e)

            try:
                k8s_client.delete_service(namespace=del_namespace,name=name)
            except Exception as e:
                print(e)

            notebook.delete()
            db.session.commit()
            res = make_response(__("删除成功"))
            res.status_code = 200
            return res

        notebook = db.session.query(Notebook).filter(Notebook.id==int(notebook_id)).first()

        port = 3000
        name = notebook.name

        # 创建新的service
        labels = {"app": notebook.name, 'user': notebook.created_by.username, 'pod-type': "notebook"}
        try:
            exist_service = k8s_client.v1.read_namespaced_service(name=name, namespace=namespace)
        except:
            exist_service = None
        if not exist_service:
            k8s_client.create_service(
                namespace=namespace,
                name=name,
                username=notebook.created_by.username,
                ports=[port],
                selector=labels
            )
        try:
            exist_crd = k8s_client.CustomObjectsApi.get_namespaced_custom_object(
                group=crd_info['group'], version=crd_info['version'],
                plural=crd_info['plural'], namespace=namespace,
                name=name
            )
        except:
            exist_crd=None
        host = notebook.project.cluster.get('HOST', request.host).split('|')[-1].strip().split(':')[0]

        if not exist_crd:
            # 创建vs
            crd_json = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": {
                    "gateways": [
                        "kubeflow/kubeflow-gateway"
                    ],
                    "hosts": [
                        "*" if core.checkip(host) else host
                    ],
                    "http": [
                        {
                            "match": [
                                {
                                    "uri": {
                                        "prefix": f"/notebook/jupyter/{notebook.name}/"
                                    },
                                    "headers": {
                                        "cookie": {
                                            "regex": ".*myapp_username=.*"
                                        }
                                    }

                                }
                            ],
                            "rewrite": {
                                "uri": '/notebook/jupyter/%s/' % notebook.name
                            },
                            "route": [
                                {
                                    "destination": {
                                        "host": "%s.%s.svc.cluster.local" % (notebook.name, namespace),
                                        "port": {
                                            "number": port
                                        }
                                    }
                                }
                            ],
                            "timeout": "300s"
                        }
                    ]
                }
            }

            # print(crd_json)
            crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                                        namespace=namespace, body=crd_json)

        exist_pod = k8s_client.get_pods(pod_name=name, namespace=namespace)
        if exist_pod:
            exist_pod = exist_pod[0]
            if exist_pod['status'].lower() != 'running':
                k8s_client.v1.delete_namespaced_pod(name, namespace, grace_period_seconds=0)
                exist_pod = None

        rewrite_url = '/notebook/jupyter/%s/' % notebook.name
        username=g.user.username
        if not exist_pod:

            pre_command = '(nohup sh /init.sh > /notebook_init.log 2>&1 &) ; (nohup sh /mnt/%s/init.sh > /init.log 2>&1 &) ; ' % username
            working_dir = '/mnt/%s' % username
            command = ["sh", "-c", "%s jupyter lab --notebook-dir=%s --ip=0.0.0.0 "
                                   "--no-browser --allow-root --port=%s "
                                   "--NotebookApp.token='' --NotebookApp.password='' --ServerApp.disable_check_xsrf=True "
                                   "--NotebookApp.allow_origin='*' "
                                   "--NotebookApp.base_url=%s" % (pre_command, "/mnt/"+username, port, rewrite_url)]
            env = {
                "NO_AUTH": "true",
            }
            pod, pod_spec = k8s_client.make_pod(
                namespace=namespace,
                name=name,
                labels=labels,
                annotations = {"project":project_name},
                command=command,
                args=None,
                volume_mount=notebook.volume_mount,
                working_dir=working_dir,
                node_selector=notebook.get_node_selector(),
                resource_memory="0G~" + notebook.resource_memory,
                resource_cpu="0~" + notebook.resource_cpu,
                resource_gpu=notebook.resource_gpu,
                image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
                image_pull_secrets=conf.get('HUBSECRET', []),
                image=notebook.images,
                hostAliases=conf.get('HOSTALIASES', ''),
                env=env,
                privileged=None,
                accounts=conf.get('JUPYTER_ACCOUNTS',''),
                username=username,
                restart_policy='Never'
            )
            # print(pod)
            try:
                pod = k8s_client.v1.create_namespaced_pod(namespace, pod)
                notebook.namespace=namespace
                db.session.commit()
                time.sleep(1)
            except Exception as e:
                print(e)
        # print(pod)

        left_retry = 10
        status=''
        while(left_retry):
            pod = k8s_client.get_pods(namespace=namespace, pod_name=name)
            if pod:
                pod=pod[0]
                status=json.dumps(pod['status_more'],ensure_ascii=False,indent=4, default=str).replace('\n',"<br>")
                if pod['status']=='Running':
                    if file_path:
                        if file_path.lstrip('/'):
                            file_path=f'/notebook/jupyter/{name}/lab/tree/'+file_path.strip('/')
                            time.sleep(2)
                            return redirect(file_path)
                        else:
                            file_path = f'/notebook/jupyter/{name}/lab?#/mnt/{g.user.username}'
                            time.sleep(2)
                            return redirect(file_path)

                    return redirect('%s%s'%(request.host_url.strip('/'),rewrite_url))
            left_retry=left_retry-1
            time.sleep(2)
        res = make_response(__("notebook未就绪，刷新此页面。<br> notebook状态：<br><br>")+Markup(status))
        res.status_code=200
        return res


    # @pysnooper.snoop(watch_explode=('message'))
    def reset_notebook(self, notebook):
        notebook.changed_on = datetime.datetime.now()
        db.session.commit()
        self.reset_theia(notebook)

    # 部署pod，service，VirtualService
    # @pysnooper.snoop(watch_explode=('notebook',))
    def reset_theia(self, notebook):
        try:
            # 先清理notebook
            self.pre_delete(notebook)
        except:
            pass
        k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG', ''))
        new_namespace = notebook.project.notebook_namespace
        old_namespace = notebook.namespace
        SERVICE_EXTERNAL_IP = []

        # 先使用项目组的
        if notebook.project.expand:
            SERVICE_EXTERNAL_IP = json.loads(notebook.project.expand).get('SERVICE_EXTERNAL_IP', '')
        # 使用集群的ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = notebook.project.cluster.get('HOST','')
        # 使用全局ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', None)
        # 使用当前url的
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = request.host

        if SERVICE_EXTERNAL_IP and type(SERVICE_EXTERNAL_IP) == str:
            SERVICE_EXTERNAL_IP = [SERVICE_EXTERNAL_IP]

        port = 3000
        security_context=None
        command = None
        workingDir = None
        health=None
        volume_mount = notebook.volume_mount
        # 端口+0是jupyterlab  +1是sshd   +2 +3 是预留的用户自己启动应用占用的端口
        port_str = conf.get('NOTEBOOK_PORT','10000+10*ID').replace('ID', str(notebook.id))
        meet_ports = core.get_not_black_port(int(eval(port_str)))

        env = {
            "NO_AUTH": "true",
            "DISPLAY": ":10.0",   # 屏幕投屏时使用
            "USERNAME": notebook.created_by.username,
            "NODE_OPTIONS": "--max-old-space-size=%s" % str(int(notebook.resource_memory.replace("G", '')) * 1024),
            "SSH_PORT": str(meet_ports[1]),
            "PORT1": str(meet_ports[2]),
            "PORT2": str(meet_ports[3]),
            "NOTEBOOK_NAME":notebook.name
        }

        rewrite_url = '/'

        pre_command = '(nohup sh /init.sh > /notebook_init.log 2>&1 &) ; (nohup sh /mnt/%s/init.sh > /init.log 2>&1 &) ; ' % notebook.created_by.username
        if notebook.ide_type == 'jupyter' or notebook.ide_type == 'bigdata' or notebook.ide_type == 'machinelearning' or notebook.ide_type == 'deeplearning':
            rewrite_url = '/notebook/jupyter/%s/' % notebook.name
            workingDir = '/mnt/%s' % notebook.created_by.username
            command = ["sh", "-c", "%s jupyter lab --notebook-dir=%s --ip=0.0.0.0 "
                                   "--no-browser --allow-root --port=%s "
                                   "--NotebookApp.token='' --NotebookApp.password='' --ServerApp.disable_check_xsrf=True "
                                   "--NotebookApp.allow_origin='*' "
                                   "--NotebookApp.base_url=%s" % (pre_command, notebook.mount, port, rewrite_url)]



        elif notebook.ide_type=='theia':
            command = ["bash",'-c','%s node /home/theia/src-gen/backend/main.js /home/project --hostname=0.0.0.0 --port=%s'%(pre_command,port)]
            workingDir = '/home/theia'



        image_pull_secrets = conf.get('HUBSECRET', [])
        user_repositorys = db.session.query(Repository).filter(Repository.created_by_fk == g.user.id).all()
        image_pull_secrets = list(set(image_pull_secrets + [rep.hubsecret for rep in user_repositorys]))

        labels = {"app": notebook.name, 'user': notebook.created_by.username, 'pod-type': "notebook"}
        notebook_env = []
        if notebook.env:
            notebook_env = [x.strip() for x in notebook.env.split('\n') if x.strip()]
            notebook_env = [env.split("=") for env in notebook_env if '=' in env]
            notebook_env = dict(zip([env[0] for env in notebook_env], [env[1] for env in notebook_env]))
        if notebook_env:
            env.update(notebook_env)
        if SERVICE_EXTERNAL_IP:
            env["SERVICE_EXTERNAL_IP"] = SERVICE_EXTERNAL_IP[0].split('|')[-1].split(':')[0]


        annotations={
            'project': notebook.project.name
        }
        notebook.namespace = new_namespace
        db.session.commit()

        k8s_client.create_debug_pod(
            namespace=new_namespace,
            name=notebook.name,
            labels=labels,
            annotations=annotations,
            command=command,
            args=None,
            volume_mount=volume_mount,
            working_dir=workingDir,
            node_selector=notebook.get_node_selector(),
            resource_memory=notebook.resource_memory if conf.get('NOTEBOOK_EXCLUSIVE',False) else ("0G~" + notebook.resource_memory),
            resource_cpu=notebook.resource_cpu if conf.get('NOTEBOOK_EXCLUSIVE',False) else ("0G~" + notebook.resource_cpu),
            resource_gpu=notebook.resource_gpu,
            image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
            image_pull_secrets=image_pull_secrets,
            image=notebook.images,
            hostAliases=conf.get('HOSTALIASES', ''),
            env=env,
            privileged=None,   # 这里设置privileged 才能看到所有的gpu卡，小心权限太高
            accounts=conf.get('JUPYTER_ACCOUNTS',''),
            username=notebook.created_by.username
        )
        k8s_client.create_service(
            namespace=new_namespace,
            name=notebook.name,
            username=notebook.created_by.username,
            ports=[port, ],
            selector=labels
        )

        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})
        old_crd_name = f"notebook-{old_namespace}-{notebook.name.replace('_', '-')}" #  notebook.name.replace('_', '-')
        new_crd_name = f"notebook-{new_namespace}-{notebook.name.replace('_', '-')}"  # notebook.name.replace('_', '-')
        vs_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=old_namespace, name=old_crd_name)
        if vs_obj:
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=old_namespace, name=old_crd_name)
            time.sleep(1)

        host = notebook.project.cluster.get('HOST', request.host).split('|')[-1].strip().split(':')[0]
        web_crd_json = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": new_crd_name,
                "namespace": new_namespace
            },
            "spec": {
                "gateways": [
                    "kubeflow/kubeflow-gateway"
                ],
                "hosts": [
                    "*" if core.checkip(host) else host
                ],
                "http": [
                    {
                        "match": [
                            {
                                "uri": {
                                    "prefix": f"/notebook/jupyter/{notebook.name}/"
                                },
                                "headers": {
                                    "cookie":{
                                        "regex": ".*myapp_username=.*"
                                    }
                                }
                            }
                        ],
                        "rewrite": {
                            "uri": rewrite_url
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": "%s.%s.svc.cluster.local" % (notebook.name, new_namespace),
                                    "port": {
                                        "number": port
                                    }
                                }
                            }
                        ],
                        "timeout": "300s"
                    }
                ]
            }
        }

        # print(crd_json)
        try:
            crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=new_namespace, body=web_crd_json)
        except:
            pass
        # 边缘模式时，需要根据项目组中的配置设置代理ip
        if meet_ports[0]>=20000:
            flash(__('端口已耗尽，ssh连接notebook请通过跳板机连接'), 'warning')

        if SERVICE_EXTERNAL_IP and SERVICE_EXTERNAL_IP[0]!='127.0.0.1' and meet_ports[0]<20000:
            external_ip = [ip.split('|')[0].strip().split(':')[0] for ip in SERVICE_EXTERNAL_IP]
            external_ip = [ip for ip in external_ip if core.checkip(ip)]
            ports = [port]
            ports.append(meet_ports[1])   # 给每个notebook多开一个端口，ssh的端口

            # for index in range(1, 4):
            #     ports.append(meet_ports[index])

            # ports = list(set(ports))  # 这里会乱序
            service_ports = [[meet_ports[index], port] for index, port in enumerate(ports)]
            service_external_name = (notebook.name + "-external").lower()[:60].strip('-')
            k8s_client.create_service(
                namespace=new_namespace,
                name=service_external_name,
                username=notebook.created_by.username,
                ports=service_ports,
                selector=labels,
                service_type='ClusterIP' if notebook.project.cluster['K8S_NETWORK_MODE']!='ipvs' else 'NodePort',
                external_ip=external_ip if notebook.project.cluster['K8S_NETWORK_MODE']!='ipvs' else None
            )

        expand = json.loads(notebook.expand) if notebook.expand else {}
        expand['status']='online'
        notebook.expand = json.dumps(expand)
        db.session.commit()

    def _get_owned_notebook(self, notebook_id, for_update=False):
        query = db.session.query(Notebook).filter_by(id=notebook_id)
        if for_update:
            query = query.with_for_update()
        notebook = query.first()
        if not notebook:
            abort(404)
        if not g.user.is_admin() and notebook.created_by_fk != g.user.id:
            abort(403)
        return notebook

    @staticmethod
    def _save_expand(notebook, **values):
        notebook = (
            db.session.query(Notebook)
            .populate_existing()
            .with_for_update()
            .filter_by(id=notebook.id)
            .first()
        )
        expand = json.loads(notebook.expand) if notebook.expand else {}
        expand.update(values)
        notebook.expand = json.dumps(expand, ensure_ascii=False)
        db.session.commit()
        return expand

    @staticmethod
    def _save_in_cooldown(expand, field, cooldown=None):
        value = expand.get(field)
        if not value:
            return False
        try:
            last_time = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            return False
        if cooldown is None:
            cooldown = int(conf.get('NOTEBOOK_SAVE_COOLDOWN_SEC', 120))
        return (datetime.datetime.now() - last_time).total_seconds() < cooldown

    @staticmethod
    def _running_notebook_pod(k8s_client, notebook):
        try:
            pod = k8s_client.v1.read_namespaced_pod(
                name=notebook.name,
                namespace=notebook.namespace,
                _request_timeout=5
            )
        except Exception:
            return None
        if not pod.status or pod.status.phase != 'Running':
            return None
        return pod

    @staticmethod
    def _save_env_relative_dir(notebook):
        relative_dir = conf.get(
            'NOTEBOOK_SAVE_ENV_DIR',
            'notebooks/{name}/'
        ).format(name=_safe_image_component(notebook.name))
        relative_dir = relative_dir.strip().strip('/')
        if (
            not relative_dir
            or '..' in relative_dir.split('/')
            or not re.fullmatch(r'[a-zA-Z0-9._/-]+', relative_dir)
        ):
            raise ValueError(__('NOTEBOOK_SAVE_ENV_DIR 配置不合法'))
        return relative_dir

    @event_logger.log_this
    @expose_api(description="保存 Notebook 轻量环境", url='/save_env/<notebook_id>', methods=['GET', 'POST'])
    def save_env(self, notebook_id):
        redirect_url = conf.get('MODEL_URLS', {}).get('notebook', '')
        if not conf.get('NOTEBOOK_SAVE_ENABLED', True):
            flash(__('Notebook 环境保存功能未启用'), 'warning')
            return redirect(redirect_url)

        notebook = self._get_owned_notebook(notebook_id, for_update=True)
        expand = json.loads(notebook.expand) if notebook.expand else {}
        if self._save_in_cooldown(expand, 'save_env_last_request_time'):
            flash(__('环境保存请求过于频繁，请稍后重试'), 'warning')
            return redirect(redirect_url)

        k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG', ''))
        if not self._running_notebook_pod(k8s_client, notebook):
            flash(__('Notebook 未运行，请先启动或 reset 后再保存环境'), 'warning')
            return redirect(redirect_url)

        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_expand(
            notebook,
            save_env_status='saving',
            save_env_last_request_time=now
        )

        username = notebook.created_by.username
        try:
            if (
                not re.fullmatch(r'[a-zA-Z0-9._-]+', username or '')
                or username in ('.', '..')
            ):
                raise ValueError(__('用户名无法用于环境保存路径'))
            relative_dir = self._save_env_relative_dir(notebook)
            save_dir = '/mnt/{}/{}'.format(username, relative_dir)
            init_path = '/mnt/{}/init.sh'.format(username)
            start_marker = '# >>> notebook-env-save:{} >>>'.format(notebook.id)
            end_marker = '# <<< notebook-env-save:{} <<<'.format(notebook.id)
            script = r'''
set -eu
save_dir={save_dir}
init_path={init_path}
start_marker={start_marker}
end_marker={end_marker}
mkdir -p "$save_dir"
kind=pip
env_path="$save_dir/requirements.txt"
conda_tmp="$save_dir/.environment.yml.tmp"
pip_tmp="$save_dir/.requirements.txt.tmp"
if command -v conda >/dev/null 2>&1 && conda env export --no-builds > "$conda_tmp" 2>/dev/null; then
    env_path="$save_dir/environment.yml"
    mv "$conda_tmp" "$env_path"
    rm -f "$pip_tmp"
    kind=conda
else
    rm -f "$conda_tmp"
    if command -v python >/dev/null 2>&1; then
        python -m pip freeze > "$pip_tmp"
    else
        python3 -m pip freeze > "$pip_tmp"
    fi
    mv "$pip_tmp" "$env_path"
fi
touch "$init_path"
lock_dir=""
if command -v flock >/dev/null 2>&1; then
    exec 9>"${{init_path}}.notebook-env-save.lock"
    flock -w 30 9
else
    lock_dir="${{init_path}}.notebook-env-save.lockdir"
    retries=30
    while ! mkdir "$lock_dir" 2>/dev/null; do
        retries=$((retries - 1))
        [ "$retries" -gt 0 ] || exit 1
        sleep 1
    done
fi
init_tmp=$(mktemp "${{init_path}}.notebook-env-save.XXXXXX")
cleanup() {{
    [ -z "$init_tmp" ] || rm -f "$init_tmp"
    [ -z "$lock_dir" ] || rmdir "$lock_dir" 2>/dev/null || true
}}
trap cleanup EXIT
awk -v start="$start_marker" -v end="$end_marker" '
    $0 == start && !skipping {{ skipping=1; buffered=$0 ORS; next }}
    skipping {{
        buffered=buffered $0 ORS
        if ($0 == end) {{ skipping=0; buffered="" }}
        next
    }}
    {{ print }}
    END {{ if (skipping) printf "%s", buffered }}
' "$init_path" > "$init_tmp"
printf '%s\n' "$start_marker" >> "$init_tmp"
printf '%s\n' '# managed by platform; do not edit this block manually' >> "$init_tmp"
if [ "$kind" = conda ]; then
    printf 'conda env update -f %s || true\n' "$env_path" >> "$init_tmp"
else
    printf 'pip install -r %s || true\n' "$env_path" >> "$init_tmp"
fi
printf '%s\n' "$end_marker" >> "$init_tmp"
chmod --reference="$init_path" "$init_tmp" 2>/dev/null || true
mv "$init_tmp" "$init_path"
init_tmp=""
printf '__NOTEBOOK_ENV_SAVE_OK__:%s:%s\n' "$kind" "$env_path"
'''.format(
                save_dir=shlex.quote(save_dir),
                init_path=shlex.quote(init_path),
                start_marker=shlex.quote(start_marker),
                end_marker=shlex.quote(end_marker)
            )
            output = k8s_client.exec_pod(
                name=notebook.name,
                namespace=notebook.namespace,
                container=notebook.name,
                command=['sh', '-c', script],
                timeout=120
            )
            match = re.search(r'__NOTEBOOK_ENV_SAVE_OK__:(pip|conda):([^\r\n]+)', output or '')
            if not match:
                raise RuntimeError(__('Pod 内环境导出失败'))

            success_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._save_expand(
                notebook,
                save_env_status='success',
                save_env_path=match.group(2),
                save_env_success_last_time=success_time
            )
            flash(__('Notebook 环境已保存'), 'success')
        except Exception as ex:
            fail_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._save_expand(
                notebook,
                save_env_status='fail',
                save_env_fail_last_time=fail_time
            )
            flash(__('Notebook 环境保存失败：') + str(ex), 'warning')
        return redirect(redirect_url)

    @event_logger.log_this
    @expose_api(description="保存 Notebook 为镜像", url='/save_image/<notebook_id>', methods=['GET', 'POST'])
    def save_image(self, notebook_id):
        redirect_url = conf.get('MODEL_URLS', {}).get('notebook', '')
        if not conf.get('NOTEBOOK_SAVE_ENABLED', True):
            flash(__('Notebook 环境保存功能未启用'), 'warning')
            return redirect(redirect_url)

        notebook = self._get_owned_notebook(notebook_id, for_update=True)
        expand = json.loads(notebook.expand) if notebook.expand else {}
        if (
            expand.get('save_image_status') == 'saving'
            and self._save_in_cooldown(
                expand,
                'save_image_last_request_time',
                int(conf.get('NOTEBOOK_SAVE_TIMEOUT_SEC', 1800))
            )
        ):
            flash(__('镜像正在保存中，请勿重复提交'), 'warning')
            return redirect(redirect_url)
        if self._save_in_cooldown(expand, 'save_image_last_request_time'):
            flash(__('镜像保存请求过于频繁，请稍后重试'), 'warning')
            return redirect(redirect_url)

        k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG', ''))
        pod = self._running_notebook_pod(k8s_client, notebook)
        node_name = pod.spec.node_name if pod and pod.spec else ''
        container_id = ''
        if pod and pod.status and pod.status.container_statuses:
            containers = [
                container for container in pod.status.container_statuses
                if container.name == notebook.name and container.container_id
            ]
            if containers:
                container_id = re.sub(
                    r'^(docker|containerd)://',
                    '',
                    containers[0].container_id
                )
        if container_id and not re.fullmatch(r'[a-fA-F0-9]{12,128}', container_id):
            container_id = ''
        if not node_name or not container_id:
            flash(__('没有发现正在运行的 Notebook，请先启动或 reset 后再保存镜像'), 'warning')
            return redirect(redirect_url)

        username = _safe_image_component(notebook.created_by.username, fallback='user')
        notebook_name = _safe_image_component(notebook.name)
        target_image = request.values.get('target_image', '').strip()
        if not target_image:
            target_image = '{}{}/{}:{}'.format(
                _notebook_save_image_prefix(),
                username,
                notebook_name,
                datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            )
        if not _valid_notebook_target_image(target_image):
            flash(__('目标镜像必须位于 Notebook 专用 Harbor 项目且名称合法'), 'warning')
            return redirect(redirect_url)

        all_repositories = db.session.query(Repository).all()
        repositories = [
            repo for repo in all_repositories
            if target_image == repo.server.rstrip('/')
            or target_image.startswith(repo.server.rstrip('/') + '/')
        ]
        repository_config = conf.get('NOTEBOOK_SAVE_REPOSITORY', {}) or {}
        configured_server = repository_config.get('server', '').strip().rstrip('/')
        if (
            not repositories
            and configured_server
            and target_image.startswith(configured_server + '/')
            and repository_config.get('user')
            and repository_config.get('password')
        ):
            repositories = [SimpleNamespace(
                server=configured_server,
                user=repository_config['user'],
                password=repository_config['password'],
                hubsecret=repository_config.get('hubsecret', ''),
                created_by_fk=g.user.id
            )]
        if not repositories and configured_server and target_image.startswith(configured_server + '/'):
            registry = configured_server.split('/')[0]
            repositories = [
                repo for repo in all_repositories
                if repo.server.strip().split('/')[0] == registry
            ]
        if not repositories:
            flash(__('保存镜像前，请配置 Notebook Harbor 推送凭证或添加对应镜像仓库'), 'warning')
            return redirect(conf.get('MODEL_URLS', {}).get('repository', ''))
        repo = max(
            repositories,
            key=lambda item: (
                int(item.created_by_fk == g.user.id),
                len(item.server)
            )
        )

        legacy_commit_pod_name = 'notebook-commit-{}-{}'.format(
            notebook.created_by.username,
            notebook.id
        )
        try:
            commit_pod = k8s_client.v1.read_namespaced_pod(
                name=legacy_commit_pod_name,
                namespace=notebook.namespace,
                _request_timeout=5
            )
            if commit_pod.status and commit_pod.status.phase in ('Pending', 'Running'):
                flash(__('镜像正在保存中，请勿重复提交'), 'warning')
                return redirect(redirect_url)
        except Exception:
            pass

        cluster = notebook.project.cluster
        cli_name = cluster.get('CONTAINER_CLI', conf.get('CONTAINER_CLI', 'docker'))
        if cli_name not in ('docker', 'nerdctl'):
            flash(__('集群 CONTAINER_CLI 仅支持 docker 或 nerdctl'), 'warning')
            return redirect(redirect_url)
        cli = 'nerdctl --namespace k8s.io' if cli_name == 'nerdctl' else 'docker'
        registry = repo.server.split('/')[0]
        login_command = '{} login --username {} --password {} {}'.format(
            cli_name,
            shlex.quote(repo.user),
            shlex.quote(repo.password),
            shlex.quote(registry)
        )
        command = [
            'sh',
            '-c',
            '{} && {} commit {} {} && {} push {}'.format(
                login_command,
                cli,
                shlex.quote(container_id),
                shlex.quote(target_image),
                cli,
                shlex.quote(target_image)
            )
        ]
        image_pull_secrets = conf.get('HUBSECRET', [])
        user_repositories = db.session.query(Repository).filter(
            Repository.created_by_fk == g.user.id
        ).all()
        image_pull_secrets = list(set(
            [secret for secret in image_pull_secrets if secret]
            + [repo_item.hubsecret for repo_item in user_repositories if repo_item.hubsecret]
            + ([repo.hubsecret] if repo.hubsecret else [])
        ))

        save_run_id = uuid.uuid4().hex
        commit_pod_name = 'notebook-commit-{}-{}-{}'.format(
            username[:20].rstrip('._-') or 'user',
            notebook.id,
            save_run_id[:8]
        )[:63].rstrip('-')
        request_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_expand(
            notebook,
            save_image_status='saving',
            save_target_image=target_image,
            save_image_last_request_time=request_time,
            save_image_run_id=save_run_id,
            save_image_pod_name=commit_pod_name
        )
        try:
            k8s_client.create_debug_pod(
                namespace=notebook.namespace,
                name=commit_pod_name,
                command=command,
                labels={
                    'app': 'notebook-commit',
                    'user': notebook.created_by.username,
                    'pod-type': 'notebook-commit',
                    'notebook-id': str(notebook.id),
                    'save-run-id': save_run_id
                },
                annotations={'project': notebook.project.name},
                args=None,
                volume_mount=cluster.get(
                    'DOCKER_SOCKET' if cli_name == 'docker' else 'CONTAINERD_SOCKET',
                    conf.get('DOCKER_SOCKET' if cli_name == 'docker' else 'CONTAINERD_SOCKET', '')
                ),
                working_dir='/mnt/{}'.format(notebook.created_by.username),
                node_selector=None,
                resource_memory='0~10G',
                resource_cpu='0~10',
                resource_gpu='0',
                image_pull_policy='IfNotPresent',
                image_pull_secrets=image_pull_secrets,
                image=conf.get(
                    'DOCKER_IMAGES' if cli_name == 'docker' else 'NERDCTL_IMAGES',
                    '{}:latest'.format(cli_name)
                ),
                hostAliases=conf.get('HOSTALIASES', ''),
                env={'USERNAME': notebook.created_by.username},
                privileged=True,
                accounts=None,
                username=notebook.created_by.username,
                node_name=node_name
            )
            from myapp.tasks.async_task import check_notebook_commit
            check_notebook_commit.apply_async(kwargs={
                'notebook_id': notebook.id,
                'target_image': target_image,
                'save_run_id': save_run_id,
                'commit_pod_name': commit_pod_name
            })
        except Exception as ex:
            current = (
                db.session.query(Notebook)
                .populate_existing()
                .with_for_update()
                .filter_by(id=notebook.id)
                .first()
            )
            current_expand = json.loads(current.expand) if current and current.expand else {}
            if current and current_expand.get('save_image_run_id') == save_run_id:
                try:
                    k8s_client.delete_pods(
                        namespace=notebook.namespace,
                        pod_name=commit_pod_name
                    )
                except Exception:
                    pass
                fail_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                current_expand.update({
                    'save_image_status': 'fail',
                    'save_image_fail_last_time': fail_time
                })
                current.expand = json.dumps(current_expand, ensure_ascii=False)
                db.session.commit()
            else:
                db.session.rollback()
            flash(__('Notebook 镜像保存启动失败：') + str(ex), 'warning')
            return redirect(redirect_url)

        flash(__('新镜像正在保存推送中，请留意消息通知'), 'success')
        return redirect('/k8s/web/log/{}/{}/{}'.format(
            notebook.project.cluster.get('NAME', ''),
            notebook.namespace,
            commit_pod_name
        ))

    @event_logger.log_this
    @expose_api(description="重置在线ide",url='/reset/<notebook_id>', methods=['GET', 'POST'])
    def reset(self, notebook_id):

        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        try:
            self.reset_notebook(notebook)
            flash(__('已重置，Running状态后可进入。注意：notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。'), 'info')
        except Exception as e:
            message = __('重置失败，稍后重试。') + str(e)
            flash(message, 'warning')
            return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))
            # return self.response(400, **{"message": message, "status": 1, "result": {}})

        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    @event_logger.log_this
    @expose_api(description="在线ide续期",url='/renew/<notebook_id>', methods=['GET', 'POST'])
    def renew(self, notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        notebook.changed_on = datetime.datetime.now()
        db.session.commit()
        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    # 基础批量删除
    # @pysnooper.snoop()
    def base_stop(self, items, cluster=None):
        if not items:
            return
            # abort(404)

        for item in items:
            try:
                if not cluster:
                    cluster = item.project.cluster['NAME']
                k8s_client = K8s(conf.get('CLUSTERS').get(cluster).get('KUBECONFIG', ''))
                namespace = item.namespace
                k8s_client.delete_pods(namespace=namespace,pod_name=item.name)
                commit_pod_name = "notebook-commit-%s-%s" % (item.created_by.username, str(item.id))
                k8s_client.delete_pods(namespace=namespace, pod_name=commit_pod_name)
                k8s_client.delete_pods(
                    namespace=namespace,
                    labels={
                        'pod-type': 'notebook-commit',
                        'notebook-id': str(item.id)
                    }
                )
                k8s_client.delete_service(namespace=namespace,name=item.name)
                k8s_client.delete_service(namespace=namespace, name=(item.name + "-external").lower()[:60].strip('-'))
                crd_info = conf.get("CRD_INFO", {}).get('virtualservice', {})
                if crd_info:
                    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=item.namespace, name="notebook-jupyter-%s" % item.name.replace('_', '-'))
                    # k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=item.namespace,name="ssh-notebook-jupyter-%s" % item.name.replace('_', '-'))

                self._save_expand(item, status='offline')
            except Exception as e:
                flash(str(e), "warning")

    def pre_delete(self, item):
        self.base_stop([item])

    @expose_api(description="在线ide的列表查询",url="/list/")
    @has_access
    def list(self):
        args = request.args.to_dict()
        if '_flt_0_created_by' in args and args['_flt_0_created_by'] == '':
            print(request.url)
            print(request.path)
            return redirect(request.url.replace('_flt_0_created_by=', '_flt_0_created_by=%s' % g.user.id))

        widgets = self._list()
        res = self.render_template(
            self.list_template, title=self.list_title, widgets=widgets
        )
        return res

    # @event_logger.log_this
    # @expose_api(description="",url="/delete/<pk>")
    # @has_access
    # def delete(self, pk):
    #     pk = self._deserialize_pk_if_composite(pk)
    #     self.base_delete(pk)
    #     url = url_for(f"{self.endpoint}.list")
    #     return redirect(url)

    @action("stop_all", "停止", "停止所有选中的notebook?", "fa-trash", single=False)
    def stop_all(self, items):
        self.base_stop(items)
        self.update_redirect()
        return redirect(self.get_redirect())

    @event_logger.log_this
    @expose_api(description="停止在线ide",url='/stop/<notebook_id>', methods=['GET', 'POST'])
    def stop(self, notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        self.base_stop([notebook])
        flash(_('清理完成'),'info')
        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))
    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        return self._muldelete(items)

# 添加api
class Notebook_ModelView_Api(Notebook_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Notebook)
    route_base = '/notebook_modelview/api'


appbuilder.add_api(Notebook_ModelView_Api)


# 添加api
class Notebook_ModelView_SDK_Api(Notebook_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Notebook)
    route_base = '/notebook_modelview/sdk'
    add_columns = ['project', 'name', 'describe', 'images', 'working_dir', 'volume_mount', 'resource_memory','resource_cpu', 'resource_gpu','volume_mount','image_pull_policy','expand']
    edit_columns = add_columns
    list_columns = ['project', 'ide_type_html', 'name_url', 'status', 'describe', 'reset', 'resource', 'renew']
    show_columns = ['project', 'name', 'namespace', 'describe', 'images', 'working_dir', 'env', 'volume_mount','resource_memory', 'resource_cpu', 'resource_gpu', 'status', 'ide_type', 'image_pull_policy', 'expand']
appbuilder.add_api(Notebook_ModelView_SDK_Api)
