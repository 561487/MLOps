import json
import re

import yaml
from flask import g, request
from flask_appbuilder.baseviews import expose_api
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2ManyWidget, Select2Widget
from flask_babel import lazy_gettext as _
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms import SelectField, StringField
from wtforms.validators import DataRequired, Length, Regexp

from myapp import app, appbuilder, db
from myapp.forms import MySelect2Widget, MySelectMultipleField
from myapp.models.model_storage import Storage
from myapp.models.model_team import Project
from myapp.utils.storage_volume import available_volume_items, user_can_access_project
from myapp.utils.py.py_k8s import K8s
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from myapp.views.view_team import Project_Join_Filter, filter_join_org_project
from .baseApi import MyappModelRestApi


conf = app.config

DEFAULT_NFS_SERVER = '10.121.177.20'
DEFAULT_NFS_PATH = '/data/nfs/storage'
STORAGE_TYPE_NFS = 'nfs'
STORAGE_TYPE_MINIO_JUICEFS = 'minio_juicefs'
STORAGE_TYPE_S3_MINIO_LABEL = 's3/minio'


def default_cluster():
    clusters = conf.get('CLUSTERS', {})
    environment = conf.get('ENVIRONMENT', '')
    if environment in clusters:
        return environment
    return list(clusters.keys())[0] if clusters else environment


def default_namespaces():
    namespaces = [
        conf.get('NOTEBOOK_NAMESPACE', 'jupyter'),
        conf.get('PIPELINE_NAMESPACE', 'pipeline'),
        conf.get('SERVICE_NAMESPACE', 'service'),
    ]
    return ','.join(dict.fromkeys([namespace for namespace in namespaces if namespace]))


def default_namespace_choices():
    return [[namespace, namespace] for namespace in default_namespaces().split(',') if namespace]


def default_nfs_config():
    return json.dumps({
        "nfs": {
            "server": DEFAULT_NFS_SERVER,
            "path": DEFAULT_NFS_PATH,
            "mount_options": []
        }
    }, indent=4, ensure_ascii=False)


class Storage_ModelView_Base():
    datamodel = SQLAInterface(Storage)
    label_title = _('存储资源')
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']
    base_order = ('id', 'desc')
    order_columns = ['id']
    page_size = 100

    list_columns = ['name', 'storage_type_display', 'project', 'namespace', 'capacity']
    show_columns = ['name', 'storage_type_display', 'project', 'namespace', 'capacity']
    add_columns = ['name', 'storage_type', 'project', 'namespace', 'capacity']
    edit_columns = add_columns
    search_columns = ['name', 'storage_type', 'namespace']

    cols_width = {
        "name": {"type": "ellip1", "width": 160},
        "storage_type": {"type": "ellip1", "width": 100},
        "project": {"type": "ellip1", "width": 160},
        "namespace": {"type": "ellip2", "width": 220},
        "capacity": {"type": "ellip1", "width": 120},
    }

    spec_label_columns = {
        "project": _("所属项目组"),
        "namespace": _("PVC所在命名空间"),
        "storage_type": _("存储类型"),
        "storage_type_display": _("存储类型"),
        "mount_path": _("推荐挂载路径"),
        "capacity": _("申请容量"),
        "access_modes": _("访问模式"),
        "storage_class": _("StorageClass"),
        "pv_name": _("PV名称"),
        "pvc_name": _("PVC名称"),
        "mount_expr": _("挂载表达式"),
        "config": _("配置"),
        "config_html": _("配置"),
        "remark": _("备注"),
    }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields

    add_form_extra_fields = {
        "name": StringField(
            _('名称'),
            default='storage',
            widget=BS3TextFieldWidget(),
            description=_('英文名，小写字母、数字、-组成'),
            validators=[DataRequired(), Length(1, 50), Regexp('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')]
        ),
        "storage_type": SelectField(
            _('存储类型'),
            widget=Select2Widget(),
            default=STORAGE_TYPE_S3_MINIO_LABEL,
            choices=[[STORAGE_TYPE_NFS, 'nfs'], [STORAGE_TYPE_MINIO_JUICEFS, STORAGE_TYPE_S3_MINIO_LABEL]],
            description=_('对象存储类型为nfs或s3/minio')
        ),
        "project": QuerySelectField(
            _('所属项目组'),
            default='',
            query_factory=filter_join_org_project,
            widget=MySelect2Widget(new_web=False),
            validators=[DataRequired()]
        ),
        "namespace": MySelectMultipleField(
            _('PVC所在命名空间'),
            default=conf.get('NOTEBOOK_NAMESPACE', 'jupyter'),
            widget=Select2ManyWidget(),
            choices=default_namespace_choices(),
            description=_('选择需要创建PVC的命名空间，可多选'),
            validators=[DataRequired()]
        ),
        "capacity": StringField(
            _('申请容量'),
            default='5Gi',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp('^[0-9]+(Mi|Gi|Ti)$')]
        ),
    }
    edit_form_extra_fields = add_form_extra_fields

    def _assert_admin(self):
        if not g.user.is_admin():
            raise Exception('only admin can manage storage')

    def _split_names(self, value):
        value = value or ''
        return [item.strip() for item in re.split(',|;|\\n|\\t', value) if item.strip()]

    def _namespaces(self, storage):
        namespaces = self._split_names(storage.namespace)
        if namespaces:
            return namespaces
        return self._split_names(default_namespaces())

    def _access_modes(self, storage):
        access_modes = self._split_names(storage.access_modes)
        return access_modes or ['ReadWriteMany']

    def _load_config(self, config_value):
        if isinstance(config_value, dict):
            config = config_value
        else:
            config = json.loads(config_value or '{}')

        if 'nfs' not in config:
            config = {
                "nfs": {
                    "server": config.get('server', DEFAULT_NFS_SERVER),
                    "path": config.get('path', DEFAULT_NFS_PATH),
                    "mount_options": config.get('mount_options', [])
                }
            }

        nfs_config = config.get('nfs') or {}
        if not nfs_config.get('server'):
            nfs_config['server'] = DEFAULT_NFS_SERVER
        if not nfs_config.get('path'):
            nfs_config['path'] = DEFAULT_NFS_PATH
        if 'mount_options' not in nfs_config:
            nfs_config['mount_options'] = []
        config['nfs'] = nfs_config
        return config

    def _cluster_config(self, storage, key, default=''):
        clusters = conf.get('CLUSTERS', {})
        cluster_config = clusters.get(storage.cluster, {}) if clusters else {}
        return cluster_config.get(key, conf.get(key, default))

    def _safe_bucket_path_part(self, value):
        value = (value or '').strip().lower()
        value = re.sub('[^a-z0-9-]+', '-', value).strip('-')
        return value or 'default'

    def _load_minio_juicefs_config(self, storage):
        try:
            config = json.loads(storage.config or '{}')
        except Exception:
            config = {}
        minio_config = config.get(STORAGE_TYPE_MINIO_JUICEFS) or {}
        project_name = self._safe_bucket_path_part(storage.project.name if storage.project else '')
        bucket_prefix = self._cluster_config(storage, 'STORAGE_JUICEFS_BUCKET_PREFIX', 'projects')
        bucket_path = minio_config.get('bucket_path') or '/'.join([
            self._safe_bucket_path_part(bucket_prefix),
            project_name,
            self._safe_bucket_path_part(storage.name),
        ])
        minio_config.update({
            "backend": minio_config.get("backend") or "minio",
            "filesystem": minio_config.get("filesystem") or "juicefs",
            "storage_class": minio_config.get("storage_class") or self._cluster_config(storage, 'STORAGE_JUICEFS_STORAGE_CLASS', 'juicefs-sc'),
            "bucket": minio_config.get("bucket") or self._cluster_config(storage, 'STORAGE_JUICEFS_BUCKET', 'juicefs'),
            "bucket_path": bucket_path,
            "secret_name": minio_config.get("secret_name") or self._cluster_config(storage, 'STORAGE_JUICEFS_SECRET_NAME', 'juicefs-minio-secret'),
            "secret_namespace": minio_config.get("secret_namespace") or self._cluster_config(storage, 'STORAGE_JUICEFS_SECRET_NAMESPACE', 'kube-system'),
            "pvc_annotations": minio_config.get("pvc_annotations") or {},
            "mount_options": minio_config.get("mount_options") or [],
        })
        return minio_config

    def _project_from_value(self, value):
        if isinstance(value, Project):
            return value
        if isinstance(value, dict):
            value = value.get('id') or value.get('value')
        if value:
            return db.session.query(Project).filter_by(id=int(value)).filter_by(type='org').first()
        return None

    def _project_namespaces(self, project):
        if not project:
            return default_namespaces()
        namespaces = [
            project.notebook_namespace,
            project.pipeline_namespace,
            project.service_namespace,
        ]
        return ','.join(dict.fromkeys([namespace for namespace in namespaces if namespace]))

    def _normalize_namespace_value(self, value):
        if isinstance(value, list):
            namespaces = []
            for item in value:
                namespaces.extend(self._split_names(item))
        else:
            namespaces = self._split_names(value)
        return ','.join(dict.fromkeys(namespaces))

    def _validate_namespaces(self, value, project):
        namespace = self._normalize_namespace_value(value)
        namespaces = self._split_names(namespace)
        if not namespaces:
            raise Exception('namespace is required')

        allowed_namespaces = self._split_names(self._project_namespaces(project))
        invalid_namespaces = [item for item in namespaces if item not in allowed_namespaces]
        if invalid_namespaces:
            raise Exception('namespace must be one of: %s' % ','.join(allowed_namespaces))
        return namespace

    def _project_cluster(self, project):
        return project.cluster_name if project else default_cluster()

    def _backend_config(self, name, project, storage_type):
        if storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            bucket_prefix = conf.get('STORAGE_JUICEFS_BUCKET_PREFIX', 'projects')
            bucket_path = '/'.join([
                self._safe_bucket_path_part(bucket_prefix),
                self._safe_bucket_path_part(project.name if project else ''),
                self._safe_bucket_path_part(name),
            ])
            return {
                "minio_juicefs": {
                    "backend": "minio",
                    "filesystem": "juicefs",
                    "storage_class": conf.get('STORAGE_JUICEFS_STORAGE_CLASS', 'juicefs-sc'),
                    "bucket": conf.get('STORAGE_JUICEFS_BUCKET', 'juicefs'),
                    "bucket_path": bucket_path,
                    "secret_name": conf.get('STORAGE_JUICEFS_SECRET_NAME', 'juicefs-minio-secret'),
                    "secret_namespace": conf.get('STORAGE_JUICEFS_SECRET_NAMESPACE', 'kube-system'),
                    "pvc_annotations": {},
                    "mount_options": [],
                }
            }
        return {
            "nfs": {
                "server": DEFAULT_NFS_SERVER,
                "path": DEFAULT_NFS_PATH,
                "mount_options": []
            }
        }

    def _apply_backend_defaults(self, storage):
        project = storage.project
        storage.label = storage.label or storage.name
        storage.cluster = self._project_cluster(project)
        storage.namespace = self._validate_namespaces(storage.namespace, project)
        storage.mount_path = f'/mnt/storage/{storage.name}'
        storage.access_modes = 'ReadWriteMany'
        storage.pvc_name = storage.pvc_name or storage.name
        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            try:
                config = json.loads(storage.config or '{}')
            except Exception:
                config = {}
            if STORAGE_TYPE_MINIO_JUICEFS not in config:
                storage.config = json.dumps(
                    self._backend_config(storage.name, project, storage.storage_type),
                    indent=4,
                    ensure_ascii=False
                )
            minio_config = self._load_minio_juicefs_config(storage)
            storage.storage_class = minio_config.get('storage_class', 'juicefs-sc')
            storage.config = json.dumps({STORAGE_TYPE_MINIO_JUICEFS: minio_config}, indent=4, ensure_ascii=False)
        elif not storage.status:
            storage.config = json.dumps(
                self._backend_config(storage.name, project, storage.storage_type),
                indent=4,
                ensure_ascii=False
            )
            storage.storage_class = ''
            storage.status = 'draft'
        else:
            storage.config = json.dumps(
                self._backend_config(storage.name, project, storage.storage_type),
                indent=4,
                ensure_ascii=False
            )
            storage.storage_class = ''
        return storage

    def _validate_req(self, req_json):
        self._assert_admin()

        name = (req_json.get('name') or '').strip()
        if not re.match('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', name):
            raise Exception('name must match k8s dns label: lowercase letters, numbers and -')
        if len(name) > 50:
            raise Exception('name is too long, max 50 characters')

        storage_type = (req_json.get('storage_type') or STORAGE_TYPE_NFS).strip().lower()
        if storage_type in ['minio/juicefs', STORAGE_TYPE_S3_MINIO_LABEL]:
            storage_type = STORAGE_TYPE_MINIO_JUICEFS
        if storage_type not in [STORAGE_TYPE_NFS, STORAGE_TYPE_MINIO_JUICEFS]:
            raise Exception('storage_type must be nfs or s3/minio')

        project = self._project_from_value(req_json.get('project') or req_json.get('project_id'))
        if not project:
            raise Exception('project is required')

        req_json['name'] = name
        req_json['storage_type'] = storage_type
        req_json['project'] = project.id
        req_json['label'] = req_json.get('label') or name
        req_json['cluster'] = self._project_cluster(project)
        req_json['namespace'] = self._validate_namespaces(req_json.get('namespace'), project)
        req_json['mount_path'] = req_json.get('mount_path') or f'/mnt/storage/{name}'
        req_json['capacity'] = req_json.get('capacity') or '500Gi'
        req_json['access_modes'] = 'ReadWriteMany'
        req_json['storage_class'] = ''
        req_json['config'] = json.dumps(self._backend_config(name, project, storage_type), indent=4, ensure_ascii=False)
        if storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            req_json['storage_class'] = conf.get('STORAGE_JUICEFS_STORAGE_CLASS', 'juicefs-sc')
        req_json['status'] = req_json.get('status') or 'draft'
        return req_json

    def pre_add_req(self, req_json, *args, **kwargs):
        return self._validate_req(req_json)

    def pre_update_req(self, req_json, *args, **kwargs):
        src_item = kwargs.get('src_item')
        if src_item:
            merged = {
                "name": src_item.name,
                "label": src_item.label,
                "storage_type": src_item.storage_type,
                "project": src_item.project_id,
                "cluster": src_item.cluster,
                "namespace": src_item.namespace,
                "mount_path": src_item.mount_path,
                "capacity": src_item.capacity,
                "access_modes": src_item.access_modes,
                "storage_class": src_item.storage_class,
                "config": src_item.config,
                "remark": src_item.remark,
                "status": src_item.status,
            }
            merged.update(req_json)
            return self._validate_req(merged)
        return self._validate_req(req_json)

    def _pv_name(self, storage, namespace):
        cluster = (storage.cluster or default_cluster()).lower()
        pv_name = f'storage-{cluster}-{namespace}-{storage.name}'
        return re.sub('[^a-z0-9-]', '-', pv_name.lower())[-63:].strip('-')

    def _fill_generated_fields(self, storage):
        self._apply_backend_defaults(storage)
        if storage.storage_type == STORAGE_TYPE_NFS:
            storage.pv_name = ','.join([self._pv_name(storage, namespace) for namespace in self._namespaces(storage)])
        else:
            storage.pv_name = ''
        return storage

    def post_add(self, item):
        self._fill_generated_fields(item)
        db.session.commit()
        if item.storage_type in [STORAGE_TYPE_NFS, STORAGE_TYPE_MINIO_JUICEFS]:
            try:
                self._sync_storage(item)
            except Exception:
                db.session.rollback()
                item.status = 'error'
                db.session.commit()
                raise

    def pre_update(self, item):
        self._fill_generated_fields(item)

    def check_edit_permission(self, item):
        return g.user.is_admin()

    check_delete_permission = check_edit_permission

    def pre_delete(self, item):
        self._delete_storage_resources(item)

    def _build_nfs_resources(self, storage):
        storage = self._fill_generated_fields(storage)
        config = self._load_config(storage.config)
        nfs_config = config['nfs']
        resources = []
        for namespace in self._namespaces(storage):
            pv_name = self._pv_name(storage, namespace)
            pvc_name = storage.pvc_name or storage.name
            labels = {
                "mlops/storage-name": storage.name,
                "mlops/storage-type": storage.storage_type,
            }
            pv_spec = {
                "capacity": {"storage": storage.capacity or '500Gi'},
                "accessModes": self._access_modes(storage),
                "persistentVolumeReclaimPolicy": "Delete",
                "storageClassName": storage.storage_class or "",
                "nfs": {
                    "server": nfs_config["server"],
                    "path": nfs_config["path"],
                }
            }
            mount_options = nfs_config.get('mount_options') or []
            if mount_options:
                pv_spec["mountOptions"] = mount_options
            pv = {
                "apiVersion": "v1",
                "kind": "PersistentVolume",
                "metadata": {
                    "name": pv_name,
                    "labels": labels,
                },
                "spec": pv_spec,
            }
            pvc = {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": pvc_name,
                    "namespace": namespace,
                    "labels": labels,
                },
                "spec": {
                    "accessModes": self._access_modes(storage),
                    "resources": {
                        "requests": {
                            "storage": storage.capacity or '500Gi'
                        }
                    },
                    "storageClassName": storage.storage_class or "",
                    "volumeName": pv_name,
                }
            }
            resources.append({
                "namespace": namespace,
                "pv": pv,
                "pvc": pvc,
            })
        return resources

    def _build_minio_juicefs_resources(self, storage):
        storage = self._fill_generated_fields(storage)
        minio_config = self._load_minio_juicefs_config(storage)
        storage_class = minio_config.get('storage_class') or 'juicefs-sc'
        resources = []
        for namespace in self._namespaces(storage):
            pvc_name = storage.pvc_name or storage.name
            labels = {
                "mlops/storage-name": storage.name,
                "mlops/storage-type": storage.storage_type,
                "mlops/storage-backend": "minio",
            }
            metadata = {
                "name": pvc_name,
                "namespace": namespace,
                "labels": labels,
            }
            annotations = minio_config.get('pvc_annotations') or {}
            if minio_config.get('bucket_path'):
                annotations.setdefault('juicefs.com/path', minio_config['bucket_path'])
            if annotations:
                metadata["annotations"] = annotations
            pvc = {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": metadata,
                "spec": {
                    "accessModes": self._access_modes(storage),
                    "resources": {
                        "requests": {
                            "storage": storage.capacity or '500Gi'
                        }
                    },
                    "storageClassName": storage_class,
                }
            }
            resources.append({
                "namespace": namespace,
                "pvc": pvc,
            })
        return resources

    def _build_resources(self, storage):
        if storage.storage_type == STORAGE_TYPE_NFS:
            return self._build_nfs_resources(storage)
        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            return self._build_minio_juicefs_resources(storage)
        raise Exception('unsupported storage_type')

    def _get_item(self, storage_id):
        storage = self.datamodel.get(storage_id)
        if not storage:
            raise Exception('storage not found')
        return storage

    def _project_from_args(self):
        project_id = request.args.get('project_id') or request.args.get('project')
        project_name = request.args.get('project_name') or request.args.get('project')
        project = None
        if project_id and str(project_id).isdigit():
            project = db.session.query(Project).filter_by(id=int(project_id)).filter_by(type='org').first()
        elif project_name:
            project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if project and not user_can_access_project(g.user, project):
            raise Exception('no permission')
        return project

    def _k8s_client(self, storage):
        clusters = conf.get('CLUSTERS', {})
        kubeconfig = clusters.get(storage.cluster, {}).get('KUBECONFIG', '') if clusters else ''
        return K8s(kubeconfig, cluster_name=storage.cluster)

    def _sync_nfs_storage(self, storage):
        k8s_client = self._k8s_client(storage)
        resources = self._build_nfs_resources(storage)
        result = []
        for resource in resources:
            pv_status = k8s_client.create_or_patch_pv(resource['pv'])
            pvc_status = k8s_client.create_or_patch_pvc(resource['namespace'], resource['pvc'])
            result.append({
                "namespace": resource['namespace'],
                "pv": pv_status,
                "pvc": pvc_status,
            })
        storage.status = 'synced'
        storage.pv_name = ','.join([resource['pv']['metadata']['name'] for resource in resources])
        storage.pvc_name = storage.pvc_name or storage.name
        db.session.commit()
        return result

    def _sync_minio_juicefs_storage(self, storage):
        k8s_client = self._k8s_client(storage)
        resources = self._build_minio_juicefs_resources(storage)
        result = []
        pv_names = []
        minio_config = self._load_minio_juicefs_config(storage)
        for resource in resources:
            pvc_status = k8s_client.create_or_patch_pvc(resource['namespace'], resource['pvc'])
            if pvc_status.get('volume_name'):
                pv_names.append(pvc_status['volume_name'])
            result.append({
                "namespace": resource['namespace'],
                "pvc": pvc_status,
            })
        storage.status = 'synced'
        storage.pv_name = ','.join(sorted(set(pv_names)))
        storage.pvc_name = storage.pvc_name or storage.name
        storage.storage_class = minio_config.get('storage_class') or 'juicefs-sc'
        storage.config = json.dumps({STORAGE_TYPE_MINIO_JUICEFS: minio_config}, indent=4, ensure_ascii=False)
        db.session.commit()
        return result

    def _sync_storage(self, storage):
        if storage.storage_type == STORAGE_TYPE_NFS:
            return self._sync_nfs_storage(storage)
        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            return self._sync_minio_juicefs_storage(storage)
        raise Exception('unsupported storage_type')

    def _delete_storage_resources(self, storage):
        if storage.storage_type not in [STORAGE_TYPE_NFS, STORAGE_TYPE_MINIO_JUICEFS]:
            return []

        k8s_client = self._k8s_client(storage)
        resources = self._build_resources(storage)
        result = []
        pv_names = set(self._split_names(storage.pv_name))

        for resource in resources:
            pvc = resource.get('pvc') or {}
            pvc_name = pvc.get('metadata', {}).get('name') or storage.pvc_name or storage.name
            namespace = resource.get('namespace')
            if pvc_name and namespace:
                pvc_status = k8s_client.get_pvc(name=pvc_name, namespace=namespace)
                if pvc_status.get('volume_name'):
                    pv_names.add(pvc_status['volume_name'])
                result.append({
                    "namespace": namespace,
                    "pvc": k8s_client.delete_pvc(namespace=namespace, name=pvc_name),
                })

            pv = resource.get('pv') or {}
            pv_name = pv.get('metadata', {}).get('name')
            if pv_name:
                pv_names.add(pv_name)

        for pv_name in sorted(pv_names):
            result.append({
                "pv": k8s_client.delete_pv(pv_name),
            })

        return result

    @expose_api(description="检查存储资源状态", url="/check/<storage_id>", methods=["GET"])
    def check(self, storage_id):
        try:
            self._assert_admin()
            storage = self._get_item(storage_id)
            k8s_client = self._k8s_client(storage)
            result = []
            for resource in self._build_resources(storage):
                pv_status = None
                if resource.get('pv'):
                    pv_status = k8s_client.get_pv(resource['pv']['metadata']['name'])
                pvc_status = k8s_client.get_pvc(resource['pvc']['metadata']['name'], resource['namespace'])
                result.append({
                    "namespace": resource['namespace'],
                    "pv": pv_status,
                    "pvc": pvc_status,
                })
            return self.response(200, status=0, message='success', result=result)
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="查看存储资源YAML", url="/manifest/<storage_id>", methods=["GET"])
    def manifest(self, storage_id):
        try:
            self._assert_admin()
            storage = self._get_item(storage_id)
            resources = self._build_resources(storage)
            manifests = []
            for resource in resources:
                if resource.get('pv'):
                    manifests.append(resource['pv'])
                manifests.append(resource['pvc'])
            yaml_text = yaml.safe_dump_all(manifests, allow_unicode=True, sort_keys=False)
            return self.response(200, status=0, message='success', result={
                "items": resources,
                "yaml": yaml_text,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="获取挂载表达式", url="/mount_expr/<storage_id>", methods=["GET"])
    def mount_expr(self, storage_id):
        try:
            self._assert_admin()
            storage = self._get_item(storage_id)
            return self.response(200, status=0, message='success', result={
                "mount_expr": storage.mount_expr,
                "volume_mount": storage.mount_expr,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="获取项目可选挂载卷", url="/available_volumes", methods=["GET"])
    def available_volumes(self):
        try:
            project = self._project_from_args()
            namespace = request.args.get('namespace') or ''
            current_volume_mount = request.args.get('volume_mount') or ''
            volumes = available_volume_items(
                g.user,
                project=project,
                namespace=namespace,
                current_volume_mount=current_volume_mount,
                include_system=True,
            )
            return self.response(200, status=0, message='success', result={
                "project": project.name if project else '',
                "volumes": volumes,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))


class Storage_ModelView_Api(Storage_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Storage)
    route_base = '/storage_modelview/api'


appbuilder.add_api(Storage_ModelView_Api)
