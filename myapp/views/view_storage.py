import base64
import json
import os
import posixpath
import re
import uuid
from datetime import datetime
from urllib.parse import quote

import yaml
from flask import Response, g, render_template, request
from flask_appbuilder.baseviews import expose_api
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2ManyWidget, Select2Widget
from flask_babel import lazy_gettext as _
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms import SelectField, StringField
from wtforms.validators import DataRequired, Length, Regexp

from myapp import app, appbuilder, db
from myapp.forms import MySelect2Widget, MySelectMultipleField
from myapp.models.model_storage import Storage, StoragePvcBinding
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

    list_columns = ['name', 'storage_type_display', 'project', 'namespace', 'capacity', 'status', 'files_html']
    show_columns = ['name', 'storage_type_display', 'project', 'namespace', 'capacity', 'status', 'bindings_html']
    add_columns = ['name', 'storage_type', 'project', 'namespace', 'capacity']
    edit_columns = add_columns
    search_columns = ['name', 'storage_type', 'namespace']

    cols_width = {
        "name": {"type": "ellip1", "width": 160},
        "storage_type": {"type": "ellip1", "width": 100},
        "project": {"type": "ellip1", "width": 160},
        "namespace": {"type": "ellip2", "width": 220},
        "capacity": {"type": "ellip1", "width": 120},
        "files_html": {"type": "ellip1", "width": 80},
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
        "files_html": _("文件"),
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
            description=_('单选：创建独立PVC。多选：会在多个命名空间分别创建PVC，并要求它们共享同一个底层存储路径。如果希望各命名空间数据独立，请分别创建多个存储资源。'),
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

    def _is_multi_namespace(self, storage):
        return len(self._namespaces(storage)) > 1

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

    def _apply_storage_class_secret_ref(self, storage, minio_config):
        minio_config = dict(minio_config or {})
        storage_class = minio_config.get('storage_class') or 'juicefs-sc'
        try:
            sc = self._k8s_client(storage).get_storage_class(storage_class)
            params = sc.get('parameters') or {}
            secret_name = params.get('csi.storage.k8s.io/node-publish-secret-name')
            secret_namespace = params.get('csi.storage.k8s.io/node-publish-secret-namespace')
            if secret_name:
                minio_config['secret_name'] = secret_name
            if secret_namespace:
                minio_config['secret_namespace'] = secret_namespace
        except Exception:
            pass
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
        if storage.storage_type == STORAGE_TYPE_NFS or (
                storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS and self._is_multi_namespace(storage)):
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

    def pre_show(self, item):
        try:
            self._refresh_storage_bindings(item)
        except Exception:
            pass

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
        minio_config = self._apply_storage_class_secret_ref(storage, self._load_minio_juicefs_config(storage))
        storage_class = minio_config.get('storage_class') or 'juicefs-sc'
        resources = []
        static_binding = self._is_multi_namespace(storage)
        for namespace in self._namespaces(storage):
            pvc_name = storage.pvc_name or storage.name
            pv_name = self._pv_name(storage, namespace)
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
            resource = {
                "namespace": namespace,
                "pvc": pvc,
            }
            if static_binding:
                pv = {
                    "apiVersion": "v1",
                    "kind": "PersistentVolume",
                    "metadata": {
                        "name": pv_name,
                        "labels": labels,
                    },
                    "spec": {
                        "capacity": {"storage": storage.capacity or '500Gi'},
                        "accessModes": self._access_modes(storage),
                        "persistentVolumeReclaimPolicy": "Delete",
                        "storageClassName": storage_class,
                        "csi": {
                            "driver": minio_config.get("csi_driver") or "csi.juicefs.com",
                            "fsType": "juicefs",
                            "volumeHandle": pv_name,
                            "nodePublishSecretRef": {
                                "name": minio_config.get("secret_name"),
                                "namespace": minio_config.get("secret_namespace"),
                            },
                            "volumeAttributes": {
                                "backend": minio_config.get("backend") or "minio",
                                "filesystem": minio_config.get("filesystem") or "juicefs",
                                "bucket": minio_config.get("bucket") or "",
                                "subPath": minio_config.get("bucket_path") or "",
                            },
                        },
                    },
                }
                mount_options = minio_config.get('mount_options') or []
                if mount_options:
                    pv["spec"]["mountOptions"] = mount_options
                pvc["spec"]["volumeName"] = pv_name
                resource["pv"] = pv
            resources.append({
                **resource,
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

    def _check_storage_permission(self, storage, write=False):
        if g.user.is_admin():
            return True
        if user_can_access_project(g.user, storage.project):
            return True
        raise Exception('no permission')

    def _file_manager_namespace(self, storage):
        entry = self._file_manager_entry(storage)
        if entry.get('namespace'):
            return entry['namespace']
        raise Exception('namespace is required')

    def _file_manager_entry(self, storage):
        order = self._candidate_file_manager_namespaces(storage)
        order_index = {namespace: index for index, namespace in enumerate(order)}
        bindings = db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).all()
        bindings = sorted(bindings, key=lambda item: (
            item.status != 'Bound',
            order_index.get(item.namespace, len(order_index)),
            item.namespace or '',
        ))
        if bindings:
            binding = bindings[0]
            return {
                "namespace": binding.namespace,
                "pvc_name": binding.pvc_name or storage.pvc_name or storage.name,
                "pv_name": binding.pv_name or '',
                "status": binding.status or '',
                "backend_match_status": binding.backend_match_status or '',
            }
        namespace = order[0] if order else ''
        return {
            "namespace": namespace,
            "pvc_name": storage.pvc_name or storage.name,
            "pv_name": '',
            "status": '',
            "backend_match_status": '',
        }

    def _candidate_file_manager_namespaces(self, storage):
        candidates = []
        if storage.project:
            for attr in ['pipeline_namespace', 'service_namespace', 'notebook_namespace']:
                try:
                    namespace = getattr(storage.project, attr)
                    if namespace:
                        candidates.append(namespace)
                except Exception:
                    pass
        candidates.extend([
            conf.get('PIPELINE_NAMESPACE', 'pipeline'),
            conf.get('SERVICE_NAMESPACE', 'service'),
            conf.get('NOTEBOOK_NAMESPACE', 'jupyter'),
        ])
        candidates.extend(self._namespaces(storage))
        return list(dict.fromkeys([namespace for namespace in candidates if namespace]))

    def _assert_file_manager_allowed(self, storage):
        if storage.status == 'backend_mismatch':
            raise Exception('storage backend mismatch, please verify shared backend before file management')

    def _resolve_file_manager_pvc(self, storage, k8s_client):
        self._assert_file_manager_allowed(storage)
        pvc_name = storage.pvc_name or storage.name
        if not pvc_name:
            raise Exception('pvc_name is required')

        def find_pvc():
            pending_pvc = None
            checked_namespaces = []
            bindings = db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).all()
            order = self._candidate_file_manager_namespaces(storage)
            order_index = {namespace: index for index, namespace in enumerate(order)}
            bindings = sorted(bindings, key=lambda item: (
                item.status != 'Bound',
                order_index.get(item.namespace, len(order_index)),
                item.namespace or '',
            ))
            for binding in bindings:
                checked_namespaces.append(binding.namespace)
                pvc = k8s_client.get_pvc(name=binding.pvc_name or pvc_name, namespace=binding.namespace)
                if not pvc:
                    continue
                if pvc.get('status') == 'Bound':
                    return binding.namespace, pvc
                pending_pvc = (binding.namespace, pvc)
            for namespace in order:
                if namespace in checked_namespaces:
                    continue
                pvc = k8s_client.get_pvc(name=pvc_name, namespace=namespace)
                if not pvc:
                    continue
                if pvc.get('status') == 'Bound':
                    return namespace, pvc
                pending_pvc = (namespace, pvc)
            if pending_pvc:
                namespace, pvc = pending_pvc
                raise Exception('PVC {} in namespace {} is {}, please wait until it is Bound'.format(
                    pvc_name, namespace, pvc.get('status') or 'unknown'
                ))
            return None, None

        namespace, pvc = find_pvc()
        if namespace and pvc:
            return namespace, pvc

        if storage.storage_type in [STORAGE_TYPE_NFS, STORAGE_TYPE_MINIO_JUICEFS]:
            self._sync_storage(storage)
            namespace, pvc = find_pvc()
            if namespace and pvc:
                return namespace, pvc

        raise Exception('PVC {} not found in namespaces: {}'.format(
            pvc_name, ','.join(self._candidate_file_manager_namespaces(storage))
        ))

    def _ensure_file_manager(self, storage):
        k8s_client = self._k8s_client(storage)
        namespace, pvc = self._resolve_file_manager_pvc(storage, k8s_client)
        pod = k8s_client.ensure_storage_file_manager_pod(
            namespace=namespace,
            storage_name=storage.name,
            pvc_name=pvc['name'],
            image=conf.get('STORAGE_FILE_MANAGER_IMAGE', 'busybox:1.36'),
            mount_path='/mnt/storage',
            timeout=int(conf.get('STORAGE_FILE_MANAGER_POD_TIMEOUT', 30)),
        )
        return k8s_client, pod

    def _ensure_file_manager_for_binding(self, storage, k8s_client, binding):
        pod = k8s_client.ensure_storage_file_manager_pod(
            namespace=binding.namespace,
            storage_name=storage.name,
            pvc_name=binding.pvc_name or storage.pvc_name or storage.name,
            image=conf.get('STORAGE_FILE_MANAGER_IMAGE', 'busybox:1.36'),
            mount_path='/mnt/storage',
            timeout=int(conf.get('STORAGE_FILE_MANAGER_POD_TIMEOUT', 30)),
        )
        return pod

    def _verified_bindings(self, storage):
        bindings = db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).order_by(
            StoragePvcBinding.namespace.asc()
        ).all()
        namespaces = set(self._namespaces(storage))
        return [binding for binding in bindings if binding.namespace in namespaces]

    def _verify_shared_storage(self, storage):
        if len(self._namespaces(storage)) <= 1:
            storage.status = 'shared_verified'
            db.session.commit()
            return {"verified": True, "message": "single namespace storage", "items": []}

        if not self._verified_bindings(storage):
            self._sync_storage(storage)

        match_status = self._refresh_backend_match(storage)
        if match_status != 'matched':
            db.session.commit()
            raise Exception('backend identity is {}, active shared verification is blocked'.format(match_status))

        bindings = self._verified_bindings(storage)
        if len(bindings) <= 1:
            raise Exception('at least two namespace bindings are required')

        k8s_client = self._k8s_client(storage)
        token = '{}-{}'.format(storage.name, uuid.uuid4().hex)
        verify_path = '/mnt/storage/.mlops-shared-verify-{}.txt'.format(uuid.uuid4().hex)
        writer = bindings[0]
        pods = []
        try:
            writer_pod = self._ensure_file_manager_for_binding(storage, k8s_client, writer)
            pods.append(writer_pod)
            k8s_client.exec_pod(
                writer_pod['name'],
                writer_pod['namespace'],
                "printf %s {} > {}".format(self._sh_quote(token), self._sh_quote(verify_path)),
            )

            results = []
            for binding in bindings[1:]:
                pod = self._ensure_file_manager_for_binding(storage, k8s_client, binding)
                pods.append(pod)
                output = k8s_client.exec_pod(
                    pod['name'],
                    pod['namespace'],
                    "cat {}".format(self._sh_quote(verify_path)),
                )
                matched = (output or '').strip() == token
                results.append({
                    "namespace": binding.namespace,
                    "pvc_name": binding.pvc_name,
                    "matched": matched,
                })
                if not matched:
                    storage.status = 'backend_mismatch'
                    for item in bindings:
                        item.backend_match_status = 'mismatch'
                        item.backend_match_message = 'active shared verification failed'
                    db.session.commit()
                    return {
                        "verified": False,
                        "writer": writer.namespace,
                        "items": results,
                    }

            storage.status = 'shared_verified'
            for item in bindings:
                item.backend_match_status = 'matched'
                item.backend_match_message = 'active shared verification passed'
            db.session.commit()
            return {
                "verified": True,
                "writer": writer.namespace,
                "items": results,
            }
        except Exception:
            db.session.rollback()
            storage.status = 'error'
            for item in self._verified_bindings(storage):
                item.backend_match_status = 'error'
                item.backend_match_message = 'active shared verification error'
            db.session.commit()
            raise
        finally:
            for pod in pods:
                try:
                    k8s_client.exec_pod(
                        pod['name'],
                        pod['namespace'],
                        "rm -f -- {}".format(self._sh_quote(verify_path)),
                    )
                except Exception:
                    pass

    def _normalize_storage_path(self, value, allow_root=True):
        value = (value or '/').strip()
        if '\x00' in value:
            raise Exception('invalid path: null byte detected')
        if '\\' in value:
            raise Exception('invalid path: backslash is not allowed')
        if ':' in value:
            raise Exception('invalid path: colon is not allowed')
        parts = [part for part in value.split('/') if part]
        if any(part == '..' for part in parts):
            raise Exception('path traversal is not allowed')
        if not value.startswith('/'):
            value = '/' + value
        normalized = posixpath.normpath(value)
        if normalized == '.':
            normalized = '/'
        if normalized != '/' and normalized.endswith('/'):
            normalized = normalized.rstrip('/')
        if normalized == '/' and not allow_root:
            raise Exception('root path is not allowed')
        return normalized

    def _pod_path(self, storage_path):
        storage_path = self._normalize_storage_path(storage_path)
        if storage_path == '/':
            return '/mnt/storage'
        return '/mnt/storage{}'.format(storage_path)

    def _sh_quote(self, value):
        return "'{}'".format((value or '').replace("'", "'\\''"))

    def _format_mtime(self, value):
        try:
            return datetime.fromtimestamp(int(float(value))).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return ''

    def _file_operation_context(self, storage_id, write=False):
        storage = self._get_item(storage_id)
        self._check_storage_permission(storage, write=write)
        return storage, self._ensure_file_manager(storage)

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

    def _backend_identity_from_status(self, storage, pv_status, pvc_status=None):
        pv_status = pv_status or {}
        pvc_status = pvc_status or {}
        if storage.storage_type == STORAGE_TYPE_NFS:
            nfs = pv_status.get('nfs') or {}
            return {
                "type": STORAGE_TYPE_NFS,
                "server": nfs.get('server') or '',
                "path": nfs.get('path') or '',
            }

        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            csi = pv_status.get('csi') or {}
            attrs = csi.get('volume_attributes') or {}
            secret_ref = csi.get('node_publish_secret_ref') or {}
            return {
                "type": STORAGE_TYPE_MINIO_JUICEFS,
                "driver": csi.get('driver') or '',
                "secret": {
                    "name": secret_ref.get('name') or '',
                    "namespace": secret_ref.get('namespace') or '',
                },
                "filesystem": attrs.get('filesystem') or attrs.get('name') or '',
                "bucket": attrs.get('bucket') or '',
                "path": attrs.get('subPath') or '',
                "legacyPath": attrs.get('path') or '',
                "storageClassName": pv_status.get('storage_class') or pvc_status.get('storage_class') or '',
            }

        return {}

    def _identity_key(self, identity):
        return json.dumps(identity or {}, sort_keys=True, ensure_ascii=False)

    def _identity_has_backend_fields(self, storage, identity):
        identity = identity or {}
        if storage.storage_type == STORAGE_TYPE_NFS:
            return bool(identity.get('server') and identity.get('path'))
        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            secret = identity.get('secret') or {}
            return bool(
                identity.get('driver')
                and secret.get('name')
                and secret.get('namespace')
                and identity.get('filesystem')
                and identity.get('bucket')
                and identity.get('path')
            )
        return bool(identity)

    def _binding_for_namespace(self, storage, namespace):
        binding = db.session.query(StoragePvcBinding).filter_by(
            storage_id=storage.id,
            namespace=namespace,
        ).first()
        if binding:
            return binding
        binding = StoragePvcBinding(storage_id=storage.id, namespace=namespace)
        db.session.add(binding)
        return binding

    def _sync_binding_record(self, storage, namespace, pvc_status=None, pv_status=None, resource=None):
        pvc_status = pvc_status or {}
        pv_status = pv_status or {}
        resource = resource or {}
        resource_pvc = resource.get('pvc') or {}
        resource_pv = resource.get('pv') or {}
        pvc_name = (
            pvc_status.get('name')
            or resource_pvc.get('metadata', {}).get('name')
            or storage.pvc_name
            or storage.name
        )
        pv_name = (
            pvc_status.get('volume_name')
            or pv_status.get('name')
            or resource_pv.get('metadata', {}).get('name')
            or ''
        )
        identity = self._backend_identity_from_status(storage, pv_status, pvc_status) if pv_status else {}

        binding = self._binding_for_namespace(storage, namespace)
        binding.cluster = storage.cluster or default_cluster()
        binding.namespace = namespace
        binding.pvc_name = pvc_name or ''
        binding.pv_name = pv_name or ''
        binding.status = pvc_status.get('status') or 'Missing'
        binding.storage_class = (
            pvc_status.get('storage_class')
            or pv_status.get('storage_class')
            or storage.storage_class
            or ''
        )
        binding.backend_identity = json.dumps(identity, sort_keys=True, ensure_ascii=False) if identity else '{}'
        binding.last_checked_at = datetime.utcnow()
        return binding

    def _refresh_backend_match(self, storage):
        bindings = db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).order_by(
            StoragePvcBinding.namespace.asc()
        ).all()
        expected_namespaces = set(self._namespaces(storage))
        bindings = [binding for binding in bindings if binding.namespace in expected_namespaces]
        if len(bindings) <= 1:
            for binding in bindings:
                binding.backend_match_status = 'not_required'
                binding.backend_match_message = 'single namespace storage'
            return 'not_required'

        identities = []
        missing = []
        for binding in bindings:
            try:
                identity = json.loads(binding.backend_identity or '{}')
            except Exception:
                identity = {}
            if not self._identity_has_backend_fields(storage, identity):
                missing.append(binding.namespace)
            else:
                identities.append(identity)

        if missing or len(bindings) != len(expected_namespaces):
            message = 'backend identity missing: {}'.format(','.join(missing or sorted(expected_namespaces)))
            for binding in bindings:
                binding.backend_match_status = 'unknown'
                binding.backend_match_message = message
            if storage.status == 'shared_verified':
                storage.status = 'synced'
            return 'unknown'

        first_key = self._identity_key(identities[0])
        matched = all(self._identity_key(identity) == first_key for identity in identities[1:])
        if matched:
            for binding in bindings:
                binding.backend_match_status = 'matched'
                binding.backend_match_message = 'backend identity matched'
            all_bound = all(binding.status == 'Bound' for binding in bindings)
            if all_bound:
                storage.status = 'shared_verified'
            elif storage.status == 'backend_mismatch':
                storage.status = 'synced'
            return 'matched'

        for binding in bindings:
            binding.backend_match_status = 'mismatch'
            binding.backend_match_message = 'backend identity mismatch'
        storage.status = 'backend_mismatch'
        return 'mismatch'

    def _prune_stale_bindings(self, storage):
        namespaces = set(self._namespaces(storage))
        db.session.query(StoragePvcBinding).filter(
            StoragePvcBinding.storage_id == storage.id,
            ~StoragePvcBinding.namespace.in_(namespaces),
        ).delete(synchronize_session=False)

    def _sync_nfs_storage(self, storage):
        k8s_client = self._k8s_client(storage)
        resources = self._build_nfs_resources(storage)
        result = []
        self._prune_stale_bindings(storage)
        for resource in resources:
            pv_status = k8s_client.create_or_patch_pv(resource['pv'])
            pvc_status = k8s_client.create_or_patch_pvc(resource['namespace'], resource['pvc'])
            binding = self._sync_binding_record(
                storage,
                resource['namespace'],
                pvc_status=pvc_status,
                pv_status=pv_status,
                resource=resource,
            )
            result.append({
                "namespace": resource['namespace'],
                "pv": pv_status,
                "pvc": pvc_status,
                "binding_id": binding.id,
            })
        storage.status = 'synced'
        storage.pv_name = ','.join([resource['pv']['metadata']['name'] for resource in resources])
        storage.pvc_name = storage.pvc_name or storage.name
        self._refresh_backend_match(storage)
        db.session.commit()
        return result

    def _sync_minio_juicefs_storage(self, storage):
        k8s_client = self._k8s_client(storage)
        resources = self._build_minio_juicefs_resources(storage)
        result = []
        pv_names = []
        minio_config = self._apply_storage_class_secret_ref(storage, self._load_minio_juicefs_config(storage))
        self._prune_stale_bindings(storage)
        for resource in resources:
            pv_status = None
            if resource.get('pv'):
                pv_status = k8s_client.create_or_patch_pv(resource['pv'])
                if pv_status.get('name'):
                    pv_names.append(pv_status['name'])
            pvc_status = k8s_client.create_or_patch_pvc(resource['namespace'], resource['pvc'])
            if pvc_status.get('volume_name'):
                pv_names.append(pvc_status['volume_name'])
            if not pv_status and pvc_status.get('volume_name'):
                pv_status = k8s_client.get_pv(pvc_status['volume_name'])
            binding = self._sync_binding_record(
                storage,
                resource['namespace'],
                pvc_status=pvc_status,
                pv_status=pv_status,
                resource=resource,
            )
            result.append({
                "namespace": resource['namespace'],
                "pv": pv_status,
                "pvc": pvc_status,
                "binding_id": binding.id,
            })
        storage.status = 'synced'
        storage.pv_name = ','.join(sorted(set(pv_names)))
        storage.pvc_name = storage.pvc_name or storage.name
        storage.storage_class = minio_config.get('storage_class') or 'juicefs-sc'
        storage.config = json.dumps({STORAGE_TYPE_MINIO_JUICEFS: minio_config}, indent=4, ensure_ascii=False)
        self._refresh_backend_match(storage)
        db.session.commit()
        return result

    def _sync_storage(self, storage):
        if storage.storage_type == STORAGE_TYPE_NFS:
            return self._sync_nfs_storage(storage)
        if storage.storage_type == STORAGE_TYPE_MINIO_JUICEFS:
            return self._sync_minio_juicefs_storage(storage)
        raise Exception('unsupported storage_type')

    def _refresh_storage_bindings(self, storage):
        k8s_client = self._k8s_client(storage)
        result = []
        for resource in self._build_resources(storage):
            pv_status = None
            if resource.get('pv'):
                pv_status = k8s_client.get_pv(resource['pv']['metadata']['name'])
            pvc_status = k8s_client.get_pvc(resource['pvc']['metadata']['name'], resource['namespace'])
            if not pv_status and pvc_status.get('volume_name'):
                pv_status = k8s_client.get_pv(pvc_status['volume_name'])
            binding = self._sync_binding_record(
                storage,
                resource['namespace'],
                pvc_status=pvc_status,
                pv_status=pv_status,
                resource=resource,
            )
            result.append({
                "namespace": resource['namespace'],
                "pv": pv_status,
                "pvc": pvc_status,
                "binding": {
                    "id": binding.id,
                    "status": binding.status,
                    "backend_match_status": binding.backend_match_status,
                    "backend_match_message": binding.backend_match_message,
                },
            })
        self._refresh_backend_match(storage)
        binding_map = {
            binding.namespace: binding
            for binding in db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).all()
        }
        for item in result:
            binding = binding_map.get(item.get('namespace'))
            if binding:
                item['binding'] = {
                    "id": binding.id,
                    "status": binding.status,
                    "backend_match_status": binding.backend_match_status,
                    "backend_match_message": binding.backend_match_message,
                }
        db.session.commit()
        return result

    def _delete_storage_resources(self, storage):
        if storage.storage_type not in [STORAGE_TYPE_NFS, STORAGE_TYPE_MINIO_JUICEFS]:
            return []

        k8s_client = self._k8s_client(storage)
        resources = self._build_resources(storage)
        result = []
        pv_names = set(self._split_names(storage.pv_name))
        bindings = db.session.query(StoragePvcBinding).filter_by(storage_id=storage.id).all()

        if bindings:
            for binding in bindings:
                if binding.pv_name:
                    pv_names.add(binding.pv_name)
                pvc_status = k8s_client.get_pvc(name=binding.pvc_name, namespace=binding.namespace)
                if pvc_status.get('volume_name'):
                    pv_names.add(pvc_status['volume_name'])
                result.append({
                    "namespace": binding.namespace,
                    "pvc": k8s_client.delete_pvc(namespace=binding.namespace, name=binding.pvc_name),
                })
        else:
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
            result = self._refresh_storage_bindings(storage)
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

    @expose_api(description="验证多命名空间存储共享", url="/verify_shared/<storage_id>", methods=["POST"])
    def verify_shared(self, storage_id):
        try:
            self._assert_admin()
            storage = self._get_item(storage_id)
            result = self._verify_shared_storage(storage)
            return self.response(200, status=0, message='success', result=result)
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源文件管理页面", url="/files/<storage_id>", methods=["GET"])
    def files(self, storage_id):
        try:
            storage = self._get_item(storage_id)
            self._check_storage_permission(storage)
            self._assert_file_manager_allowed(storage)
            file_entry = self._file_manager_entry(storage)
            return render_template(
                'storage_files.html',
                storage=storage,
                namespace=file_entry.get('namespace'),
                file_entry=file_entry,
            )
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源文件列表", url="/file/list/<storage_id>", methods=["GET"])
    def file_list(self, storage_id):
        try:
            storage_path = self._normalize_storage_path(request.args.get('path') or '/')
            _, (k8s_client, pod) = self._file_operation_context(storage_id)
            pod_path = self._pod_path(storage_path)
            command = """
                set -e
                target={target}
                [ -d "$target" ] || exit 2
                for item in "$target"/* "$target"/.[!.]* "$target"/..?*; do
                    [ -e "$item" ] || continue
                    name="$(basename "$item")"
                    if [ -L "$item" ]; then type="symlink";
                    elif [ -d "$item" ]; then type="directory";
                    elif [ -f "$item" ]; then type="file";
                    else type="other"; fi
                    size="$(stat -c '%s' "$item" 2>/dev/null || echo 0)"
                    mtime="$(stat -c '%Y' "$item" 2>/dev/null || echo 0)"
                    printf '%s\\t%s\\t%s\\t%s\\n' "$type" "$size" "$mtime" "$name"
                done
            """.format(target=self._sh_quote(pod_path))
            output = k8s_client.exec_pod(pod['name'], pod['namespace'], command)
            items = []
            for line in (output or '').splitlines():
                parts = line.split('\t', 3)
                if len(parts) != 4:
                    continue
                item_type, size, mtime, name = parts
                item_path = posixpath.join(storage_path, name)
                if not item_path.startswith('/'):
                    item_path = '/' + item_path
                items.append({
                    "name": name,
                    "path": item_path,
                    "type": item_type,
                    "size": int(size) if str(size).isdigit() else 0,
                    "modified": self._format_mtime(mtime),
                })
            items.sort(key=lambda item: (item['type'] != 'directory', item['name'].lower()))
            return self.response(200, status=0, message='success', result={
                "path": storage_path,
                "namespace": pod['namespace'],
                "pod": pod['name'],
                "items": items,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源创建目录", url="/file/mkdir/<storage_id>", methods=["POST"])
    def file_mkdir(self, storage_id):
        try:
            req_json = request.get_json(silent=True) or {}
            storage_path = self._normalize_storage_path(req_json.get('path') or '/', allow_root=False)
            _, (k8s_client, pod) = self._file_operation_context(storage_id, write=True)
            pod_path = self._pod_path(storage_path)
            command = "mkdir -p -- {}".format(self._sh_quote(pod_path))
            k8s_client.exec_pod(pod['name'], pod['namespace'], command)
            return self.response(200, status=0, message='success', result={"path": storage_path})
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源上传文件", url="/file/upload/<storage_id>", methods=["POST"])
    def file_upload(self, storage_id):
        try:
            target_dir = self._normalize_storage_path(request.form.get('path') or '/')
            upload_file = request.files.get('file')
            if not upload_file:
                raise Exception('file is required')
            filename = os.path.basename(upload_file.filename or '').strip()
            if not filename or filename in ['.', '..'] or '/' in filename or '\\' in filename:
                raise Exception('invalid filename')
            if len(filename) > 255:
                raise Exception('filename is too long')
            max_size = int(conf.get('STORAGE_FILE_UPLOAD_MAX_SIZE', 200 * 1024 * 1024))
            data = upload_file.read(max_size + 1)
            if len(data) > max_size:
                raise Exception('file is too large')
            _, (k8s_client, pod) = self._file_operation_context(storage_id, write=True)
            pod_dir = self._pod_path(target_dir)
            check_command = '[ -d {target} ] && [ ! -L {target} ]'.format(target=self._sh_quote(pod_dir))
            k8s_client.exec_pod(pod['name'], pod['namespace'], check_command)
            target_path = posixpath.join(pod_dir, filename)
            k8s_client.upload_to_pod(pod['name'], pod['namespace'], data, target_path)
            return self.response(200, status=0, message='success', result={
                "path": posixpath.join(target_dir, filename),
                "size": len(data),
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源上传文件夹", url="/file/upload_folder/<storage_id>", methods=["POST"])
    def file_upload_folder(self, storage_id):
        try:
            target_dir = self._normalize_storage_path(request.form.get('path') or '/')
            files = request.files.getlist('files')
            relative_paths = request.form.getlist('relative_paths')

            if not files:
                raise Exception('files are required')

            max_size = int(conf.get('STORAGE_FILE_UPLOAD_MAX_SIZE', 200 * 1024 * 1024))
            _, (k8s_client, pod) = self._file_operation_context(storage_id, write=True)
            pod_dir = self._pod_path(target_dir)

            check_command = '[ -d {target} ] && [ ! -L {target} ]'.format(target=self._sh_quote(pod_dir))
            k8s_client.exec_pod(pod['name'], pod['namespace'], check_command)

            uploaded_count = 0
            failed_files = []

            for i, upload_file in enumerate(files):
                try:
                    if i < len(relative_paths):
                        relative_path = (relative_paths[i] or '').strip()
                    else:
                        relative_path = upload_file.filename or ''

                    if not relative_path:
                        raise Exception('invalid relative path: empty')

                    if relative_path.startswith('/'):
                        raise Exception('relative path must not start with /: {}'.format(relative_path))
                    if '..' in relative_path.split('/'):
                        raise Exception('path traversal in relative path: {}'.format(relative_path))
                    if '\\' in relative_path:
                        raise Exception('backslash in relative path: {}'.format(relative_path))
                    if '\x00' in relative_path:
                        raise Exception('null byte in relative path: {}'.format(relative_path))
                    if ':' in relative_path:
                        raise Exception('colon in relative path: {}'.format(relative_path))
                    if len(relative_path) > 1024:
                        raise Exception('relative path too long: {}'.format(relative_path))

                    filename = posixpath.basename(relative_path)
                    if not filename or filename in ['.', '..'] or '/' in filename or '\\' in filename:
                        raise Exception('invalid filename in path: {}'.format(relative_path))
                    if len(filename) > 255:
                        raise Exception('filename too long: {}'.format(filename))

                    target_file_pod = posixpath.join(pod_dir, relative_path)
                    target_real = posixpath.normpath(target_file_pod)
                    pod_dir_real = posixpath.normpath(pod_dir)
                    if target_real != pod_dir_real and not target_real.startswith(pod_dir_real + '/'):
                        raise Exception('path traversal detected: {}'.format(relative_path))

                    parent_dir = posixpath.dirname(target_file_pod)
                    mkdir_command = "mkdir -p -- {}".format(self._sh_quote(parent_dir))
                    k8s_client.exec_pod(pod['name'], pod['namespace'], mkdir_command)

                    data = upload_file.read(max_size + 1)
                    if len(data) > max_size:
                        raise Exception('file too large: {}'.format(relative_path))

                    k8s_client.upload_to_pod(pod['name'], pod['namespace'], data, target_file_pod)
                    uploaded_count += 1

                except Exception as e:
                    failed_files.append({
                        "path": upload_file.filename or '',
                        "error": str(e),
                    })

            return self.response(200, status=0, message='success', result={
                "current_path": target_dir,
                "uploaded_count": uploaded_count,
                "failed_count": len(failed_files),
                "failed_files": failed_files,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源下载文件", url="/file/download/<storage_id>", methods=["GET"])
    def file_download(self, storage_id):
        try:
            storage_path = self._normalize_storage_path(request.args.get('path') or '', allow_root=False)
            _, (k8s_client, pod) = self._file_operation_context(storage_id)
            pod_path = self._pod_path(storage_path)
            check_command = '[ -f {target} ] && [ ! -L {target} ]'.format(target=self._sh_quote(pod_path))
            k8s_client.exec_pod(pod['name'], pod['namespace'], check_command)
            data = k8s_client.download_from_pod(pod['name'], pod['namespace'], pod_path)
            filename = os.path.basename(storage_path)
            response = Response(data, content_type='application/octet-stream')
            response.headers['Content-Disposition'] = "attachment; filename*=UTF-8''{}".format(quote(filename))
            return response
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源删除文件", url="/file/delete/<storage_id>", methods=["DELETE"])
    def file_delete(self, storage_id):
        try:
            req_json = request.get_json(silent=True) or {}
            storage_path = self._normalize_storage_path(req_json.get('path') or '', allow_root=False)
            item_type = req_json.get('type', '')
            recursive = bool(req_json.get('recursive'))
            _, (k8s_client, pod) = self._file_operation_context(storage_id, write=True)
            pod_path = self._pod_path(storage_path)
            if item_type == 'directory' or recursive:
                command = "rm -rf -- {}".format(self._sh_quote(pod_path))
            else:
                command = "rm -f -- {}".format(self._sh_quote(pod_path))
            k8s_client.exec_pod(pod['name'], pod['namespace'], command)
            return self.response(200, status=0, message='success', result={
                "path": storage_path,
                "type": item_type or ('directory' if recursive else 'file'),
            })
        except Exception as e:
            return self.response_error(500, message=str(e))

    @expose_api(description="存储资源文本预览", url="/file/preview/<storage_id>", methods=["GET"])
    def file_preview(self, storage_id):
        try:
            storage_path = self._normalize_storage_path(request.args.get('path') or '', allow_root=False)
            limit = min(int(request.args.get('limit') or 20000), 100000)
            _, (k8s_client, pod) = self._file_operation_context(storage_id)
            pod_path = self._pod_path(storage_path)
            check_command = '[ -f {target} ] && [ ! -L {target} ]'.format(target=self._sh_quote(pod_path))
            k8s_client.exec_pod(pod['name'], pod['namespace'], check_command)
            command = "head -c {} {} | base64".format(limit + 1, self._sh_quote(pod_path))
            output = k8s_client.exec_pod(pod['name'], pod['namespace'], command)
            data = base64.b64decode(''.join((output or '').split()))
            truncated = len(data) > limit
            content = data[:limit].decode('utf-8', errors='replace')
            return self.response(200, status=0, message='success', result={
                "path": storage_path,
                "content": content,
                "truncated": truncated,
            })
        except Exception as e:
            return self.response_error(500, message=str(e))


class Storage_ModelView_Api(Storage_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Storage)
    route_base = '/storage_modelview/api'


appbuilder.add_api(Storage_ModelView_Api)
