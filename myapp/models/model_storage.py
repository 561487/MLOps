import json

from flask import Markup
from flask_appbuilder import Model
from flask_babel import lazy_gettext as _
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable


metadata = Model.metadata


class Storage(Model, AuditMixinNullable, MyappModelBase):
    __tablename__ = 'storage'

    id = Column(Integer, primary_key=True, comment='id主键')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False, comment='项目组id')
    project = relationship(
        "Project", foreign_keys=[project_id], lazy='selectin'
    )
    name = Column(String(100), nullable=False, unique=True, comment='英文名，用于生成k8s pvc名称')
    label = Column(String(200), nullable=True, default='', comment='显示名称')
    storage_type = Column(String(50), nullable=False, default='nfs', comment='存储类型，nfs/minio_juicefs')
    cluster = Column(String(100), nullable=False, default='', comment='所属集群')
    namespace = Column(String(200), nullable=False, default='', comment='PVC所在命名空间，多个用逗号分隔')
    mount_path = Column(String(500), nullable=True, default='/mnt/storage', comment='推荐容器挂载路径')
    capacity = Column(String(50), nullable=False, default='5Gi', comment='存储容量')
    access_modes = Column(String(200), nullable=False, default='ReadWriteMany', comment='访问模式，多个用逗号分隔')
    storage_class = Column(String(200), nullable=True, default='', comment='StorageClass，静态NFS PV默认留空')
    pv_name = Column(String(500), nullable=True, default='', comment='K8s PV名称，多个用逗号分隔')
    pvc_name = Column(String(200), nullable=True, default='', comment='K8s PVC名称')
    status = Column(String(50), nullable=False, default='draft', comment='状态')
    config = Column(Text(65536), nullable=True, default='{}', comment='非敏感配置JSON')
    remark = Column(Text, nullable=True, default='', comment='备注')
    expand = Column(Text(65536), nullable=True, default='{}', comment='扩展参数')

    label_columns = {
        **MyappModelBase.label_columns,
        "project": _("所属项目组"),
        "storage_type": _("存储类型"),
        "storage_type_display": _("存储类型"),
        "cluster": _("集群"),
        "namespace": _("命名空间"),
        "mount_path": _("推荐挂载路径"),
        "capacity": _("申请容量"),
        "access_modes": _("访问模式"),
        "storage_class": _("StorageClass"),
        "pv_name": _("PV名称"),
        "pvc_name": _("PVC名称"),
        "config": _("配置"),
        "config_html": _("配置"),
        "mount_expr": _("挂载表达式"),
        "files_html": _("文件"),
        "bindings_html": _("PVC绑定"),
        "remark": _("备注"),
    }

    def __repr__(self):
        return self.name

    @property
    def storage_type_display(self):
        if self.storage_type == 'minio_juicefs':
            return 's3/minio'
        return self.storage_type

    @property
    def config_html(self):
        try:
            config = json.dumps(json.loads(self.config or '{}'), indent=4, ensure_ascii=False)
        except Exception:
            config = self.config or ''
        return Markup('<pre><code>%s</code></pre>' % config)

    @property
    def mount_expr(self):
        pvc_name = self.pvc_name or self.name
        mount_path = self.mount_path or '/mnt/storage'
        return f'{pvc_name}(storage):{mount_path}'

    @property
    def files_html(self):
        return Markup(
            '<a class="btn btn-sm btn-primary" target="_blank" '
            'href="/storage_modelview/api/files/{id}">文件</a>'.format(id=self.id)
        )

    @property
    def bindings_html(self):
        rows = []
        for binding in sorted(self.bindings or [], key=lambda item: item.namespace or ''):
            rows.append(
                '<tr><td>{namespace}</td><td>{pvc}</td><td>{pv}</td><td>{status}</td><td>{match}</td></tr>'.format(
                    namespace=binding.namespace or '',
                    pvc=binding.pvc_name or '',
                    pv=binding.pv_name or '',
                    status=binding.status or '',
                    match=binding.backend_match_status or '',
                )
            )
        if not rows:
            return Markup('')
        return Markup(
            '<table class="table table-condensed">'
            '<thead><tr><th>Namespace</th><th>PVC</th><th>PV</th><th>Status</th><th>后端</th></tr></thead>'
            '<tbody>{}</tbody></table>'.format(''.join(rows))
        )


class StoragePvcBinding(Model, MyappModelBase):
    __tablename__ = 'storage_pvc_binding'

    id = Column(Integer, primary_key=True, comment='id主键')
    storage_id = Column(Integer, ForeignKey('storage.id'), nullable=False, comment='存储资源id')
    storage = relationship(
        "Storage",
        foreign_keys=[storage_id],
        back_populates="bindings",
        lazy='selectin',
    )
    cluster = Column(String(100), nullable=False, default='', comment='所属集群')
    namespace = Column(String(200), nullable=False, default='', comment='PVC所在命名空间')
    pvc_name = Column(String(200), nullable=False, default='', comment='K8s PVC名称')
    pv_name = Column(String(500), nullable=True, default='', comment='K8s PV名称')
    status = Column(String(50), nullable=False, default='Missing', comment='Kubernetes PVC状态')
    storage_class = Column(String(200), nullable=True, default='', comment='StorageClass')
    backend_identity = Column(Text(65536), nullable=True, default='{}', comment='后端标识JSON')
    backend_match_status = Column(String(50), nullable=False, default='unknown', comment='后端一致性状态')
    backend_match_message = Column(Text, nullable=True, default='', comment='后端一致性说明')
    last_checked_at = Column(DateTime, nullable=True, comment='最后检查时间')

    label_columns = {
        **MyappModelBase.label_columns,
        "storage": _("存储资源"),
        "cluster": _("集群"),
        "namespace": _("命名空间"),
        "pvc_name": _("PVC名称"),
        "pv_name": _("PV名称"),
        "status": _("状态"),
        "storage_class": _("StorageClass"),
        "backend_identity": _("后端标识"),
        "backend_match_status": _("后端一致性"),
        "backend_match_message": _("后端一致性说明"),
        "last_checked_at": _("最后检查时间"),
    }

    def __repr__(self):
        return '{}:{}'.format(self.namespace, self.pvc_name)


Storage.bindings = relationship(
    "StoragePvcBinding",
    back_populates="storage",
    cascade="all, delete-orphan",
    lazy='selectin',
)
