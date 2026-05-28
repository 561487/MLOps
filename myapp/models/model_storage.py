import json

from flask import Markup
from flask_appbuilder import Model
from flask_babel import lazy_gettext as _
from sqlalchemy import Column, ForeignKey, Integer, String, Text
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
    capacity = Column(String(50), nullable=False, default='500Gi', comment='存储容量')
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
        "remark": _("备注"),
    }

    def __repr__(self):
        return self.name

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
