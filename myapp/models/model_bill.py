import json

from flask import Markup
from flask_appbuilder import Model
from flask_babel import lazy_gettext as _
from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint

from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable


metadata = Model.metadata


class PodChargeRecord(Model, AuditMixinNullable, MyappModelBase):
    __tablename__ = "pod_charge_record"
    __table_args__ = (
        UniqueConstraint("cluster", "namespace", "pod_name", "start_time", name="uq_pod_charge_identity"),
    )

    id = Column(Integer, primary_key=True, comment="id主键")
    username = Column(String(100), nullable=False, default="", index=True, comment="用户名")
    project = Column(String(200), nullable=True, default="", comment="项目组")
    cluster = Column(String(100), nullable=False, default="", index=True, comment="集群")
    resource_group = Column(String(100), nullable=True, default="", comment="资源组")
    namespace = Column(String(200), nullable=False, default="", index=True, comment="命名空间")
    pod_type = Column(String(100), nullable=True, default="", comment="Pod类型")
    node = Column(String(200), nullable=True, default="", comment="机器")
    pod_name = Column(String(300), nullable=False, default="", index=True, comment="Pod名称")
    cpu = Column(Float, nullable=False, default=0, comment="CPU核数")
    memory = Column(Float, nullable=False, default=0, comment="内存GB")
    gpu = Column(Float, nullable=False, default=0, comment="GPU卡数")
    vgpu = Column(Float, nullable=False, default=0, comment="VGPU卡数")
    start_time = Column(DateTime, nullable=False, index=True, comment="开始时间")
    end_time = Column(DateTime, nullable=True, index=True, comment="截止时间")
    duration_hours = Column(Float, nullable=False, default=0, comment="耗时小时")
    status = Column(String(50), nullable=False, default="", comment="状态")
    price = Column(Float, nullable=False, default=0, comment="价格")
    labels = Column(Text(65536), nullable=True, default="{}", comment="Labels")
    annotations = Column(Text(65536), nullable=True, default="{}", comment="Annotations")
    events = Column(Text(65536), nullable=True, default="[]", comment="Events")
    raw_pod = Column(Text(16777216), nullable=True, default="{}", comment="原始Pod信息")

    label_columns = {
        **MyappModelBase.label_columns,
        "username": _("用户"),
        "project": _("项目组"),
        "cluster": _("集群"),
        "resource_group": _("资源组"),
        "namespace": _("命名空间"),
        "pod_type": _("Pod类型"),
        "node": _("机器"),
        "pod_name": _("名称"),
        "name": _("名称"),
        "resource": _("资源"),
        "start_time": _("开始时间"),
        "end_time": _("截止时间"),
        "duration": _("耗时"),
        "duration_hours": _("耗时"),
        "status": _("状态"),
        "price": _("价格"),
        "labels": _("标签"),
        "labels_html": _("标签"),
        "annotations": _("机器选择"),
        "annotations_html": _("机器选择"),
        "events": _("Events"),
        "raw_pod": _("原始Pod信息"),
    }

    def __repr__(self):
        return self.pod_name

    @property
    def name(self):
        return self.pod_name

    @property
    def resource(self):
        parts = [
            "cpu:%s" % self._fmt(self.cpu),
            "memory:%sG" % self._fmt(self.memory),
        ]
        if self.gpu:
            parts.append("gpu:%s" % self._fmt(self.gpu))
        if self.vgpu:
            parts.append("vgpu:%s" % self._fmt(self.vgpu))
        return ", ".join(parts)

    @property
    def duration(self):
        return "%sh" % self._fmt(self.duration_hours)

    @property
    def labels_html(self):
        return self._json_html(self.labels)

    @property
    def annotations_html(self):
        return self._json_html(self.annotations)

    @staticmethod
    def _fmt(value):
        try:
            return ("%0.2f" % float(value)).rstrip("0").rstrip(".")
        except Exception:
            return value

    @staticmethod
    def _json_html(value):
        try:
            data = json.dumps(json.loads(value or "{}"), indent=4, ensure_ascii=False)
        except Exception:
            data = value or ""
        return Markup("<pre><code>%s</code></pre>" % data)


class BillRecord(Model, AuditMixinNullable, MyappModelBase):
    __tablename__ = "bill_record"

    id = Column(Integer, primary_key=True, comment="id主键")
    bill_type = Column(String(50), nullable=False, default="pod", comment="账单类型")
    bill_date = Column(Date, nullable=False, index=True, comment="账单日期")
    bill_id = Column(String(200), nullable=False, unique=True, index=True, comment="账单ID")
    amount = Column(Float, nullable=False, default=0, comment="金额")
    status = Column(String(50), nullable=False, default="unpaid", comment="状态")
    discount_price = Column(Float, nullable=False, default=0, comment="优惠价格")
    balance_pay = Column(Float, nullable=False, default=0, comment="余额支付")
    username = Column(String(100), nullable=False, default="", index=True, comment="用户名")

    label_columns = {
        **MyappModelBase.label_columns,
        "bill_type": _("账单类型"),
        "bill_date": _("日期"),
        "bill_id": _("账单 ID"),
        "amount": _("金额"),
        "status": _("状态"),
        "discount_price": _("优惠价格"),
        "balance_pay": _("余额支付"),
        "username": _("用户名"),
        "detail": _("详情"),
    }

    def __repr__(self):
        return self.bill_id

    @property
    def detail(self):
        return "/bill_modelview/api/detail/%s" % self.bill_id
