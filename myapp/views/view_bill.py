import datetime

from flask import g
from flask_babel import lazy_gettext as _

from myapp import app, appbuilder, db
from myapp.models.model_bill import BillRecord, PodChargeRecord
from myapp.views.base import MyappFilter
from myapp.views.baseApi import MyappModelRestApi, expose
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface


conf = app.config


class PodCharge_Filter(MyappFilter):
    def apply(self, query, func):
        if g.user.is_admin():
            return query
        return query.filter(self.model.username == g.user.username)


class BillRecord_Filter(MyappFilter):
    def apply(self, query, func):
        if g.user.is_admin():
            return query
        return query.filter(self.model.username == g.user.username)


def _record_day_price(record, duration_hours):
    prices = conf.get("RESOURCE_BILL_PRICE", {}) or {}
    return round(float(duration_hours) * (
        float(record.cpu or 0) * float(prices.get("cpu", 0)) +
        float(record.memory or 0) * float(prices.get("memory", 0)) +
        float(record.gpu or 0) * float(prices.get("gpu", 0)) +
        float(record.vgpu or 0) * float(prices.get("vgpu", 0))
    ), 4)


class Pod_ModelView_Api(MyappModelRestApi):
    datamodel = SQLAInterface(PodChargeRecord)
    route_base = "/pod_modelview/api"
    primary_key = "id"
    base_permissions = ["can_list", "can_show"]
    base_filters = [["id", PodCharge_Filter, lambda: []]]
    base_order = ("start_time", "desc")
    order_columns = ["id", "start_time", "end_time", "duration_hours", "price"]
    page_size = 100

    label_title = _("按量计费")
    list_title = _("按量计费列表")
    show_title = _("Pod 计费详情")

    list_columns = [
        "username",
        "project",
        "cluster",
        "resource_group",
        "namespace",
        "pod_type",
        "node",
        "resource",
        "start_time",
        "end_time",
        "duration",
        "status",
        "price",
        "name",
    ]
    show_columns = [
        "project",
        "username",
        "name",
        "cluster",
        "namespace",
        "labels_html",
        "annotations_html",
        "status",
        "resource_group",
        "pod_type",
        "node",
        "resource",
        "start_time",
        "end_time",
        "changed_on",
        "events",
        "raw_pod",
    ]
    search_columns = ["username", "project", "cluster", "namespace", "pod_name", "status"]
    cols_width = {
        "username": {"type": "ellip2", "width": 100},
        "project": {"type": "ellip2", "width": 120},
        "cluster": {"type": "ellip2", "width": 100},
        "resource_group": {"type": "ellip2", "width": 100},
        "namespace": {"type": "ellip2", "width": 120},
        "pod_type": {"type": "ellip2", "width": 100},
        "node": {"type": "ellip2", "width": 140},
        "resource": {"type": "ellip2", "width": 180},
        "start_time": {"type": "ellip2", "width": 180},
        "end_time": {"type": "ellip2", "width": 180},
        "duration": {"type": "ellip2", "width": 90},
        "status": {"type": "ellip2", "width": 100},
        "price": {"type": "ellip2", "width": 90},
        "name": {"type": "ellip2", "width": 260},
    }


class Bill_ModelView_Api(MyappModelRestApi):
    datamodel = SQLAInterface(BillRecord)
    route_base = "/bill_modelview/api"
    primary_key = "bill_id"
    base_permissions = ["can_list", "can_show"]
    base_filters = [["id", BillRecord_Filter, lambda: []]]
    base_order = ("bill_date", "desc")
    order_columns = ["bill_date", "amount", "discount_price", "balance_pay"]
    page_size = 100

    label_title = _("账单支付")
    list_title = _("账单支付列表")
    show_title = _("账单详情")

    list_columns = [
        "bill_type",
        "bill_date",
        "bill_id",
        "amount",
        "status",
        "discount_price",
        "balance_pay",
        "username",
        "detail",
    ]
    show_columns = list_columns
    search_columns = ["bill_type", "bill_date", "bill_id", "status", "username"]
    cols_width = {
        "bill_type": {"type": "ellip2", "width": 100},
        "bill_date": {"type": "ellip2", "width": 120},
        "bill_id": {"type": "ellip2", "width": 220},
        "amount": {"type": "ellip2", "width": 100},
        "status": {"type": "ellip2", "width": 100},
        "discount_price": {"type": "ellip2", "width": 100},
        "balance_pay": {"type": "ellip2", "width": 100},
        "username": {"type": "ellip2", "width": 120},
        "detail": {"type": "ellip2", "width": 220},
    }

    @expose("/detail/<bill_id>", methods=["GET"])
    def bill_detail(self, bill_id, **kwargs):
        bill = db.session.query(BillRecord).filter(BillRecord.bill_id == bill_id).first()
        if not bill:
            return self.response_error(404, message="bill not found")
        if not g.user.is_admin() and bill.username != g.user.username:
            return self.response_error(403, message="no permission to show")

        day_start = datetime.datetime.combine(bill.bill_date, datetime.time.min)
        day_end = datetime.datetime.combine(bill.bill_date, datetime.time.max)
        records = (
            db.session.query(PodChargeRecord)
            .filter(PodChargeRecord.username == bill.username)
            .filter(PodChargeRecord.start_time <= day_end)
            .filter(PodChargeRecord.end_time >= day_start)
            .order_by(PodChargeRecord.start_time.asc())
            .all()
        )

        data = []
        for record in records:
            start_at = max(record.start_time, day_start)
            end_at = min(record.end_time or day_end, day_end)
            duration_hours = round(max((end_at - start_at).total_seconds(), 0) / 3600, 2)
            data.append({
                "name": record.pod_name,
                "namespace": record.namespace,
                "resource": record.resource,
                "day_start_time": start_at.strftime("%Y-%m-%d %H:%M:%S"),
                "day_end_time": end_at.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": "%sh" % duration_hours,
                "price": _record_day_price(record, duration_hours),
            })

        return self.response(200, status=0, message="success", result={
            "bill": {
                "bill_id": bill.bill_id,
                "bill_date": bill.bill_date.strftime("%Y-%m-%d"),
                "username": bill.username,
                "amount": bill.amount,
                "status": bill.status,
            },
            "data": data,
            "count": len(data),
        })


appbuilder.add_api(Pod_ModelView_Api)
appbuilder.add_api(Bill_ModelView_Api)
