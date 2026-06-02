import json
import traceback

from flask import g, jsonify, request
from flask_babel import lazy_gettext as _

from myapp import app, appbuilder, cache
from myapp.utils.py.py_k8s import K8s

from .baseFormApi import MyappFormRestApi, expose


conf = app.config
NODE_RESOURCE_CACHE_KEY = "node_modelview_resource_data"
WRITABLE_NODE_LABELS = {
    "org",
    "share",
    "cpu",
    "gpu",
    "vgpu",
    "gpu-type",
    "mps",
    "service",
    "notebook",
    "train",
    "rdma",
}
COMPUTE_TYPE_LABELS = ["notebook", "train", "service"]


def clear_node_resource_cache():
    cache.delete(NODE_RESOURCE_CACHE_KEY)


def _format_time(value):
    if not value:
        return ""
    try:
        return (value.replace(tzinfo=None)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def _node_device(labels):
    return labels.get("gpu-type", "")


def _key_labels(labels):
    keys = [
        "org",
        "share",
        "cpu",
        "gpu",
        "vgpu",
        "gpu-type",
        "mps",
        "notebook",
        "train",
        "service",
        "rdma",
        "kubernetes.io/arch",
        "kubernetes.io/os",
    ]
    return ",".join(["%s=%s" % (key, labels.get(key)) for key in keys if key in labels])


def _share_mode(labels):
    return "true" if labels.get("mps", "") == "true" or labels.get("share", "") == "true" else "false"


def _rdma_enabled(labels):
    return "true" if labels.get("rdma", "") == "true" else "false"


def _compute_type(labels):
    return ",".join([key for key in COMPUTE_TYPE_LABELS if labels.get(key) == "true"])


def node_resource(force_refresh=False):
    if force_refresh:
        clear_node_resource_cache()

    all_node_resource = cache.get(NODE_RESOURCE_CACHE_KEY) or []
    if all_node_resource:
        return all_node_resource

    clusters = conf.get("CLUSTERS", {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        try:
            k8s_client = K8s(cluster.get("KUBECONFIG", ""))
            all_node = k8s_client.get_node(cache=True)

            for node in all_node:
                labels = node.get("labels", {}) or {}
                node_info = node.get("node_info", {}) or {}

                all_node_resource.append({
                    "id": "%s:%s" % (cluster_name, node.get("name", "")),
                    "cluster": cluster_name,
                    "resource_group": labels.get("org", "unknown"),
                    "name": node.get("name", ""),
                    "ip": node.get("hostip", ""),
                    "status": node.get("status", "Unknown"),
                    "device": _node_device(labels),
                    "share_mode": _share_mode(labels),
                    "cpu": labels.get("cpu", ""),
                    "gpu": labels.get("gpu", ""),
                    "vgpu": labels.get("vgpu", ""),
                    "gpu_type": labels.get("gpu-type", ""),
                    "mps": labels.get("mps", ""),
                    "compute_type": _compute_type(labels),
                    "gpu_exclusive": node.get("gpu", 0),
                    "gpu_shared": node.get("gpu_shared", 0),
                    "rdma": _rdma_enabled(labels),
                    "architecture": node_info.get("architecture", labels.get("kubernetes.io/arch", "")),
                    "operating_system": node_info.get("operating_system", labels.get("kubernetes.io/os", "")),
                    "kernel_version": node_info.get("kernel_version", ""),
                    "joined_at": _format_time(node.get("create_time")),
                    "labels": labels,
                    "key_labels": _key_labels(labels),
                })
        except Exception:
            traceback.print_exc()

    cache.set(NODE_RESOURCE_CACHE_KEY, all_node_resource, timeout=10)
    return all_node_resource


class Node_ModelView_Api(MyappFormRestApi):
    route_base = "/node_modelview/api"
    primary_key = "id"
    order_columns = [
        "cluster",
        "resource_group",
        "name",
        "ip",
        "status",
        "gpu_exclusive",
        "gpu_shared",
        "joined_at",
    ]
    cols_width = {
        "cluster": {"type": "ellip2", "width": 90},
        "resource_group": {"type": "ellip2", "width": 90},
        "name": {"type": "ellip2", "width": 180},
        "ip": {"type": "ellip2", "width": 130},
        "status": {"type": "ellip2", "width": 90},
        "device": {"type": "ellip2", "width": 110},
        "share_mode": {"type": "ellip2", "width": 90},
        "gpu_exclusive": {"type": "ellip2", "width": 100},
        "gpu_shared": {"type": "ellip2", "width": 100},
        "rdma": {"type": "ellip2", "width": 80},
        "architecture": {"type": "ellip2", "width": 100},
        "operating_system": {"type": "ellip2", "width": 110},
        "kernel_version": {"type": "ellip2", "width": 160},
        "joined_at": {"type": "ellip2", "width": 180},
        "key_labels": {"type": "ellip2", "width": 260},
    }
    label_columns = {
        "cluster": _("集群"),
        "resource_group": _("资源组"),
        "name": _("节点名"),
        "ip": _("IP"),
        "status": _("状态"),
        "device": _("设备类型"),
        "share_mode": _("共享模式"),
        "cpu": _("CPU调度"),
        "gpu": _("GPU调度"),
        "vgpu": _("VGPU调度"),
        "gpu_type": _("GPU卡型"),
        "mps": _("GPU共享"),
        "compute_type": _("计算类型"),
        "gpu_exclusive": _("独占AI卡"),
        "gpu_shared": _("共享AI卡"),
        "rdma": _("RDMA"),
        "architecture": _("系统架构"),
        "operating_system": _("操作系统"),
        "kernel_version": _("系统内核"),
        "joined_at": _("加入时间"),
        "key_labels": _("关键标签"),
        "labels": _("Labels"),
        "id": _("ID"),
    }
    list_columns = [
        "cluster",
        "resource_group",
        "name",
        "ip",
        "status",
        "device",
        "share_mode",
        "gpu_exclusive",
        "gpu_shared",
        "rdma",
        "architecture",
        "operating_system",
        "kernel_version",
        "joined_at",
        "key_labels",
    ]
    show_columns = list_columns + ["labels"]
    query_columns = [
        "name",
        "device",
        "resource_group",
    ]
    edit_columns = ["id"] + [
        "cluster",
        "name",
        "resource_group",
        "cpu",
        "gpu",
        "vgpu",
        "rdma",
        "gpu_type",
        "mps",
        "compute_type",
    ]
    label_title = _("机器资源")
    list_title = _("机器资源列表")
    edit_title = _("修改机器资源标签")
    show_title = _("机器资源详情")
    page_size = 1000
    base_permissions = ["can_list", "can_show", "can_edit", "can_delete"]

    def add_more_info(self, info, **kwargs):
        info["filters"] = self._query_filters()
        info["edit_columns"] = self._edit_form_columns(request.args.get("id", ""))
        info["edit_fieldsets"] = [{
            "group": _("标签配置"),
            "expanded": True,
            "fields": self.edit_columns,
        }]

    def _query_filters(self):
        return {
            "name": self._filter_info("name", "input", operator="ct"),
            "device": self._filter_info("device", "input", operator="ct"),
            "resource_group": self._filter_info("resource_group", "input", operator="ct"),
        }

    def _filter_info(self, name, ui_type, values=None, operator="ct"):
        return {
            "name": name,
            "label": self.label_columns.get(name, name),
            "ui-type": ui_type,
            "values": values or [],
            "default": "",
            "filter": [{"operator": operator}],
        }

    def _edit_form_columns(self, edit_id=""):
        clusters = [{"id": name, "value": name} for name in conf.get("CLUSTERS", {})]
        bool_values = [{"id": "true", "value": "true"}, {"id": "false", "value": "false"}]
        compute_values = [
            {"id": "notebook", "value": "notebook"},
            {"id": "train", "value": "train"},
            {"id": "service", "value": "service"},
        ]
        return [
            self._field_info("id", "input", default=edit_id, required=True, disable=True),
            self._field_info("cluster", "select", clusters, required=True, disable=True),
            self._field_info("name", "input", required=True, disable=True),
            self._field_info("resource_group", "input", default="public", required=True),
            self._field_info("cpu", "select", bool_values, default="true"),
            self._field_info("gpu", "select", bool_values, default="false"),
            self._field_info("vgpu", "select", bool_values, default="false"),
            self._field_info("rdma", "select", bool_values, default="false"),
            self._field_info("gpu_type", "input"),
            self._field_info("mps", "select", bool_values, default="false"),
            self._field_info("compute_type", "select2", compute_values, default="notebook,train,service"),
        ]

    def _field_info(self, name, ui_type, values=None, default="", required=False, disable=False):
        return {
            "name": name,
            "label": self.label_columns.get(name, name),
            "description": self.description_columns.get(name, ""),
            "default": default,
            "type": "String",
            "ui-type": ui_type,
            "values": values or [],
            "validators": [{"type": "DataRequired"}] if required else [],
            "required": required,
            "disable": disable,
        }

    def _json_response(self, message="success", status=0, result=None, http_status=200):
        response = jsonify({
            "message": message,
            "status": status,
            "result": result if result is not None else {},
        })
        response.status_code = http_status
        return response

    def _require_admin(self):
        if not getattr(g, "user", None) or not g.user.is_admin():
            return self._json_response("no permission", 1, http_status=403)
        return None

    def _cluster_config(self, cluster):
        return conf.get("CLUSTERS", {}).get(cluster)

    def _split_pk(self, pk):
        if ":" not in pk:
            return "", pk
        return pk.split(":", 1)

    def _get_node(self, cluster, node_name, force_refresh=False):
        for node in node_resource(force_refresh=force_refresh):
            if node.get("cluster") == cluster and node.get("name") == node_name:
                return node
        return None

    def _normalize_bool(self, value, default="false"):
        if value in [True, "true", "True", "1", 1, "yes"]:
            return "true"
        if value in [False, "false", "False", "0", 0, "no"]:
            return "false"
        return default

    def _request_json(self):
        data = request.get_json(silent=True) or {}
        return {key: value.strip() if isinstance(value, str) else value for key, value in data.items()}

    def _labels_from_request(self, data, clear_unselected_compute=False):
        labels = {
            "org": data.get("resource_group", ""),
            "cpu": self._normalize_bool(data.get("cpu"), "true"),
            "gpu": self._normalize_bool(data.get("gpu")),
            "vgpu": self._normalize_bool(data.get("vgpu")),
            "rdma": self._normalize_bool(data.get("rdma")),
            "mps": self._normalize_bool(data.get("mps")),
        }
        labels["share"] = labels["mps"]

        gpu_type = data.get("gpu_type", "")
        labels["gpu-type"] = gpu_type if gpu_type else None

        compute_type = data.get("compute_type", "")
        if isinstance(compute_type, str):
            compute_types = [item.strip() for item in compute_type.split(",") if item.strip()]
        else:
            compute_types = [str(item) for item in compute_type]
        for label in COMPUTE_TYPE_LABELS:
            if label in compute_types:
                labels[label] = "true"
            elif clear_unselected_compute:
                labels[label] = None

        return self._validate_labels(labels, require_org=True)

    def _validate_labels(self, labels, require_org=False):
        unknown_keys = [key for key in labels if key not in WRITABLE_NODE_LABELS]
        if unknown_keys:
            raise ValueError("invalid label keys: %s" % ",".join(unknown_keys))
        if require_org and not labels.get("org"):
            raise ValueError("resource_group is required")
        for key, value in labels.items():
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError("invalid label value for %s" % key)
            if len(value) > 63:
                raise ValueError("label value too long for %s" % key)
        return labels

    def _patch_labels(self, cluster, node_name, labels):
        labels = self._validate_labels(labels)
        cluster_config = self._cluster_config(cluster)
        if not cluster_config:
            raise ValueError("cluster not found")
        k8s_client = K8s(cluster_config.get("KUBECONFIG", ""))
        if not k8s_client.get_node(name=node_name):
            raise ValueError("node not found")
        if not k8s_client.patch_node_labels(node_name, labels):
            raise RuntimeError("patch node labels failed")
        clear_node_resource_cache()
        return self._get_node(cluster, node_name, force_refresh=True) or {
            "id": "%s:%s" % (cluster, node_name),
            "cluster": cluster,
            "name": node_name,
        }

    def _request_filters(self):
        args = request.get_json(silent=True) or {}
        try:
            args.update(json.loads(request.args.get("form_data", "{}")))
        except Exception:
            args.update({})
        args.update(request.args)

        filter_values = {}
        for item in args.get("filters", []) or []:
            key = item.get("col")
            if key in self.query_columns and item.get("value") not in [None, ""]:
                filter_values[key] = item.get("value")

        for key in self.query_columns:
            if args.get(key) not in [None, ""]:
                filter_values[key] = args.get(key)
        return filter_values

    def _contains(self, data, key, value):
        return str(value).lower() in str(data.get(key, "")).lower()

    def query_list(self, order_column, order_direction, page_index, page_size, filters=None, **kargs):
        filter_values = self._request_filters()
        force_refresh = request.args.get("refresh", "").lower() in ["1", "true", "yes"]
        lst = node_resource(force_refresh=force_refresh)

        for key, value in filter_values.items():
            lst = [node for node in lst if self._contains(node, key, value)]

        if order_column and order_column in self.order_columns and lst:
            reverse = order_direction != "asc"
            lst = sorted(lst, key=lambda node: node.get(order_column, ""), reverse=reverse)
        return len(lst), lst

    @expose("/<path:pk>", methods=["GET"])
    def get(self, pk):
        cluster, node_name = self._split_pk(pk)
        node = self._get_node(cluster, node_name)
        if not node:
            return self._json_response("node not found", 1, http_status=404)
        return self._json_response(result=node)

    @expose("/<path:pk>", methods=["PUT"])
    def put(self, pk):
        deny = self._require_admin()
        if deny:
            return deny
        try:
            data = self._request_json()
            cluster, node_name = self._split_pk(pk)
            cluster = cluster or data.get("cluster", "")
            node_name = node_name or data.get("name", "")
            if not cluster or not node_name:
                raise ValueError("cluster and name are required")
            labels = self._labels_from_request(data, clear_unselected_compute=True)
            node = self._patch_labels(cluster, node_name, labels)
            return self._json_response(result=node)
        except Exception as e:
            traceback.print_exc()
            return self._json_response(str(e), 1, http_status=400)

    @expose("/<path:pk>", methods=["DELETE"])
    def delete(self, pk):
        deny = self._require_admin()
        if deny:
            return deny
        try:
            cluster, node_name = self._split_pk(pk)
            labels = {key: None for key in WRITABLE_NODE_LABELS}
            node = self._patch_labels(cluster, node_name, labels)
            return self._json_response(result=node)
        except Exception as e:
            traceback.print_exc()
            return self._json_response(str(e), 1, http_status=400)

    @expose("/label", methods=["POST"])
    def label(self):
        deny = self._require_admin()
        if deny:
            return deny
        try:
            data = self._request_json()
            cluster = data.get("cluster", "")
            node_name = data.get("name", "")
            labels = data.get("labels", {}) or {}
            if not isinstance(labels, dict):
                raise ValueError("labels must be object")
            node = self._patch_labels(cluster, node_name, self._validate_labels(labels))
            return self._json_response(result=node)
        except Exception as e:
            traceback.print_exc()
            return self._json_response(str(e), 1, http_status=400)

    @expose("/resource_group", methods=["POST"])
    def resource_group(self):
        deny = self._require_admin()
        if deny:
            return deny
        try:
            data = self._request_json()
            node = self._patch_labels(data.get("cluster", ""), data.get("name", ""), {
                "org": data.get("resource_group", ""),
            })
            return self._json_response(result=node)
        except Exception as e:
            traceback.print_exc()
            return self._json_response(str(e), 1, http_status=400)

    @expose("/share", methods=["POST"])
    def share(self):
        deny = self._require_admin()
        if deny:
            return deny
        try:
            data = self._request_json()
            share = self._normalize_bool(data.get("share"), "false")
            node = self._patch_labels(data.get("cluster", ""), data.get("name", ""), {
                "mps": share,
                "share": share,
            })
            return self._json_response(result=node)
        except Exception as e:
            traceback.print_exc()
            return self._json_response(str(e), 1, http_status=400)

    @expose("/data", methods=["GET"])
    def data(self):
        count, data = self.query_list(
            request.args.get("order_column", ""),
            request.args.get("order_direction", "asc"),
            request.args.get("page_index", 0),
            request.args.get("page_size", self.page_size),
        )
        return jsonify({
            "message": "success",
            "status": 0,
            "result": {
                "data": data,
                "count": count,
            },
        })

    @expose("/event/<cluster>/<node_name>", methods=["GET"])
    def event(self, cluster, node_name):
        cluster_config = conf.get("CLUSTERS", {}).get(cluster)
        if not cluster_config:
            return jsonify({
                "message": "cluster not found",
                "status": 1,
                "result": [],
            })
        try:
            k8s_client = K8s(cluster_config.get("KUBECONFIG", ""))
            events = k8s_client.get_node_event(node_name)
            return jsonify({
                "message": "success",
                "status": 0,
                "result": events,
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "message": str(e),
                "status": 1,
                "result": [],
            })


appbuilder.add_api(Node_ModelView_Api)
