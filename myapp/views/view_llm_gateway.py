import datetime
import hmac
import json
import secrets
import time
import uuid

import requests
from flask import Response, g, jsonify, request
from flask_appbuilder.baseviews import expose_api
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, DateTimePickerWidget, Select2Widget
from flask_babel import lazy_gettext as _
from sqlalchemy import func
from wtforms import DateTimeField, SelectField, StringField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, Length, Optional, Regexp

from myapp import app, appbuilder, db, security_manager
from myapp.forms import MySelect2Widget
from myapp.models.model_llm_gateway import LlmGateway, LlmGatewayLog
from myapp.models.model_serving import InferenceService
from myapp.views.base import MyappFilter
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from myapp.views.view_team import Project_Join_Filter, filter_join_org_project
from .baseApi import MyappModelRestApi


conf = app.config


def generate_api_key():
    return 'sk-' + secrets.token_urlsafe(32)


def api_key_prefix(api_key):
    return api_key[:10]


def default_expire_at():
    days = int(conf.get('LLM_GATEWAY_DEFAULT_EXPIRE_DAYS', 365))
    return datetime.datetime.now() + datetime.timedelta(days=days)


def filter_join_inferenceservice():
    query = db.session.query(InferenceService).order_by(InferenceService.id.desc())
    if g and hasattr(g, 'user') and g.user and not g.user.is_admin():
        join_projects_id = security_manager.get_join_projects_id(db.session)
        query = query.filter(InferenceService.project_id.in_(join_projects_id))
    return query.all()


class LlmGateway_Filter(MyappFilter):
    def apply(self, query, func):
        if g and hasattr(g, 'user') and g.user and g.user.is_admin():
            return query
        join_projects_id = security_manager.get_join_projects_id(db.session)
        return query.filter(self.model.project_id.in_(join_projects_id))


def json_response(payload, status_code=200):
    response = jsonify(payload)
    response.status_code = status_code
    return response


def extract_bearer_token():
    authorization = request.headers.get('Authorization', '').strip()
    if not authorization.lower().startswith('bearer '):
        return ''
    return authorization.split(' ', 1)[1].strip()


def get_gateway(api_key, model_name):
    if not api_key:
        return None, 'missing api key'
    if not model_name:
        return None, 'missing model'

    gateway = db.session.query(LlmGateway).filter_by(api_key=api_key).first()
    if not gateway:
        return None, 'invalid api key'
    if not hmac.compare_digest(gateway.api_key, api_key):
        return None, 'invalid api key'
    if gateway.status != 'enabled':
        return None, 'gateway disabled'
    if gateway.expire_at and gateway.expire_at < datetime.datetime.now():
        return None, 'api key expired'
    if gateway.model_name != model_name:
        return None, 'model not found for this api key'
    return gateway, ''


def normalize_token_quota(value):
    value = str(value or 'no-limit').strip()
    if value == 'no-limit':
        return value
    if not value.isdigit() or int(value) <= 10000:
        raise ValueError('token额度必须为 no-limit 或大于10000的整数')
    return value


def quota_exceeded(gateway):
    token_quota = normalize_token_quota(gateway.token_quota)
    if token_quota == 'no-limit':
        return False
    used_tokens = (
        db.session.query(func.coalesce(func.sum(LlmGatewayLog.total_tokens), 0))
        .filter(LlmGatewayLog.gateway_id == gateway.id)
        .scalar()
    )
    return int(used_tokens or 0) >= int(token_quota)


def normalize_base_url(value):
    value = (value or '').strip()
    if not value:
        return ''
    if value.startswith('http://') or value.startswith('https://'):
        return value.rstrip('/')
    return 'http://' + value.rstrip('/')


def service_base_url(service):
    service_host = normalize_base_url(service.host)
    if service_host:
        return service_host

    port = (service.ports or '80').split(',')[0].strip() or '80'
    host = f'{service.name}.{service.namespace}.svc.cluster.local'
    return f'http://{host}:{port}'


def join_openai_path(base_url, endpoint):
    base_url = base_url.rstrip('/')
    if base_url.endswith('/v1'):
        return f'{base_url}/{endpoint}'
    if base_url.endswith(f'/v1/{endpoint}'):
        return base_url
    return f'{base_url}/v1/{endpoint}'


def backend_chat_url(gateway):
    if not gateway.service:
        raise Exception('gateway has no backend service')
    return join_openai_path(service_base_url(gateway.service), 'chat/completions')


def record_gateway_log(gateway, body, status_code, success, latency_ms, error_message=''):
    usage = body.get('usage', {}) if isinstance(body, dict) else {}
    log = LlmGatewayLog(
        gateway_id=gateway.id if gateway else None,
        request_id=request.headers.get('X-Request-Id', str(uuid.uuid4())),
        model_name=body.get('model', '') if isinstance(body, dict) else '',
        api_key_prefix=api_key_prefix(gateway.api_key) if gateway else '',
        client_ip=request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip(),
        path=request.path,
        method=request.method,
        status_code=status_code,
        success=success,
        latency_ms=latency_ms,
        prompt_tokens=usage.get('prompt_tokens', 0),
        completion_tokens=usage.get('completion_tokens', 0),
        total_tokens=usage.get('total_tokens', 0),
        error_message=(error_message or '')[:2000],
    )
    db.session.add(log)
    db.session.commit()


def proxy_chat_completions(gateway, body):
    if body.get('stream'):
        raise ValueError('stream is not supported in phase one')

    url = backend_chat_url(gateway)
    headers = {'Content-Type': 'application/json'}
    timeout = int(conf.get('LLM_GATEWAY_TIMEOUT', 60))
    return requests.post(url, headers=headers, json=body, timeout=timeout, verify=False)


@app.route('/llm/api/<api_version>/v1/chat/completions', methods=['POST'])
def llm_gateway_chat_completions(api_version):
    start_time = time.time()
    body = request.get_json(silent=True) or {}
    gateway = None
    try:
        expected_version = str(conf.get('LLM_GATEWAY_API_VERSION', '1'))
        if str(api_version) != expected_version:
            return json_response({'error': {'message': 'invalid api version'}}, 404)

        api_key = extract_bearer_token()
        gateway, error_message = get_gateway(api_key, body.get('model', ''))
        if error_message:
            return json_response({'error': {'message': error_message}}, 401)
        if quota_exceeded(gateway):
            latency_ms = int((time.time() - start_time) * 1000)
            record_gateway_log(gateway, body, 429, False, latency_ms, 'token quota exceeded')
            return json_response({'error': {'message': 'token quota exceeded'}}, 429)

        response = proxy_chat_completions(gateway, body)
        latency_ms = int((time.time() - start_time) * 1000)
        success = 200 <= response.status_code < 300
        response_body = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else {}
        record_gateway_log(
            gateway=gateway,
            body=response_body if response_body else body,
            status_code=response.status_code,
            success=success,
            latency_ms=latency_ms,
            error_message='' if success else response.text[:2000],
        )
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type', 'application/json'))
    except ValueError as err:
        latency_ms = int((time.time() - start_time) * 1000)
        if gateway:
            record_gateway_log(gateway, body, 400, False, latency_ms, str(err))
        return json_response({'error': {'message': str(err)}}, 400)
    except Exception as err:
        latency_ms = int((time.time() - start_time) * 1000)
        if gateway:
            record_gateway_log(gateway, body, 500, False, latency_ms, str(err))
        return json_response({'error': {'message': 'backend request failed', 'detail': str(err)}}, 500)


class LlmGateway_ModelView_Base():
    datamodel = SQLAInterface(LlmGateway)
    label_title = _('服务网关')
    base_order = ('id', 'desc')
    order_columns = ['id']
    page_size = 100

    list_columns = ['name', 'label', 'model_name', 'project', 'service', 'gateway_url', 'api_key', 'token_quota', 'call_count', 'last_call_at', 'status', 'operate_html']
    show_columns = ['name', 'label', 'model_name', 'project', 'service', 'gateway_url', 'api_key', 'token_quota', 'expire_at', 'status']
    add_columns = ['name', 'label', 'model_name', 'project', 'service', 'api_key', 'token_quota', 'expire_at', 'status']
    edit_columns = add_columns
    search_columns = ['name', 'label', 'model_name', 'status']
    base_filters = [["id", LlmGateway_Filter, lambda: []]]
    description_columns = {
        "expire_at": _("过期时间格式：YYYY-MM-DD HH:mm:ss，例如 2026-12-31 23:59:59；新增时留空默认一年后过期。"),
    }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields

    add_form_extra_fields = {
        "name": StringField(
            _('名称'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Length(1, 100), Regexp('^[a-z0-9]([-a-z0-9_]*[a-z0-9])?$')]
        ),
        "project": QuerySelectField(
            _('所属项目组'),
            query_factory=filter_join_org_project,
            widget=MySelect2Widget(new_web=False),
            validators=[DataRequired()]
        ),
        "service": QuerySelectField(
            _('后端服务'),
            query_factory=filter_join_inferenceservice,
            widget=MySelect2Widget(new_web=False),
            validators=[DataRequired()]
        ),
        "api_key": StringField(
            _('API Key'),
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Length(10, 128)]
        ),
        "token_quota": StringField(
            _('token额度'),
            default='no-limit',
            widget=BS3TextFieldWidget(),
            description=_('默认 no-limit；如填写数字，必须大于10000'),
            validators=[DataRequired()]
        ),
        "expire_at": DateTimeField(
            _('过期时间'),
            format='%Y-%m-%d %H:%M:%S',
            widget=DateTimePickerWidget(),
            description=_('格式：YYYY-MM-DD HH:mm:ss，例如 2026-12-31 23:59:59；新增时留空默认一年后过期。'),
            validators=[Optional()]
        ),
        "status": SelectField(
            _('状态'),
            widget=Select2Widget(),
            default='enabled',
            choices=[['enabled', _('启用')], ['disabled', _('禁用')]]
        ),
    }
    edit_form_extra_fields = add_form_extra_fields

    cols_width = {
        "name": {"type": "ellip1", "width": 160},
        "label": {"type": "ellip1", "width": 160},
        "model_name": {"type": "ellip1", "width": 180},
        "service": {"type": "ellip1", "width": 180},
        "gateway_url": {"type": "ellip2", "width": 360},
        "api_key": {"type": "ellip2", "width": 300},
        "token_quota": {"type": "ellip1", "width": 120},
        "operate_html": {"type": "ellip1", "width": 180},
    }

    def pre_add_web(self):
        self.add_form_extra_fields['api_key'].kwargs['default'] = generate_api_key()
        self.add_form_extra_fields['token_quota'].kwargs['default'] = 'no-limit'

    def _normalize_req(self, req_json, src_item=None):
        req_json = dict(req_json or {})
        service_value = req_json.get('service') or req_json.get('service_id')
        service = None
        if service_value:
            service_id = getattr(service_value, 'id', service_value)
            service = db.session.query(InferenceService).filter_by(id=int(service_id)).first()
        elif src_item:
            service = src_item.service
        if not service:
            raise Exception('backend service is required')

        req_json['service'] = service.id
        req_json['project'] = getattr(req_json.get('project'), 'id', req_json.get('project') or (src_item.project_id if src_item else service.project_id))
        req_json['label'] = req_json.get('label') or req_json.get('name') or (src_item.label if src_item else '')
        req_json['model_name'] = req_json.get('model_name') or (src_item.model_name if src_item else '') or service.model_name or service.name
        req_json['service_type'] = 'inferenceservice'
        req_json['status'] = req_json.get('status') or (src_item.status if src_item else 'enabled')
        req_json['api_key'] = req_json.get('api_key') or (src_item.api_key if src_item else generate_api_key())
        req_json['token_quota'] = normalize_token_quota(req_json.get('token_quota') or (src_item.token_quota if src_item else 'no-limit'))
        if not req_json.get('expire_at') and not src_item:
            req_json['expire_at'] = default_expire_at().strftime('%Y-%m-%d %H:%M:%S')
        return req_json

    def pre_add_req(self, req_json, *args, **kwargs):
        return self._normalize_req(req_json)

    def pre_update_req(self, req_json, *args, **kwargs):
        return self._normalize_req(req_json, kwargs.get('src_item'))

    def set_columns_related(self, exist_add_args, response_add_columns):
        if 'api_key' in response_add_columns and not exist_add_args.get('api_key'):
            response_add_columns['api_key']['default'] = generate_api_key()
        if 'token_quota' in response_add_columns and not exist_add_args.get('token_quota'):
            response_add_columns['token_quota']['default'] = 'no-limit'

    @expose_api(description="调用示例", url="/example/<gateway_id>", methods=["GET"])
    def example(self, gateway_id):
        gateway = db.session.query(LlmGateway).filter_by(id=int(gateway_id)).first()
        if not gateway:
            return self.response(404, status=1, message='gateway not found')
        api_version = conf.get('LLM_GATEWAY_API_VERSION', '1')
        base_url = request.host_url.strip('/') + f'/llm/api/{api_version}/v1'
        url = base_url + '/chat/completions'
        payload = {
            "model": gateway.model_name,
            "messages": [{"role": "user", "content": "你是谁？"}],
            "temperature": 0.7,
            "stream": False,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, indent=4)
        curl = (
            f"curl -X POST {url} \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -H 'Authorization: Bearer {gateway.api_key}' \\\n"
            f"  -d '{payload_text}'"
        )
        return self.response(200, status=0, message='success', result={
            "url": url,
            "base_url": base_url,
            "model": gateway.model_name,
            "api_key": gateway.api_key,
            "curl": curl,
            "python": {
                "API_SECRET_KEY": gateway.api_key,
                "BASE_URL": base_url,
                "MODEL_NAME": gateway.model_name,
            }
        })

    @expose_api(description="调用测试", url="/test/<gateway_id>", methods=["POST", "GET"])
    def test(self, gateway_id):
        gateway = db.session.query(LlmGateway).filter_by(id=int(gateway_id)).first()
        if not gateway:
            return self.response(404, status=1, message='gateway not found')
        body = request.get_json(silent=True) or {
            "model": gateway.model_name,
            "messages": [{"role": "user", "content": "你是谁？"}],
            "temperature": 0.7,
            "stream": False,
        }
        try:
            response = proxy_chat_completions(gateway, body)
            summary = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text[:2000]
            return self.response(200, status=0, message='success', result={
                "backend_status_code": response.status_code,
                "response": summary,
            })
        except Exception as err:
            return self.response(500, status=1, message=str(err))

    @expose_api(description="监控统计", url="/monitor/<gateway_id>", methods=["GET"])
    def monitor(self, gateway_id):
        gateway_id = int(gateway_id)
        gateway = db.session.query(LlmGateway).filter_by(id=gateway_id).first()
        if not gateway:
            return self.response(404, status=1, message='gateway not found')

        now = datetime.datetime.now()
        today_start = datetime.datetime(now.year, now.month, now.day)
        yesterday_start = today_start - datetime.timedelta(days=1)
        tomorrow_start = today_start + datetime.timedelta(days=1)
        last_24h = now - datetime.timedelta(hours=24)

        base_query = db.session.query(LlmGatewayLog).filter_by(gateway_id=gateway_id)
        hourly_counts = {}
        recent_logs = base_query.filter(LlmGatewayLog.created_on >= last_24h).all()
        for log in recent_logs:
            if not log.created_on:
                continue
            hour = log.created_on.replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:00:00')
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        today_hourly_counts = [0] * 24
        yesterday_hourly_counts = [0] * 24
        day_logs = base_query.filter(
            LlmGatewayLog.created_on >= yesterday_start,
            LlmGatewayLog.created_on < tomorrow_start
        ).all()
        for log in day_logs:
            if not log.created_on:
                continue
            if log.created_on >= today_start:
                today_hourly_counts[log.created_on.hour] += 1
            elif log.created_on >= yesterday_start:
                yesterday_hourly_counts[log.created_on.hour] += 1
        return self.response(200, status=0, message='success', result={
            "today_date": today_start.strftime('%Y-%m-%d'),
            "yesterday_date": yesterday_start.strftime('%Y-%m-%d'),
            "today_count": base_query.filter(LlmGatewayLog.created_on >= today_start).count(),
            "yesterday_count": base_query.filter(LlmGatewayLog.created_on >= yesterday_start, LlmGatewayLog.created_on < today_start).count(),
            "total_count": base_query.count(),
            "success_count": base_query.filter(LlmGatewayLog.success == True).count(),
            "failed_count": base_query.filter(LlmGatewayLog.success == False).count(),
            "hourly": [{"hour": hour, "count": hourly_counts[hour]} for hour in sorted(hourly_counts)],
            "today_hourly": [{"hour": f"{hour:02d}:00", "count": today_hourly_counts[hour]} for hour in range(24)],
            "yesterday_hourly": [{"hour": f"{hour:02d}:00", "count": yesterday_hourly_counts[hour]} for hour in range(24)],
        })

    @expose_api(description="重置API Key", url="/reset_key/<gateway_id>", methods=["POST", "GET"])
    def reset_key(self, gateway_id):
        gateway = db.session.query(LlmGateway).filter_by(id=int(gateway_id)).first()
        if not gateway:
            return self.response(404, status=1, message='gateway not found')
        api_key = generate_api_key()
        gateway.api_key = api_key
        db.session.commit()
        return self.response(200, status=0, message='success', result={
            "api_key": api_key,
            "api_key_prefix": api_key_prefix(gateway.api_key),
        })


class LlmGateway_ModelView_Api(LlmGateway_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(LlmGateway)
    route_base = '/llm_gateway_modelview/api'


appbuilder.add_api(LlmGateway_ModelView_Api)
