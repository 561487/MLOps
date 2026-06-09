import datetime
from flask import request
from flask_appbuilder import Model
from markupsafe import Markup, escape
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, db
from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable
from myapp.models.model_serving import InferenceService
from myapp.models.model_team import Project


conf = app.config


class LlmGateway(Model, AuditMixinNullable, MyappModelBase):
    __tablename__ = 'llm_gateway'

    id = Column(Integer, primary_key=True, comment='id主键')
    name = Column(String(100), nullable=False, unique=True, comment='英文名')
    label = Column(String(100), nullable=True, comment='显示名')
    model_name = Column(String(200), nullable=False, comment='对外模型名')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False, comment='项目组id')
    project = relationship(Project, foreign_keys=[project_id], lazy='selectin')
    service_type = Column(String(50), nullable=False, default='inferenceservice', comment='后端服务类型')
    service_id = Column(Integer, ForeignKey('inferenceservice.id'), nullable=False, comment='推理服务id')
    service = relationship(InferenceService, foreign_keys=[service_id], lazy='selectin')
    api_key = Column(String(128), nullable=False, unique=True, comment='API Key')
    token_quota = Column(String(50), nullable=False, default='no-limit', comment='token额度')
    expire_at = Column(DateTime, nullable=True, comment='过期时间')
    status = Column(String(50), nullable=False, default='enabled', comment='状态')

    label_columns = {
        "name": _("名称"),
        "label": _("显示名"),
        "model_name": _("模型名称"),
        "project": _("项目组"),
        "service": _("后端服务"),
        "api_key": _("API Key"),
        "token_quota": _("token额度"),
        "gateway_url": _("统一URL"),
        "status": _("状态"),
        "call_count": _("调用次数"),
        "last_call_at": _("最近调用"),
        "operate_html": _("操作"),
    }

    @property
    def gateway_url(self):
        api_version = conf.get('LLM_GATEWAY_API_VERSION', '1')
        host = request.host_url.strip('/') if request else ''
        return f'{host}/llm/api/{api_version}/v1/chat/completions'

    @property
    def call_count(self):
        return db.session.query(LlmGatewayLog).filter_by(gateway_id=self.id).count()

    @property
    def last_call_at(self):
        log = (
            db.session.query(LlmGatewayLog)
            .filter_by(gateway_id=self.id)
            .order_by(LlmGatewayLog.created_on.desc())
            .first()
        )
        return log.created_on.strftime('%Y-%m-%d %H:%M:%S') if log and log.created_on else ''

    @property
    def operate_html(self):
        onclick = """
(function(gatewayId){
  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, function(ch) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch];
    });
  }
  function codeBlock(text) {
    var lines = String(text || '').split('\\n');
    return '<pre style="margin:0;background:#050505;color:#f8f8f2;border-radius:6px;padding:14px 0;font:14px/1.7 Consolas,Monaco,Menlo,monospace;white-space:pre;overflow:auto;counter-reset:llm-line;">' + lines.map(function(line, index) {
      return '<span style="display:block;padding:0 18px 0 54px;position:relative;"><span style="position:absolute;left:18px;color:#7a7a7a;">' + (index + 1) + '</span>' + escapeHtml(line) + '</span>';
    }).join('') + '</pre>';
  }
  function ensureDialog() {
    var mask = document.getElementById('llm-example-mask');
    if (mask) {
      return mask;
    }
    mask = document.createElement('div');
    mask.id = 'llm-example-mask';
    mask.style.cssText = 'position:fixed;z-index:9999;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,.35);display:none;align-items:flex-start;justify-content:center;padding-top:60px;';
    mask.innerHTML = '<div style="width:min(980px,92vw);max-height:86vh;overflow:auto;background:#fff;border-radius:8px;box-shadow:0 12px 36px rgba(0,0,0,.25);padding:0 0 18px;color:#333;"><div style="display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid #eee;font-size:20px;"><div>llm api 调用</div><button type="button" style="border:0;background:transparent;font-size:30px;line-height:1;color:#999;cursor:pointer;">&times;</button></div><div id="llm-example-body" style="padding:18px 20px;"></div></div>';
    document.body.appendChild(mask);
    mask.querySelector('button').onclick = function(){ mask.style.display = 'none'; };
    mask.onclick = function(event) {
      if (event.target === mask) {
        mask.style.display = 'none';
      }
    };
    return mask;
  }
  function renderExample(result) {
    var env = 'API_SECRET_KEY = "' + result.api_key + '"\\nBASE_URL = "' + result.base_url + '"';
    var params = '<ul style="line-height:1.8;margin:0 0 0 20px;padding:0;">'
      + '<li>MODEL_NAME：替换为你的模型名称，当前可访问模型名包含: ' + escapeHtml(result.model) + '</li>'
      + '<li>messages：对话历史，包含 role（user / assistant / system）和 content（消息内容）</li>'
      + '<li>temperature：控制生成文本的随机性（0-2，值越大越随机）</li>'
      + '</ul>';
    return '<div style="margin:0 0 16px;"><div style="font-size:18px;margin:0 0 10px;">外部平台环境变量配置</div>' + codeBlock(env) + '</div>'
      + '<div style="margin:0 0 16px;"><div style="font-size:18px;margin:0 0 10px;">API调用示例代码</div>' + codeBlock(result.curl) + '</div>'
      + '<div style="margin:0 0 16px;"><div style="font-size:18px;margin:0 0 10px;">参数说明：</div>' + params + '</div>';
  }
  var mask = ensureDialog();
  var body = document.getElementById('llm-example-body');
  body.innerHTML = '<div style="margin:0 0 16px;">加载中...</div>';
  mask.style.display = 'flex';
  fetch('/llm_gateway_modelview/api/example/' + gatewayId, {credentials: 'same-origin'})
    .then(function(response){ return response.json(); })
    .then(function(data) {
      if (!data || data.status !== 0) {
        throw new Error((data && data.message) || '获取调用示例失败');
      }
      body.innerHTML = renderExample(data.result || {});
    })
    .catch(function(error) {
      body.innerHTML = '<div style="margin:0 0 16px;">' + escapeHtml(error.message) + '</div>';
    });
})(this.getAttribute('data-gateway-id')); return false;
"""
        monitor_onclick = """
(function(gatewayId){
  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, function(ch) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch];
    });
  }
  function ensureDialog() {
    var mask = document.getElementById('llm-monitor-mask');
    if (mask) {
      return mask;
    }
    mask = document.createElement('div');
    mask.id = 'llm-monitor-mask';
    mask.style.cssText = 'position:fixed;z-index:9999;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,.35);display:none;align-items:flex-start;justify-content:center;padding-top:36px;';
    mask.innerHTML = '<div style="width:860px;max-width:92vw;max-height:88vh;overflow:auto;background:#fff;border-radius:2px;box-shadow:0 12px 36px rgba(0,0,0,.25);padding:0;color:#333;"><div style="display:flex;align-items:center;justify-content:space-between;padding:20px 28px;border-bottom:1px solid #eee;font-size:20px;"><div>llm api 调用监控</div><button type="button" style="border:0;background:transparent;font-size:34px;line-height:1;color:#999;cursor:pointer;">&times;</button></div><div id="llm-monitor-body" style="padding:18px 28px 28px;"></div></div>';
    document.body.appendChild(mask);
    mask.querySelector('button').onclick = function(){ mask.style.display = 'none'; };
    mask.onclick = function(event) {
      if (event.target === mask) {
        mask.style.display = 'none';
      }
    };
    return mask;
  }
  function drawChart(rows, title) {
    rows = rows || [];
    var width = 760;
    var height = 360;
    var left = 70;
    var right = 28;
    var top = 54;
    var bottom = 64;
    var chartWidth = width - left - right;
    var chartHeight = height - top - bottom;
    var maxValue = Math.max.apply(null, rows.map(function(item){ return Number(item.count || 0); }).concat([1]));
    var points = rows.map(function(item, index) {
      var x = left + (rows.length <= 1 ? 0 : index * chartWidth / (rows.length - 1));
      var y = top + chartHeight - (Number(item.count || 0) * chartHeight / maxValue);
      return x.toFixed(2) + ',' + y.toFixed(2);
    }).join(' ');
    var xLabels = rows.map(function(item, index) {
      if (index % 2 !== 0) {
        return '';
      }
      var x = left + index * chartWidth / Math.max(rows.length - 1, 1);
      return '<text x="' + x + '" y="' + (height - 38) + '" text-anchor="middle" fill="#666" font-size="12">' + escapeHtml(item.hour.replace(':00', 'h')) + '</text>';
    }).join('');
    var yLabels = [0, maxValue].map(function(value) {
      var y = top + chartHeight - (value * chartHeight / maxValue);
      return '<text x="' + (left - 16) + '" y="' + (y + 4) + '" text-anchor="end" fill="#666" font-size="12">' + value + '</text>'
        + '<line x1="' + left + '" y1="' + y + '" x2="' + (width - right) + '" y2="' + y + '" stroke="#e6e6e6"/>';
    }).join('');
    var circles = rows.map(function(item, index) {
      var x = left + index * chartWidth / Math.max(rows.length - 1, 1);
      var y = top + chartHeight - (Number(item.count || 0) * chartHeight / maxValue);
      return '<circle cx="' + x + '" cy="' + y + '" r="3" fill="#1e88e5"><title>' + escapeHtml(item.hour + ' 调用次数: ' + (item.count || 0)) + '</title></circle>';
    }).join('');
    return '<svg viewBox="0 0 ' + width + ' ' + height + '" style="width:100%;height:420px;background:#f5f6f8;display:block;">'
      + '<text x="' + (width / 2) + '" y="44" text-anchor="middle" fill="#333" font-size="22" font-weight="600">' + escapeHtml(title) + ' 指标监控</text>'
      + yLabels
      + '<line x1="' + left + '" y1="' + (top + chartHeight) + '" x2="' + (width - right) + '" y2="' + (top + chartHeight) + '" stroke="#333"/>'
      + '<line x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + (top + chartHeight) + '" stroke="#333"/>'
      + '<polyline points="' + points + '" fill="none" stroke="#1e88e5" stroke-width="3"/>'
      + circles
      + xLabels
      + '<text x="' + (width / 2) + '" y="' + (height - 10) + '" text-anchor="middle" fill="#666" font-size="12">小时</text>'
      + '</svg>';
  }
  function render(result) {
    var body = document.getElementById('llm-monitor-body');
    var active = 'today';
    function setActive(day) {
      active = day;
      var rows = day === 'today' ? result.today_hourly : result.yesterday_hourly;
      var date = day === 'today' ? result.today_date : result.yesterday_date;
      var count = day === 'today' ? result.today_count : result.yesterday_count;
      body.querySelector('#llm-monitor-chart').innerHTML = drawChart(rows, date);
      body.querySelector('#llm-monitor-summary').innerHTML = '调用次数：' + count + '，成功：' + result.success_count + '，失败：' + result.failed_count;
      body.querySelector('[data-day="today"]').style.cssText = day === 'today' ? 'padding:0 0 12px;border:0;border-bottom:3px solid #1e88e5;background:transparent;color:#1e88e5;cursor:pointer;' : 'padding:0 0 12px;border:0;border-bottom:3px solid transparent;background:transparent;color:#333;cursor:pointer;';
      body.querySelector('[data-day="yesterday"]').style.cssText = day === 'yesterday' ? 'padding:0 0 12px;border:0;border-bottom:3px solid #1e88e5;background:transparent;color:#1e88e5;cursor:pointer;' : 'padding:0 0 12px;border:0;border-bottom:3px solid transparent;background:transparent;color:#333;cursor:pointer;';
    }
    body.innerHTML = '<div style="display:flex;gap:34px;border-bottom:1px solid #eee;margin:0 0 40px;"><button type="button" data-day="today">今天</button><button type="button" data-day="yesterday">昨天</button></div><div id="llm-monitor-chart"></div><div id="llm-monitor-summary" style="margin-top:14px;color:#666;line-height:1.8;"></div>';
    body.querySelector('[data-day="today"]').onclick = function(){ setActive('today'); };
    body.querySelector('[data-day="yesterday"]').onclick = function(){ setActive('yesterday'); };
    setActive(active);
  }
  var mask = ensureDialog();
  var body = document.getElementById('llm-monitor-body');
  body.innerHTML = '<div style="margin:0 0 16px;">加载中...</div>';
  mask.style.display = 'flex';
  fetch('/llm_gateway_modelview/api/monitor/' + gatewayId, {credentials: 'same-origin'})
    .then(function(response){ return response.json(); })
    .then(function(data) {
      if (!data || data.status !== 0) {
        throw new Error((data && data.message) || '获取监控数据失败');
      }
      render(data.result || {});
    })
    .catch(function(error) {
      body.innerHTML = '<div style="margin:0 0 16px;">' + escapeHtml(error.message) + '</div>';
    });
})(this.getAttribute('data-gateway-id')); return false;
"""
        return Markup(
            f'<a href="javascript:void(0)" data-gateway-id="{self.id}" onclick="{escape(onclick)}">{__("调用示例")}</a> | '
            f'<a href="javascript:void(0)" data-gateway-id="{self.id}" onclick="{escape(monitor_onclick)}">{__("监控")}</a>'
        )

    def __repr__(self):
        return self.name


class LlmGatewayLog(Model):
    __tablename__ = 'llm_gateway_log'

    id = Column(Integer, primary_key=True, comment='id主键')
    gateway_id = Column(Integer, ForeignKey('llm_gateway.id'), nullable=True, index=True, comment='服务网关id')
    gateway = relationship(LlmGateway, foreign_keys=[gateway_id], lazy='selectin')
    request_id = Column(String(100), nullable=True, index=True, comment='请求id')
    model_name = Column(String(200), nullable=True, index=True, comment='模型名')
    api_key_prefix = Column(String(32), nullable=True, comment='API Key前缀')
    client_ip = Column(String(100), nullable=True, comment='客户端IP')
    path = Column(String(500), nullable=True, comment='请求路径')
    method = Column(String(20), nullable=True, comment='请求方法')
    status_code = Column(Integer, nullable=True, comment='状态码')
    success = Column(Boolean, nullable=False, default=False, index=True, comment='是否成功')
    latency_ms = Column(Integer, nullable=True, comment='耗时毫秒')
    prompt_tokens = Column(Integer, nullable=True, default=0, comment='输入token')
    completion_tokens = Column(Integer, nullable=True, default=0, comment='输出token')
    total_tokens = Column(Integer, nullable=True, default=0, comment='总token')
    error_message = Column(Text, nullable=True, comment='错误信息')
    created_on = Column(DateTime, nullable=True, default=datetime.datetime.now, index=True, comment='创建时间')
