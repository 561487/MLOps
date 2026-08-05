"""Best-effort Kubernetes diagnostics for failed Argo workflows."""
import json
import logging
import re
from kubernetes import client

_SECRET_PATTERNS = (
    re.compile(r'(?i)((?:api[_-]?key|token|password|passwd|secret)\s*[=:]\s*)[^\s,;]+'),
    re.compile(r'(?i)(authorization\s*:\s*(?:bearer\s+)?)[^\s,;]+'),
)
_ERROR_PATTERN = re.compile(
    r'(?i)(exception|error|failed|failure|not found|no such file|permission denied|'
    r'out of memory|oom|invalid|unsupported|cannot|could not|refused)'
)
_GENERIC_REASONS = {'', 'error', 'failed', 'node failed', 'workflow execution failed'}


def sanitize_text(value, limit=2000):
    text = str(value or '').replace('\x00', '').strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r'\1***', text)
    return ('...（内容已截断）\n' + text[-limit:]) if len(text) > limit else text


def extract_error_summary(log_text):
    """Prefer the task's structured error, then the last useful error line."""
    lines = [line.strip() for line in str(log_text or '').splitlines() if line.strip()]
    for line in reversed(lines):
        json_start = line.find('{')
        if json_start >= 0:
            try:
                payload = json.loads(line[json_start:])
                if payload.get('error'):
                    error_type = payload.get('error_type') or 'Error'
                    return '%s: %s' % (error_type, payload['error'])
            except (TypeError, ValueError):
                pass
    for line in reversed(lines):
        plain = re.sub(r'^\S+Z\s+', '', line).strip()
        if plain.lower().startswith('traceback (most recent call last)'):
            continue
        if re.match(r'(?i)^error:\s*exit status \d+$', plain):
            continue
        if _ERROR_PATTERN.search(plain):
            return plain
    return ''


def resolve_pod_name(node_id, node):
    """Argo node id omits the template name; main-logs contains the real Pod name."""
    artifacts = (node.get('outputs') or {}).get('artifacts') or []
    for artifact in artifacts:
        if artifact.get('name') != 'main-logs':
            continue
        key = ((artifact.get('s3') or {}).get('key') or '').strip('/')
        parts = key.split('/')
        if len(parts) >= 2 and parts[-1] == 'main.log':
            return parts[-2]
    return node.get('id') or node_id


def collect_failure_detail(workflow, tail_lines=50, limit=2000, core_api=None):
    try:
        status = json.loads(workflow.status_more or '{}')
    except (TypeError, ValueError):
        status = {}
    failed = [(key, node) for key, node in (status.get('nodes') or {}).items()
              if node.get('phase') in ('Failed', 'Error')]
    failed.sort(key=lambda item: item[1].get('type') != 'Pod')
    node_id, node = failed[0] if failed else ('', {})
    pod_name = resolve_pod_name(node_id, node)
    argo_reason = node.get('message') or status.get('message') or 'Workflow execution failed'
    detail = {
        'node': node.get('displayName') or pod_name,
        'pod': pod_name if node.get('type') == 'Pod' else '',
        'reason': argo_reason,
        'container_status': '',
        'exit_code': (node.get('outputs') or {}).get('exitCode', ''),
        'event': '',
        'log': '',
    }
    if not detail['pod']:
        return detail
    api = core_api or client.CoreV1Api()
    try:
        pod = api.read_namespaced_pod(detail['pod'], workflow.namespace)
        statuses = list(pod.status.init_container_statuses or []) + list(pod.status.container_statuses or [])
        statuses.sort(key=lambda item: item.name != 'main')
        for item in statuses:
            terminated = getattr(item.state, 'terminated', None)
            waiting = getattr(item.state, 'waiting', None)
            if terminated:
                detail['container_status'] = terminated.reason or 'Terminated'
                if terminated.message:
                    detail['reason'] = terminated.message
                detail['exit_code'] = terminated.exit_code
                break
            if waiting:
                detail['container_status'] = waiting.reason or 'Waiting'
                if waiting.message:
                    detail['reason'] = waiting.message
                break
    except Exception as exc:
        logging.info('cannot read failed pod %s: %s', detail['pod'], exc)
    try:
        events = api.list_namespaced_event(
            workflow.namespace, field_selector='involvedObject.name=%s' % detail['pod']).items
        warnings = [event for event in events if event.type == 'Warning']
        if warnings:
            detail['event'] = '%s: %s' % (warnings[-1].reason or 'Warning', warnings[-1].message or '')
    except Exception as exc:
        logging.info('cannot read pod events %s: %s', detail['pod'], exc)
    try:
        detail['log'] = api.read_namespaced_pod_log(
            detail['pod'], workflow.namespace, container='main',
            tail_lines=tail_lines, timestamps=True)
    except Exception as exc:
        logging.info('cannot read pod log %s: %s', detail['pod'], exc)
    error_summary = extract_error_summary(detail['log'])
    reason_key = re.sub(r'\s*\(exit code \d+\)\s*$', '', str(detail['reason']), flags=re.I).strip().lower()
    if error_summary and (reason_key in _GENERIC_REASONS or len(error_summary) > len(str(detail['reason']))):
        detail['reason'] = error_summary
    for key in ('reason', 'container_status', 'event', 'log'):
        detail[key] = sanitize_text(detail.get(key), limit)
    return detail


def format_failure_detail(detail, tail_lines=50):
    lines = ['失败详情：']
    fields = (('失败节点', 'node'), ('Pod', 'pod'), ('具体原因', 'reason'),
              ('容器状态', 'container_status'), ('退出码', 'exit_code'), ('K8s Event', 'event'))
    for label, key in fields:
        if detail.get(key) not in ('', None):
            lines.append('%s：%s' % (label, detail[key]))
    return '\n'.join(lines)