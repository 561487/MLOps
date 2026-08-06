"""Pipeline-scoped Pod resource monitoring and DingTalk alerts."""
import datetime
import json
import logging
import os
import time
from kubernetes import client, config
from kubernetes.utils.quantity import parse_quantity
from myapp import app, db
from myapp.models.model_job import Workflow
from myapp.models.model_notification import NotificationChannel
from myapp.tools.dingtalk_notifier import send_dingtalk_message
from myapp.utils.py.py_prometheus import Prometheus

LOG=logging.getLogger('workflow-resource-monitor')
INTERVAL=int(os.getenv('RESOURCE_MONITOR_INTERVAL','10'))
PROM=Prometheus(os.getenv('PROMETHEUS_BASE_URL') or app.config.get('PROMETHEUS_BASE_URL') or app.config.get('PROMETHEUS',''),5)
STATE={}

def _value(query):
    rows=PROM.instant_query(query)
    if not rows: return None
    try: return float(rows[0]['value'][1])
    except (KeyError,ValueError,TypeError,IndexError): return None

def _pod_name(node_id,node):
    name=(node.get('name') or '').replace('.','-')
    suffix=(node.get('id') or node_id).rsplit('-',1)[-1]
    return '%s-%s'%(name,suffix) if name and not name.endswith(suffix) else (node.get('id') or node_id)

def _requests(pod):
    cpu=mem=0.0
    for c in pod.spec.containers or []:
        req=(c.resources.requests or {}) if c.resources else {}
        cpu+=float(parse_quantity(req.get('cpu','0')))
        mem+=float(parse_quantity(req.get('memory','0')))
    return cpu,mem

def collect(namespace,pod_name,pod):
    cpu_req,mem_req=_requests(pod)
    cpu=_value('sum(rate(container_cpu_usage_seconds_total{namespace="%s",pod="%s",container!="POD",container!=""}[1m]))'%(namespace,pod_name))
    mem=_value('sum(container_memory_working_set_bytes{namespace="%s",pod="%s",container!="POD",container!=""})'%(namespace,pod_name))
    gpu=_value('avg(DCGM_FI_DEV_GPU_UTIL{namespace="%s",pod="%s"})'%(namespace,pod_name))
    fb_used=_value('sum(DCGM_FI_DEV_FB_USED{namespace="%s",pod="%s"})'%(namespace,pod_name))
    fb_free=_value('sum(DCGM_FI_DEV_FB_FREE{namespace="%s",pod="%s"})'%(namespace,pod_name))
    temp=_value('max(DCGM_FI_DEV_GPU_TEMP{namespace="%s",pod="%s"})'%(namespace,pod_name))
    gpu_mem=(fb_used/(fb_used+fb_free)*100) if fb_used is not None and fb_free is not None and fb_used+fb_free else None
    return {'cpu_pct':cpu/cpu_req*100 if cpu is not None and cpu_req else None,
            'cpu_cores':cpu,'memory_pct':mem/mem_req*100 if mem is not None and mem_req else None,
            'memory_gib':mem/1024**3 if mem is not None else None,'gpu_pct':gpu,
            'gpu_memory_pct':gpu_mem,'gpu_temperature':temp}

def _send(channel,wf,pod_name,node_name,title,body,event,dedup):
    pipeline_name=json.loads(wf.info_json or '{}').get('describe') or wf.name
    if pod_name == '多个 Pod':
        message='【%s】\nPipeline：%s\n%s'%(title,pipeline_name,body)
        link={}
    else:
        message='【%s】\nPipeline：%s\n节点：%s\nPod：%s\n%s'%(title,pipeline_name,node_name,pod_name,body)
        link={'查看 Pod 日志':'http://%s/k8s/web/log/%s/%s/%s/main'%(app.config.get('HOST'),wf.cluster,wf.namespace,pod_name)}
    return send_dingtalk_message([wf.username],message,link,event_type=event,dedup_key=dedup,force=True,channel_id=channel.id)

def _check_threshold(key,label,value,threshold,duration,silence,now,channel,wf,pod,node,state):
    if value is None or value < threshold:
        state['since'].pop(key,None); return
    since=state['since'].setdefault(key,now)
    last=state['sent'].get(key,0)
    if now-since>=duration and now-last>=silence:
        unit = '℃' if key == 'gpu_temperature' else ('秒' if key == 'pending' else '%')
        body='%s：%.1f%s（阈值 %s%s，持续 %s 秒）'%(label,value,unit,threshold,unit,duration)
        if _send(channel,wf,pod,node,'MLOps 资源预警',body,'resource.warning','%s:%s:%s:%s'%(wf.cluster,wf.name,pod,key)):
            state['sent'][key]=now

def monitor_once(v1):
    channel=db.session.query(NotificationChannel).filter_by(channel_type='dingtalk').first()
    if not channel or not channel.enabled or not getattr(channel,'resource_monitor_enabled',False): return
    workflows=db.session.query(Workflow).filter(Workflow.status.in_(['Pending','Running'])).all()
    active=set(); now=time.time()
    for wf in workflows:
        active.add(wf.id); state=STATE.setdefault(wf.id,{'peaks':{},'since':{},'sent':{},'pods':set()})
        try: nodes=json.loads(wf.status_more or '{}').get('nodes') or {}
        except ValueError: nodes={}
        for node_id,node in nodes.items():
            if node.get('type')!='Pod' or node.get('phase') not in ('Pending','Running'): continue
            pod_name=_pod_name(node_id,node); node_name=node.get('displayName') or pod_name; state['pods'].add(pod_name)
            try: pod=v1.read_namespaced_pod(pod_name,wf.namespace)
            except Exception: continue
            if node.get('phase')=='Pending':
                created=pod.metadata.creation_timestamp.timestamp() if pod.metadata.creation_timestamp else now
                if now-created>=channel.pending_timeout:
                    _check_threshold('pending','Pending时间',now-created,channel.pending_timeout,0,channel.silence_seconds,now,channel,wf,pod_name,node_name,state)
                continue
            metrics=collect(wf.namespace,pod_name,pod)
            for k,v in metrics.items():
                if v is not None: state['peaks'][k]=max(state['peaks'].get(k,v),v)
            duration=getattr(channel,'threshold_duration',120); silence=channel.silence_seconds
            for args in [('cpu','CPU使用率',metrics['cpu_pct'],getattr(channel,'cpu_threshold',90)),('memory','内存使用率',metrics['memory_pct'],getattr(channel,'memory_threshold',90)),('gpu','GPU利用率',metrics['gpu_pct'],getattr(channel,'gpu_threshold',95)),('gpu_memory','GPU显存使用率',metrics['gpu_memory_pct'],getattr(channel,'gpu_memory_threshold',90))]:
                _check_threshold(*args,duration,silence,now,channel,wf,pod_name,node_name,state)
            temp=metrics['gpu_temperature']
            if temp is not None:
                _check_threshold('gpu_temperature','GPU温度',temp,getattr(channel,'gpu_temperature_threshold',85),duration,silence,now,channel,wf,pod_name,node_name,state)
            for cs in list(pod.status.init_container_statuses or [])+list(pod.status.container_statuses or []):
                term=getattr(cs.state,'terminated',None)
                if term and term.reason=='OOMKilled' and now-state['sent'].get('oom',0)>=silence:
                    if _send(channel,wf,pod_name,node_name,'MLOps 资源异常','异常：OOMKilled\n退出码：%s'%term.exit_code,'resource.oom','%s:%s:%s:oom'%(wf.cluster,wf.name,pod_name)): state['sent']['oom']=now
    for wf_id in list(STATE):
        if wf_id in active: continue
        state=STATE.pop(wf_id); wf=db.session.query(Workflow).filter_by(id=wf_id).first()
        if wf and getattr(channel,'completion_summary',True) and state['peaks']:
            p=state['peaks']; body='资源峰值：\nCPU（相对 Pod 申请值）：%s%%\n内存：%s GiB（相对申请值 %s%%）\nGPU：%s%%\nGPU显存：%s%%\nGPU最高温度：%s℃'%tuple(round(p.get(k,0),1) for k in ('cpu_pct','memory_gib','memory_pct','gpu_pct','gpu_memory_pct','gpu_temperature'))
            _send(channel,wf,'多个 Pod','Pipeline汇总','MLOps 资源摘要',body,'resource.summary','%s:%s:summary'%(wf.cluster,wf.name))

def main():
    cluster=os.getenv('ENVIRONMENT','dev').lower()
    kube=(app.config.get('CLUSTERS',{}).get(cluster,{}) or {}).get('KUBECONFIG')
    if kube and os.path.exists(kube) and ''.join(open(kube).readlines()).strip():
        config.load_kube_config(config_file=kube)
    else:
        config.load_incluster_config()
    v1=client.CoreV1Api()
    while True:
        try:
            with app.app_context(): monitor_once(v1)
        except Exception: LOG.exception('resource monitor iteration failed')
        time.sleep(INTERVAL)
if __name__=='__main__': main()