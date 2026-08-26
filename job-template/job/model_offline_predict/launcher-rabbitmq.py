# -*- coding: utf-8 -*-
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)

import argparse
import datetime
import json
import time
import uuid
import os
import pysnooper
import os,sys
import re
import threading
import psutil
import copy
import traceback
import subprocess
from py_rabbit import Rabbit_info
from kubernetes import client


def log(msg):
    """带时间戳的日志"""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def dump_launcher_diagnostics():
    """启动诊断"""
    log('========== LAUNCHER DIAGNOSTICS ==========')
    log(f'[DIAG] Python: {sys.version}')
    log(f'[DIAG] PID: {os.getpid()}')
    log(f'[DIAG] Hostname: {os.uname().nodename}')
    log(f'[DIAG] Namespace: {os.getenv("KFJ_NAMESPACE", "N/A")}')
    log(f'[DIAG] Pipeline: {os.getenv("KFJ_PIPELINE_NAME", "N/A")}')
    log(f'[DIAG] Task: {os.getenv("KFJ_TASK_NAME", "N/A")}')
    log(f'[DIAG] Run ID: {os.getenv("KFJ_RUN_ID", "N/A")}')
    log(f'[DIAG] Creator: {os.getenv("KFJ_CREATOR", "N/A")}')
    log(f'[DIAG] Runner: {os.getenv("KFJ_RUNNER", "N/A")}')
    log(f'[DIAG] GPU Resource: {os.getenv("KFJ_TASK_RESOURCE_GPU", "N/A")}')
    log(f'[DIAG] CPU Resource: {os.getenv("KFJ_TASK_RESOURCE_CPU", "N/A")}')
    log(f'[DIAG] Memory Resource: {os.getenv("KFJ_TASK_RESOURCE_MEMORY", "N/A")}')
    log(f'[DIAG] Node Selector: {os.getenv("KFJ_TASK_NODE_SELECTOR", "N/A")}')
    log(f'[DIAG] Volume Mount: {os.getenv("KFJ_TASK_VOLUME_MOUNT", "N/A")}')
    log('========== END LAUNCHER DIAGNOSTICS ==========')

# print(os.environ)
from job.pkgs.k8s.py_k8s import K8s
from job.pkgs.k8s.affinity import build_pod_anti_affinity
k8s_client = K8s()

KFJ_NAMESPACE = os.getenv('KFJ_NAMESPACE', '')
KFJ_TASK_ID = os.getenv('KFJ_TASK_ID', '')
KFJ_TASK_NAME = os.getenv('KFJ_TASK_NAME', '')
task_node_selectors = re.split(',|;|\n|\t', os.getenv('KFJ_TASK_NODE_SELECTOR', 'cpu=true,train=true'))
KFJ_TASK_NODE_SELECTOR = {}
for task_node_selector in task_node_selectors:
    KFJ_TASK_NODE_SELECTOR[task_node_selector.split('=')[0]] = task_node_selector.split('=')[1]

KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME', 'public')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_CREATOR = os.getenv('KFJ_CREATOR', '')
KFJ_RUNNER = os.getenv('KFJ_RUNNER','')
KFJ_PIPELINE_NAME = os.getenv('KFJ_PIPELINE_NAME', '')
KFJ_TASK_IMAGES = os.getenv('KFJ_TASK_IMAGES', '')
KFJ_TASK_VOLUME_MOUNT = os.getenv('KFJ_TASK_VOLUME_MOUNT', '')
KFJ_TASK_RESOURCE_CPU = os.getenv('KFJ_TASK_RESOURCE_CPU', '')
KFJ_TASK_RESOURCE_MEMORY = os.getenv('KFJ_TASK_RESOURCE_MEMORY', '')
NUM_WORKER = 3

INIT_FILE=''
crd_info={
    "group": "batch.volcano.sh",
    "version": "v1alpha1",
    'kind': 'Job',
    "plural": "jobs",
    "timeout": 60 * 60 * 24 * 2
}


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)
rabbitmq_name=("rabbitmq-" + KFJ_PIPELINE_NAME.replace('_','-'))[0:54].strip('-')
GPU_RESOURCE_NAME= os.getenv('GPU_RESOURCE_NAME', '')
GPU_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
gpu_num,gpu_type,_ = k8s_client.get_gpu(GPU_RESOURCE)
if gpu_type:
    KFJ_TASK_NODE_SELECTOR['gpu-type']=gpu_type


RDMA_RESOURCE_NAME= os.getenv('RDMA_RESOURCE_NAME', '')
RDMA_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_RDMA', '0')

HUBSECRET = os.getenv('HUBSECRET','hubsecret')
HUBSECRET=[{"name":hubsecret} for hubsecret in HUBSECRET.split(',')]

DEFAULT_POD_RESOURCES = os.getenv('DEFAULT_POD_RESOURCES','')
DEFAULT_POD_RESOURCES = json.loads(DEFAULT_POD_RESOURCES) if DEFAULT_POD_RESOURCES else {}

def check_rabbit_finish():
    try:
        rabbit_client = Rabbit_info(host=rabbitmq_name)
        left_msg_num1 = int(rabbit_client.get_msg_count())
        print("======================= check finish left: %s, datetime: %s"%(left_msg_num1,datetime.datetime.now()),flush=True)
        time.sleep(60)
        left_msg_num2 = int(rabbit_client.get_msg_count())
        print("======================= check finish left: %s, datetime: %s" % (left_msg_num1, datetime.datetime.now()),flush=True)
        if left_msg_num1==0 and left_msg_num2==0:
            return True
    except Exception as e:
        print(e)
    return False


import subprocess
# @pysnooper.snoop()
def run_shell(shell):
    print('begin run shell: %s'%shell,flush=True)
    cmd = subprocess.Popen(shell, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE, universal_newlines=True, shell=True, bufsize=1)
    # 实时输出
    while True:
        line = cmd.stdout.readline()
        status = subprocess.Popen.poll(cmd)
        if status:
            print(status,line,end='', flush=True)
        else:
            print(line, end='', flush=True)
        if status == 0:  # 判断子进程是否结束
            print('shell finish %s'%status,flush=True)
            break

        if status==-9 or status==-15 or status==143:   # 外界触发kill
        # if status:
            print('shell finish %s'%status,flush=True)
            break

    return cmd.returncode






# 监控指定名称的volcanojob
# 监控任务及时宗旨stern进程，这样才能结束程序
def monitoring(crd_k8s,name,namespace):
    time.sleep(10)
    # 杀掉stern 进程
    def get_pid(name):
        '''
         作用：根据进程名获取进程pid
        '''
        pids = psutil.process_iter()
        log("[" + name + "]'s pid search:")
        back=[]
        for pid in pids:
            if name in pid.name():
                log(f'  found pid: {pid.pid}')
                back.append(pid.pid)
        return back

    def kill_stern():
        pids = get_pid("stern")
        if pids:
            for pid in pids:
                try:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    log('kill process stern pid=%s' % pid)
                except Exception as e:
                    log(f'kill process error: {e}')

    check_time = datetime.datetime.now()
    status_history = []
    while(True):
        try:
            volcanojob = crd_k8s.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=name)
        except Exception as e:
            log(f'ERROR getting volcanojob status: {e}')
            volcanojob = None

        if volcanojob:
            status = volcanojob.get('status', 'Unknown').lower()
            # 只在状态变化时打印详细信息
            if not status_history or status_history[-1] != status:
                log(f'volcanojob status changed -> {status.upper()}')
            status_history.append(status)
        else:
            log('WARNING: volcanojob not found (may be still creating or already deleted)')

        # 根据volcanojob状态决定任务是否在结束
        if volcanojob and (status=="completed" or status=="failed" or status=='aborted' or status=='terminated'):
            log(f'volcanojob reached terminal state: {status}')
            # 获取更多状态信息
            try:
                log(f'volcanojob full status: {json.dumps(volcanojob, default=str)}')
            except Exception:
                pass
            kill_stern()
            break

        # 定期杀死stern 进程，不然日志追踪有bug，但是不能退出此线程，
        if (datetime.datetime.now()-check_time).total_seconds()>3600:
            kill_stern()

        # 根据队列消费剩余情况监控任务是否该结束
        try:
            rabbit_client = Rabbit_info(host=rabbitmq_name)
            left_msg_num1 = int(rabbit_client.get_msg_count())
            log("======================= left: %s, datetime: %s" % (left_msg_num1, datetime.datetime.now()))
            if not left_msg_num1:
                # 检查队列消费情况
                finish = check_rabbit_finish()
                if finish:
                    kill_stern()
                    log("Queue fully consumed, exiting monitoring")
                    break
        except Exception as e:
            log(f'RabbitMQ check error: {e}')

        time.sleep(60)



# @pysnooper.snoop()
def make_volcanojob(name,num_workers,image,working_dir,command,env):
    # if type(command)==str:
    #     command=command.split(" ")
    #     command = [c for c in command if c]

    # ── 构建基础 Pod 模板（不含 GPU，用于 producer）──
    base_pod_spec = {
        "restartPolicy": "Never",
        "volumes": k8s_volumes,
        "imagePullSecrets": HUBSECRET,
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": node_selector_key,
                                    "operator": "In",
                                    "values": [
                                        KFJ_TASK_NODE_SELECTOR[node_selector_key]
                                    ]
                                } for node_selector_key in KFJ_TASK_NODE_SELECTOR
                            ]
                        }
                    ]
                }
            },
            "podAntiAffinity": build_pod_anti_affinity(
                {"component": name, "type": "volcanojob"},
                num_workers,
            )
        },
        "containers": [
            {
                "name": "volcanojob",
                "image": image if image else KFJ_TASK_IMAGES,
                "imagePullPolicy": "Always",
                "workingDir": working_dir,
                "env": [],
                "command": ['bash', '-c', command],
                "volumeMounts": k8s_volume_mounts,
                "resources": {
                    "requests": {
                        **{
                            "cpu": KFJ_TASK_RESOURCE_CPU,
                            "memory": KFJ_TASK_RESOURCE_MEMORY,
                        },
                        **DEFAULT_POD_RESOURCES
                    },
                    "limits": {
                        **{
                            "cpu": KFJ_TASK_RESOURCE_CPU,
                            "memory": KFJ_TASK_RESOURCE_MEMORY
                        },
                        **DEFAULT_POD_RESOURCES
                    }
                }
            }
        ]
    }

    # 注入自定义 env
    if env:
        for key in env:
            base_pod_spec['containers'][0]['env'].append({
                "name": key,
                "value": env[key]
            })

    # ── 公共 task labels ──
    common_labels = {
        "pipeline-id": KFJ_PIPELINE_ID,
        "pipeline-name": KFJ_PIPELINE_NAME,
        "task-id": KFJ_TASK_ID,
        "task-name": KFJ_TASK_NAME,
        'username': KFJ_RUNNER,
        "component": name,
        "type": "volcanojob",
        "run-id": KFJ_RUN_ID,
    }

    # ── Master task：1 副本，带 GPU ──
    producer_pod_spec = copy.deepcopy(base_pod_spec)
    producer_pod_spec['containers'][0]['env'].append({
        "name": "ROLE",
        "value": "master"
    })

    # 添加 GPU 资源（与 worker 逻辑一致）
    if int(gpu_num) >= 1:
        producer_pod_spec['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
        producer_pod_spec['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
        if 'nodeSelector' not in producer_pod_spec:
            producer_pod_spec['nodeSelector'] = {}
        producer_pod_spec['nodeSelector'].pop('cpu', None)
        producer_pod_spec['nodeSelector']['gpu'] = 'true'
    elif int(gpu_num) < 0:
        shared_count, _, shared_resource_name = k8s_client.get_gpu_shared_resource(GPU_RESOURCE)
        producer_pod_spec['containers'][0]['resources']['requests'][shared_resource_name] = shared_count
        producer_pod_spec['containers'][0]['resources']['limits'][shared_resource_name] = shared_count
        if 'nodeSelector' not in producer_pod_spec:
            producer_pod_spec['nodeSelector'] = {}
        producer_pod_spec['nodeSelector'].pop('cpu', None)
        producer_pod_spec['nodeSelector']['gpu'] = 'true'
    else:
        producer_pod_spec['containers'][0]['env'].append({
            "name": "NVIDIA_VISIBLE_DEVICES",
            "value": "none"
        })

    master_task = {
        "replicas": 1,
        "name": "master",
        "template": {
            "metadata": {
                "labels": common_labels,
                "annotations": {"project": KFJ_TASK_PROJECT_NAME}
            },
            "spec": producer_pod_spec
        },
        "policies": [{"event": "TaskCompleted", "action": "CompleteJob"},
                      {"event": "PodFailed", "action": "AbortJob"}]
    }

    tasks = [master_task]

    # ── Consumer task（worker）：N-1 副本，带 GPU ──
    if int(num_workers) > 1:
        consumer_pod_spec = copy.deepcopy(base_pod_spec)
        consumer_pod_spec['containers'][0]['env'].append({
            "name": "ROLE",
            "value": "worker"
        })

        # 添加 GPU 资源
        if int(gpu_num) >= 1:
            consumer_pod_spec['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
            consumer_pod_spec['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
            if 'nodeSelector' not in consumer_pod_spec:
                consumer_pod_spec['nodeSelector'] = {}
            consumer_pod_spec['nodeSelector'].pop('cpu', None)
            consumer_pod_spec['nodeSelector']['gpu'] = 'true'
        elif int(gpu_num) < 0:
            shared_count, _, shared_resource_name = k8s_client.get_gpu_shared_resource(GPU_RESOURCE)
            consumer_pod_spec['containers'][0]['resources']['requests'][shared_resource_name] = shared_count
            consumer_pod_spec['containers'][0]['resources']['limits'][shared_resource_name] = shared_count
            if 'nodeSelector' not in consumer_pod_spec:
                consumer_pod_spec['nodeSelector'] = {}
            consumer_pod_spec['nodeSelector'].pop('cpu', None)
            consumer_pod_spec['nodeSelector']['gpu'] = 'true'
        else:
            consumer_pod_spec['containers'][0]['env'].append({
                "name": "NVIDIA_VISIBLE_DEVICES",
                "value": "none"
            })

        # 添加 rdma
        if RDMA_RESOURCE_NAME and RDMA_RESOURCE and int(RDMA_RESOURCE):
            consumer_pod_spec['containers'][0]['resources']['requests'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)
            consumer_pod_spec['containers'][0]['resources']['limits'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)
            consumer_pod_spec['containers'][0]['securityContext'] = {
                "capabilities": {"add": ["IPC_LOCK"]}
            }

        worker_task = {
            "replicas": int(num_workers) - 1,
            "name": "worker",
            "template": {
                "metadata": {
                    "labels": common_labels,
                    "annotations": {"project": KFJ_TASK_PROJECT_NAME}
                },
                "spec": consumer_pod_spec
            },
            "policies": [{"event": "TaskCompleted", "action": "CompleteJob"},
                          {"event": "PodFailed", "action": "AbortJob"}]
        }
        tasks.append(worker_task)

    volcano_deploy = {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels":{
                "run-id":KFJ_RUN_ID,
                "run-username":KFJ_RUNNER,
                "pipeline-username": KFJ_CREATOR,
                "pipeline-id": KFJ_PIPELINE_ID,
                "pipeline-name": KFJ_PIPELINE_NAME,
                "task-id": KFJ_TASK_ID,
                "task-name": KFJ_TASK_NAME,
            },
            "annotations": {
                "project": KFJ_TASK_PROJECT_NAME
            }
        },
        "spec": {
            "minAvailable":1,
            "policies": [
                 {
                     "event":"PodFailed",
                     "action": "AbortJob"
                 }
             ],
            "schedulerName":"volcano",
            "cleanPodPolicy": "None",
            "plugins":{
                "env":[],
                "svc":[],
                "ssh":[]
            },
            "queue":"default",
            "tasks": tasks
        }
    }

    return volcano_deploy


# @pysnooper.snoop()
def launch_volcanojob(name, num_workers, image,working_dir, worker_command,env):
    if KFJ_RUN_ID:
        log('delete old volcanojobs by run-id %s'%KFJ_RUN_ID)
        try:
            k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        except Exception as e:
            log(f'delete by run-id error (expected if none exist): {e}')
        time.sleep(10)
    # 删除旧的volcanojob
    log(f'delete old volcanojob by name: {name}')
    try:
        k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    except Exception as e:
        log(f'delete by name error (expected if not exist): {e}')
    time.sleep(10)
    # 创建新的volcanojob
    volcanojob_json = make_volcanojob(name=name,num_workers= num_workers,image = image,working_dir=working_dir,command=worker_command,env=env)
    log(f'volcanojob spec (image={image}, workers={num_workers}, working_dir={working_dir}):')
    log(json.dumps(volcanojob_json, indent=2, default=str))
    log('create new volcanojob %s' % name)
    try:
        k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=volcanojob_json)
        log('volcanojob created successfully')
    except Exception as e:
        log(f'ERROR creating volcanojob: {e}')
        log(f'Full traceback:\n{traceback.format_exc()}')
        raise
    time.sleep(10)

    log('begin start monitoring thread')
    # # 后台启动监控脚本,一直跟踪日志
    monitoring_thread = threading.Thread(target=monitoring,args=(k8s_client,name,KFJ_NAMESPACE))
    monitoring_thread.start()

    while True:
        # 实时打印日志
        line='>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        log('begin follow log\n%s'%line)
        command = '''stern %s --namespace %s --since 10s --template '{{.PodName}} {{.Message}} {{"\\n"}}' '''%(name,KFJ_NAMESPACE)
        log(command)
        run_shell(command)
        log('%s\nend follow log'%line)
        time.sleep(10)

        try:
            volcanojob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        except Exception as e:
            log(f'ERROR getting volcanojob after stern exit: {e}')
            volcanojob = None

        if volcanojob and (volcanojob['status'] == "Completed" or volcanojob['status'] == "Failed"):
            log(f'volcanojob terminal state: {volcanojob["status"]}')
            break

        # 检查队列消费情况
        finish = check_rabbit_finish()
        if finish:
            log('Queue fully consumed, exiting')
            return

    try:
        volcanojob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
        log("volcanojob %s finished, status %s"%(name, volcanojob['status']))

        if volcanojob['status']!='Completed':
            log(f'volcanojob failed with status: {volcanojob["status"]}')
            log(f'Full volcanojob info: {json.dumps(volcanojob, default=str)}')
            exit(1)
    except Exception as e:
        log(f'ERROR getting final volcanojob status: {e}')
        exit(1)


# 创建单机版本rabbitmq
# @pysnooper.snoop()
def create_rabbitmq(name,create=True):
    try:
        pod_str={
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "labels": {
                    "app": name,
                    "run-id": KFJ_RUN_ID,
                    "run-username": KFJ_RUNNER,
                    "pipeline-username": KFJ_CREATOR,
                    "pipeline-id": KFJ_PIPELINE_ID,
                    "pipeline-name": KFJ_PIPELINE_NAME,
                    "task-id": KFJ_TASK_ID,
                    "task-name": KFJ_TASK_NAME,
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "rabbitmq",
                        "image": os.getenv('RABBITMQ_IMAGE',"rabbitmq:3.9.12-management"),
                        "imagePullPolicy": "IfNotPresent",
                        "env":[
                            {
                                "name":"RABBITMQ_DEFAULT_USER",
                                "value":"admin"
                            },
                            {
                                "name":"RABBITMQ_DEFAULT_PASS",
                                "value":"admin"
                            }
                        ]
                    }
                ]
            }
        }
        if create:
            k8s_client.v1.create_namespaced_pod(namespace=KFJ_NAMESPACE,body=pod_str)
        else:
            k8s_client.v1.delete_namespaced_pod(name=name,namespace=KFJ_NAMESPACE,grace_period_seconds=0)
        service_str={
          "apiVersion": "v1",
          "kind": "Service",
          "metadata": {
            "name": name
          },
          "spec": {
            "ports": [
              {
                "name": "app",
                "port": 5672,
                "targetPort": 5672,
                "protocol": "TCP"
              },
              {
                "name": "web",
                "port": 15672,
                "targetPort": 15672,
                "protocol": "TCP"
              }
            ],
            "selector": {
              "app": name
            }
          }
        }
        if create:
            k8s_client.v1.create_namespaced_service(namespace=KFJ_NAMESPACE, body=service_str)
        else:
            k8s_client.v1.delete_namespaced_service(name=name,namespace=KFJ_NAMESPACE,grace_period_seconds=0)
    except Exception as e:
        pass



if __name__ == "__main__":
    dump_launcher_diagnostics()

    arg_parser = argparse.ArgumentParser("volcanojob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录", default='')
    arg_parser.add_argument('--command', type=str, help="运行job的启动命令", default='python3 /app/predict.py')
    arg_parser.add_argument('--num_worker', type=int, help="分布式worker的数量", default=3)
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='')
    arg_parser.add_argument('--model_path', type=str, help="推理模型路径", default='')
    arg_parser.add_argument('--input_file', type=str, help="推理输入文件", default='')
    arg_parser.add_argument('--output_file', type=str, help="推理输出文件", default='')
    arg_parser.add_argument('--max_new_tokens', type=str, help="最大生成token数", default='512')
    arg_parser.add_argument('--temperature', type=str, help="采样温度", default='0.3')
    arg_parser.add_argument('--top_k', type=str, help="Top-K 采样参数", default='10')
    arg_parser.add_argument('--top_p', type=str, help="Top-P (nucleus) 采样参数", default='0.7')
    arg_parser.add_argument('--backend', type=str, help="推理后端", default='transformers')

    args = arg_parser.parse_args()
    log("{} args: {}".format(__file__, args))

    # worker 镜像 tag 优先级：命令行 --image > 环境变量 LLM_OFFLINE_PREDICT > image_tags.conf（兼容旧镜像）
    _conf_dir = os.path.dirname(os.path.abspath(__file__))
    _worker_image = os.environ.get('LLM_OFFLINE_PREDICT', '')
    if not _worker_image:
        _conf_path = os.path.join(_conf_dir, 'image_tags.conf')
        if os.path.exists(_conf_path):
            with open(_conf_path, 'r') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith('LLM_OFFLINE_PREDICT='):
                        _worker_image = _line.split('=', 1)[1]
                        break
    worker_image = args.image if args.image else _worker_image
    worker_command = args.command if args.command else "python3 /app/predict.py"

    # 清理启动rabbitmq
    create_rabbitmq(name=rabbitmq_name,create=False)
    time.sleep(10)
    create_rabbitmq(name=rabbitmq_name,create=True)
    volcanojob_name = ("volcanojob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4])[0:54].strip('-')
    # 启动volcanojob，并等待结束
    env={
        "RABBIT_HOST":rabbitmq_name,
        "MODEL_PATH": args.model_path,
        "INPUT_FILE": args.input_file,
        "OUTPUT_FILE": args.output_file,
        "MAX_NEW_TOKENS": args.max_new_tokens,
        "TEMPERATURE": args.temperature,
        "TOP_K": args.top_k,
        "TOP_P": args.top_p,
        "BACKEND": args.backend
    }
    launch_volcanojob(name=volcanojob_name,num_workers=args.num_worker,image=worker_image,working_dir=args.working_dir,worker_command=worker_command,env=env)
    # 清理rabbitmq
    create_rabbitmq(name=rabbitmq_name,create=False)
    # 删除volcanojob
    try:
        k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'], namespace=KFJ_NAMESPACE, labels={"run-id": KFJ_RUN_ID})
    except Exception as e:
        print(e)




