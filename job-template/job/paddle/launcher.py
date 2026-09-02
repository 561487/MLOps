
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

from kubernetes import client

# print(os.environ)
from job.pkgs.k8s.py_k8s import K8s
from job.pkgs.k8s.affinity import build_pod_anti_affinity
from job.pkgs.k8s.replica_specs import build_master_worker_replica_specs
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
    "group": "kubeflow.org",
    "version": "v1",
    'kind': 'PaddleJob',
    "plural": "paddlejobs",
    "timeout": 60 * 60 * 24 * 2
}


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)

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

schedulerName = 'volcano'



def default_job_name():
    name = "paddlejob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
    return name[0:54]


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
            print('shell finish %s'%status,flush=True)
            break

    return cmd.returncode






# 监控指定名称的paddlejob
def monitoring(crd_k8s,name,namespace):
    time.sleep(10)
    # 杀掉stern 进程
    def get_pid(name):
        '''
         作用：根据进程名获取进程pid
        '''
        pids = psutil.process_iter()
        print("[" + name + "]'s pid is:", flush=True)
        back=[]
        for pid in pids:
            if name in pid.name():
                print(pid.pid, flush=True)
                back.append(pid.pid)
        return back
    check_time = datetime.datetime.now()
    while(True):
        paddlejob = crd_k8s.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=name)
        if paddlejob:
            print('paddlejob status %s'%paddlejob['status'], flush=True)
            status = paddlejob.get('status','').lower()
            if status!='running':
                # 如果不是running 就打印下详情，好知道为啥不行
                events = crd_k8s.get_job_event(namespace=KFJ_NAMESPACE,job_name=name)
                if events:
                    print(events[-1].get("message",''))
                pass
        else:
            print('paddlejob not exist', flush=True)

        # 如果结束就kill stern，主进程继续
        if paddlejob and (paddlejob['status']=="Succeeded" or paddlejob['status']=="Failed"):    # Created, Running, Restarting, Succeeded, or Failed
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            break
        else:

            labels={
                "pipeline-id": KFJ_PIPELINE_ID,
                "task-id": KFJ_TASK_ID,
                "run-id": KFJ_RUN_ID,
            }
            # pods = crd_k8s.get_pods(namespace=namespace,labels=labels)

        # 因为stern 过几个小时就会崩溃，日志不再跟踪，但不报错，所以这里主动kill
        if (datetime.datetime.now()-check_time).total_seconds()>3600:
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            check_time=datetime.datetime.now()
        time.sleep(60)



# @pysnooper.snoop()
def make_paddlejob(name,num_workers,image,working_dir,command):
    # if type(command)==str:
    #     command=command.split(" ")
    #     command = [c for c in command if c]
    pod_spec={
        "replicas": 1,
        "restartPolicy": "Never",
        "template": {
            "metadata": {
                "labels": {
                    "pipeline-id": KFJ_PIPELINE_ID,
                    "pipeline-name": KFJ_PIPELINE_NAME,
                    "task-id": KFJ_TASK_ID,
                    "task-name": KFJ_TASK_NAME,
                    'username': KFJ_RUNNER,
                    "component": name,
                    "type": "paddlejob",
                    "run-id": KFJ_RUN_ID,
                },
                "annotations": {
                    "project": KFJ_TASK_PROJECT_NAME
                }
            },
            "spec": {
                "schedulerName": schedulerName,
                "restartPolicy": "Never",
                "volumes": k8s_volumes,
                "imagePullSecrets": HUBSECRET,
                "nodeSelector": KFJ_TASK_NODE_SELECTOR,
                "affinity": {
                    "podAntiAffinity": build_pod_anti_affinity(
                        {"component": name, "type": "paddlejob"},
                        num_workers,
                    )
                },
                "containers": [
                    {
                        "name": "paddle",
                        "image": image if image else KFJ_TASK_IMAGES,
                        "imagePullPolicy": os.getenv('IMAGE_PULL_POLICY','IfNotPresent'),
                        "workingDir":working_dir,
                        "env":[
                            {
                                "name": "NCCL_DEBUG",
                                "value":"INFO"
                            },
                            {
                                "name":"GPU_NUM",
                                "value": str(int(gpu_num))
                            }
                        ],
                        "command": ['bash','-c',command],
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
        }
    }


    if int(gpu_num)>=1:
        pod_spec['template']['spec']['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
        pod_spec['template']['spec']['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
        pod_spec['template']['spec']['nodeSelector'].pop('cpu', None)
        pod_spec['template']['spec']['nodeSelector']['gpu'] = 'true'
    elif int(gpu_num)<0:
        shared_count, _, shared_resource_name = k8s_client.get_gpu_shared_resource(GPU_RESOURCE)
        pod_spec['template']['spec']['containers'][0]['resources']['requests'][shared_resource_name] = shared_count
        pod_spec['template']['spec']['containers'][0]['resources']['limits'][shared_resource_name] = shared_count
        pod_spec['template']['spec']['nodeSelector'].pop('cpu', None)
        pod_spec['template']['spec']['nodeSelector']['gpu'] = 'true'
    else:
        # 添加禁用指令
        pod_spec['template']['spec']['containers'][0]['env'].append({
            "name":"NVIDIA_VISIBLE_DEVICES",
            "value":"none"
        })

    if RDMA_RESOURCE_NAME and RDMA_RESOURCE and int(RDMA_RESOURCE):
        pod_spec['template']['spec']['containers'][0]['resources']['requests'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)
        pod_spec['template']['spec']['containers'][0]['resources']['limits'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)

        pod_spec['template']['spec']['containers'][0]['securityContext']={
            "capabilities": {
                "add": [
                    "IPC_LOCK"
                ]
            }
        }


    # num_workers 含 Master；单机时不写 Worker，避免多占一份资源/默认出 worker-0
    paddle_deploy = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PaddleJob",
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
            "backoffLimit":num_workers,
            "cleanPodPolicy": "None",
            "paddleReplicaSpecs": build_master_worker_replica_specs(pod_spec, num_workers),
        }
    }

    return paddle_deploy


# 获取分布式任务每个pod的情况，获取更多状态信息
def get_paddlejob_pod():
    pass

# @pysnooper.snoop()
def launch_paddlejob(name, num_workers, image,working_dir, worker_command):
    if KFJ_RUN_ID:
        print('delete old paddlejob, run-id %s'%KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        time.sleep(10)
    # 删除旧的paddlejob
    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)
    # 创建新的paddlejob
    paddlejob_json = make_paddlejob(name=name,num_workers= num_workers,image = image,working_dir=working_dir,command=worker_command)
    print('create new paddlejob %s' % name, flush=True)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=paddlejob_json)
    time.sleep(10)

    print('begin start monitoring thread', flush=True)
    # # 后台启动监控脚本
    monitoring_thread = threading.Thread(target=monitoring,args=(k8s_client,name,KFJ_NAMESPACE))
    monitoring_thread.start()
    while True:
        # 实时打印日志
        line='>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print('begin follow log\n%s'%line, flush=True)
        command = '''stern %s --namespace %s --since 10s --template '{{.PodName}} {{.Message}} {{"\\n"}}' '''%(name,KFJ_NAMESPACE)

        print(command, flush=True)
        run_shell(command)
        print('%s\nend follow log'%line, flush=True)
        time.sleep(10)

        paddlejob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        if paddlejob and (paddlejob['status'] == "Succeeded" or paddlejob['status'] == "Failed"):
            break

    paddlejob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
    print("paddleJob %s finished, status %s"%(name, paddlejob['status']))

    if paddlejob['status']!='Succeeded':
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Paddlejob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录", default='/mnt/')
    arg_parser.add_argument('--command', type=str, help="运行job的命令", default='python3 train.py')
    arg_parser.add_argument('--num_worker', type=int, help="分布式worker的数量，包含1个master", default=3)
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='')

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))


    launch_paddlejob(name=default_job_name(),num_workers=args.num_worker,image=args.image,working_dir=args.working_dir,worker_command=args.command)


