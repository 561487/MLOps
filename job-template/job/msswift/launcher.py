"""
msswift 模板启动器
采用 PyTorchJob + Volcano Scheduler 方案，运行 ms-swift 分布式训练。
从 pytorch/launcher.py 派生，容器名和命令生成改为 swift sft 专用。
"""

import argparse
import copy
import datetime
import json
import os
import re
import shlex
import sys
import threading
import time
import uuid

import psutil

from kubernetes import client
from job.pkgs.k8s.py_k8s import K8s

k8s_client = K8s()

# ===== 平台环境变量 =====
KFJ_NAMESPACE = os.getenv('KFJ_NAMESPACE', 'pipeline')
KFJ_TASK_ID = os.getenv('KFJ_TASK_ID', '')
KFJ_TASK_NAME = os.getenv('KFJ_TASK_NAME', '')
task_node_selectors = re.split(',|;|\n|\t', os.getenv('KFJ_TASK_NODE_SELECTOR', 'cpu=true,train=true'))
KFJ_TASK_NODE_SELECTOR = {}
for task_node_selector in task_node_selectors:
    if '=' in task_node_selector:
        KFJ_TASK_NODE_SELECTOR[task_node_selector.split('=')[0]] = task_node_selector.split('=')[1]

KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME', 'public')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_CREATOR = os.getenv('KFJ_CREATOR', '')
KFJ_RUNNER = os.getenv('KFJ_RUNNER', '')
KFJ_PIPELINE_NAME = os.getenv('KFJ_PIPELINE_NAME', '')
KFJ_TASK_IMAGES = os.getenv('KFJ_TASK_IMAGES', '')
KFJ_TASK_VOLUME_MOUNT = os.getenv('KFJ_TASK_VOLUME_MOUNT', '')
KFJ_TASK_RESOURCE_CPU = os.getenv('KFJ_TASK_RESOURCE_CPU', '16')
KFJ_TASK_RESOURCE_MEMORY = os.getenv('KFJ_TASK_RESOURCE_MEMORY', '64G')

GPU_RESOURCE_NAME = os.getenv('GPU_RESOURCE_NAME', 'nvidia.com/gpu')
GPU_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_GPU', '1')
gpu_num, gpu_type, _ = k8s_client.get_gpu(GPU_RESOURCE)
if gpu_type:
    KFJ_TASK_NODE_SELECTOR['gpu-type'] = gpu_type
# 如果平台未注入 GPU 数量或为 0，通过 nvidia-smi 自动检测
if int(gpu_num) <= 0:
    import subprocess as _sp
    try:
        result = _sp.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=10)
        gpu_num = len([l for l in result.stdout.strip().split('\n') if l.strip()])
        print(f'[launcher] KFJ_TASK_RESOURCE_GPU={GPU_RESOURCE}, detected {gpu_num} GPUs via nvidia-smi')
    except Exception:
        print('[launcher] WARNING: Cannot detect GPU count, defaulting to 1')
        gpu_num = 1

RDMA_RESOURCE_NAME = os.getenv('RDMA_RESOURCE_NAME', '')
RDMA_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_RDMA', '0')

HUBSECRET = os.getenv('HUBSECRET', 'hubsecret')
HUBSECRET = [{"name": hubsecret} for hubsecret in HUBSECRET.split(',')]

DEFAULT_POD_RESOURCES = os.getenv('DEFAULT_POD_RESOURCES', '')
DEFAULT_POD_RESOURCES = json.loads(DEFAULT_POD_RESOURCES) if DEFAULT_POD_RESOURCES else {}

k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT, KFJ_CREATOR)

SCHEDULER_NAME = 'volcano'

CRD_INFO = {
    "group": "kubeflow.org",
    "version": "v1",
    "kind": "PyTorchJob",
    "plural": "pytorchjobs",
    "timeout": 60 * 60 * 24 * 2
}

MONITOR_PORT = int(os.getenv("MLOPS_MONITOR_MASTER_PORT", "29501"))
MONITOR_TOKEN = os.getenv("MLOPS_MONITOR_TOKEN", "") or uuid.uuid4().hex
MONITOR_ENV_KEYS = (
    "MLOPS_TRAINING_MONITOR_ENABLE",
    "MLOPS_TRAINING_MONITOR_TYPE",
    "MLOPS_MONITOR_REGISTER_URL",
    "MLOPS_PIPELINE_RUN_ID",
    "MLOPS_PIPELINE_ID",
    "MLOPS_WORKFLOW_NAME",
    "MLOPS_TASK_ID",
    "MLOPS_TASK_NAME",
    "MLOPS_NODE_NAME",
    "MLOPS_JOB_TEMPLATE_NAME",
    "SWANLAB_MODE",
    "SWANLAB_API_HOST",
    "SWANLAB_WEB_HOST",
    "SWANLAB_PROJ_NAME",
    "SWANLAB_WORKSPACE",
    "SWANLAB_EXP_NAME",
    "SWANLAB_GROUP",
    "SWANLAB_TAGS",
    "SWANLAB_PROBE_MONITOR_INTERVAL",
    "SWANLAB_METRIC_LEVEL",
)


def _monitor_enabled():
    return os.getenv("MLOPS_TRAINING_MONITOR_ENABLE", "").lower() == "true"


def default_job_name():
    name = "msswift-" + KFJ_PIPELINE_NAME.replace('_', '-') + "-" + uuid.uuid4().hex[:4]
    return name[0:54]


def run_shell(shell):
    """实时输出 shell 执行日志"""
    import subprocess
    print('begin run shell: %s' % shell, flush=True)
    cmd = subprocess.Popen(shell, stdin=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdout=subprocess.PIPE, universal_newlines=True, shell=True, bufsize=1)
    while True:
        line = cmd.stdout.readline()
        status = subprocess.Popen.poll(cmd)
        if status is not None:
            print(line, end='', flush=True)
            print('shell finish %s' % status, flush=True)
            break
        print(line, end='', flush=True)
    return cmd.returncode


def monitoring(crd_k8s, name, namespace):
    """后台监控 PyTorchJob 状态 + stern 日志跟踪"""
    time.sleep(10)

    def get_pid(pname):
        pids = psutil.process_iter()
        back = []
        for pid in pids:
            if pname in pid.name():
                back.append(pid.pid)
        return back

    check_time = datetime.datetime.now()
    while True:
        pytorchjob = crd_k8s.get_one_crd(
            group=CRD_INFO['group'], version=CRD_INFO['version'],
            plural=CRD_INFO['plural'], namespace=namespace, name=name)
        if pytorchjob:
            print('pytorchjob status: %s' % pytorchjob.get('status', ''), flush=True)
            status = pytorchjob.get('status', '').lower()
            if status != 'running':
                events = crd_k8s.get_job_event(namespace=namespace, job_name=name)
                if events:
                    print('event: %s' % events[-1].get("message", ''), flush=True)

        if pytorchjob and pytorchjob['status'] in ("Succeeded", "Failed"):
            pids = get_pid("stern")
            for pid in pids:
                psutil.Process(int(pid)).terminate()
            break

        # stern 可能几小时后崩溃，主动重建
        if (datetime.datetime.now() - check_time).total_seconds() > 3600:
            pids = get_pid("stern")
            for pid in pids:
                psutil.Process(int(pid)).terminate()
            check_time = datetime.datetime.now()
        time.sleep(60)


def create_monitor_service(name):
    """Expose the master metric aggregator to Worker Pods."""
    if not _monitor_enabled():
        return
    service_name = name + "-monitor"
    api = client.CoreV1Api()
    try:
        api.delete_namespaced_service(
            name=service_name,
            namespace=KFJ_NAMESPACE,
            body=client.V1DeleteOptions())
    except Exception:
        pass
    body = client.V1Service(
        metadata=client.V1ObjectMeta(
            name=service_name,
            namespace=KFJ_NAMESPACE,
            labels={"component": name, "monitor": "msswift"}),
        spec=client.V1ServiceSpec(
            selector={
                "training.kubeflow.org/job-name": name,
                "training.kubeflow.org/replica-type": "master",
            },
            ports=[client.V1ServicePort(
                name="monitor",
                port=MONITOR_PORT,
                target_port=MONITOR_PORT)]))
    try:
        api.create_namespaced_service(namespace=KFJ_NAMESPACE, body=body)
        print("created monitor service: %s:%d" %
              (service_name, MONITOR_PORT), flush=True)
    except Exception as error:
        # Monitoring is fail-open; training submission must continue.
        print("WARNING: cannot create monitor service %s: %s" %
              (service_name, error), flush=True)


def delete_monitor_service(name):
    if not _monitor_enabled():
        return
    try:
        client.CoreV1Api().delete_namespaced_service(
            name=name + "-monitor",
            namespace=KFJ_NAMESPACE,
            body=client.V1DeleteOptions())
    except Exception:
        pass


def make_pytorchjob(name, num_workers, image, command):
    """组装 PyTorchJob CRD，从 pytorch 模板复刻"""
    monitor_env = []
    if _monitor_enabled():
        for key in MONITOR_ENV_KEYS:
            value = os.environ.get(key)
            if value is not None and value != "":
                monitor_env.append({"name": key, "value": str(value)})
        monitor_env.extend([
            {"name": "MLOPS_DISTRIBUTED_HARDWARE_ENABLE", "value": "true"},
            {"name": "MLOPS_MONITOR_MASTER_ADDR", "value": name + "-monitor"},
            {"name": "MLOPS_MONITOR_MASTER_PORT", "value": str(MONITOR_PORT)},
            {"name": "MLOPS_MONITOR_EXPECTED_NODES", "value": str(num_workers)},
            {"name": "MLOPS_MONITOR_TOKEN", "value": MONITOR_TOKEN},
            {"name": "K8S_POD_NAME", "valueFrom": {
                "fieldRef": {"fieldPath": "metadata.name"}}},
        ])
        if os.getenv("SWANLAB_MODE", "").lower() in ("cloud", "online"):
            monitor_env.append({
                "name": "SWANLAB_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "swanlab-secret",
                        "key": "SWANLAB_API_KEY",
                    }
                }
            })
        command = (
            "exec python3 /mnt/swift_training_wrapper.py -- bash -lc %s" %
            shlex.quote(command)
        )

    pod_spec = {
        "replicas": 1,
        "restartPolicy": "Never",
        "template": {
            "metadata": {
                "labels": {
                    "pipeline-id": KFJ_PIPELINE_ID,
                    "pipeline-name": KFJ_PIPELINE_NAME,
                    "task-id": KFJ_TASK_ID,
                    "task-name": KFJ_TASK_NAME,
                    "username": KFJ_RUNNER,
                    "component": name,
                    "type": "pytorchjob",
                    "run-id": KFJ_RUN_ID,
                },
                "annotations": {"project": KFJ_TASK_PROJECT_NAME}
            },
            "spec": {
                "schedulerName": SCHEDULER_NAME,
                "restartPolicy": "Never",
                "volumes": k8s_volumes,
                "imagePullSecrets": HUBSECRET,
                "nodeSelector": KFJ_TASK_NODE_SELECTOR,
                "affinity": {
                    "podAntiAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [{
                            "weight": 5,
                            "podAffinityTerm": {
                                "topologyKey": "kubernetes.io/hostname",
                                "labelSelector": {
                                    "matchLabels": {
                                        "component": name,
                                        "type": "pytorchjob"
                                    }
                                }
                            }
                        }]
                    }
                },
                "containers": [{
                    "name": "pytorch",
                    "image": image if image else KFJ_TASK_IMAGES,
                    "imagePullPolicy": os.getenv('IMAGE_PULL_POLICY', 'IfNotPresent'),
                    "workingDir": "/mnt",
                    "env": [
                        {"name": "NCCL_DEBUG", "value": "INFO"},
                        {"name": "NCCL_SOCKET_IFNAME", "value": os.getenv("NCCL_SOCKET_IFNAME", "eth0")},
                        {"name": "MODELSCOPE_CACHE", "value": "/mnt/%s/.cache/modelscope" % KFJ_CREATOR},
                        {"name": "GPU_NUM", "value": str(int(gpu_num))},
                    ] + monitor_env,
                    "ports": [{"name": "swift-monitor", "containerPort": MONITOR_PORT}],
                    "command": ["bash", "-c", command],
                    "volumeMounts": k8s_volume_mounts,
                    "resources": {
                        "requests": {
                            **{"cpu": KFJ_TASK_RESOURCE_CPU, "memory": KFJ_TASK_RESOURCE_MEMORY},
                            **DEFAULT_POD_RESOURCES
                        },
                        "limits": {
                            **{"cpu": KFJ_TASK_RESOURCE_CPU, "memory": KFJ_TASK_RESOURCE_MEMORY},
                            **DEFAULT_POD_RESOURCES
                        }
                    }
                }]
            }
        }
    }

    if int(gpu_num) >= 1:
        pod_spec['template']['spec']['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
        pod_spec['template']['spec']['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
        pod_spec['template']['spec']['nodeSelector'].pop('cpu', None)
        pod_spec['template']['spec']['nodeSelector']['gpu'] = 'true'
        pod_spec['template']['spec']['nodeSelector']['mps'] = 'false'

    if RDMA_RESOURCE_NAME and RDMA_RESOURCE and int(RDMA_RESOURCE):
        pod_spec['template']['spec']['containers'][0]['resources']['requests'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)
        pod_spec['template']['spec']['containers'][0]['resources']['limits'][RDMA_RESOURCE_NAME] = int(RDMA_RESOURCE)
        pod_spec['template']['spec']['containers'][0]['securityContext'] = {
            "capabilities": {"add": ["IPC_LOCK"]}
        }

    worker_pod_spec = copy.deepcopy(pod_spec)
    worker_pod_spec['replicas'] = int(num_workers) - 1

    if _monitor_enabled():
        pod_spec['template']['spec']['containers'][0]['env'].append({
            "name": "MLOPS_MONITOR_ROLE", "value": "primary"
        })
        worker_pod_spec['template']['spec']['containers'][0]['env'].append({
            "name": "MLOPS_MONITOR_ROLE", "value": "worker"
        })

    pytorch_deploy = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels": {
                "run-id": KFJ_RUN_ID,
                "run-username": KFJ_RUNNER,
                "pipeline-username": KFJ_CREATOR,
                "pipeline-id": KFJ_PIPELINE_ID,
                "pipeline-name": KFJ_PIPELINE_NAME,
                "task-id": KFJ_TASK_ID,
                "task-name": KFJ_TASK_NAME,
            },
            "annotations": {"project": KFJ_TASK_PROJECT_NAME}
        },
        "spec": {
            "cleanPodPolicy": "None",
            "pytorchReplicaSpecs": {
                "Master": pod_spec,
                "Worker": worker_pod_spec
            }
        }
    }

    return pytorch_deploy


def launch_pytorchjob(name, num_workers, image, command):
    """提交 PyTorchJob 并等待完成"""
    if KFJ_RUN_ID:
        print('delete old pytorchjob, run-id: %s' % KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=CRD_INFO['group'], version=CRD_INFO['version'],
                              plural=CRD_INFO['plural'], namespace=KFJ_NAMESPACE,
                              labels={"run-id": KFJ_RUN_ID})
        time.sleep(10)
    k8s_client.delete_crd(group=CRD_INFO['group'], version=CRD_INFO['version'],
                          plural=CRD_INFO['plural'], namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)

    pytorchjob_json = make_pytorchjob(name=name, num_workers=num_workers, image=image, command=command)
    print('creating pytorchjob: %s' % name, flush=True)
    print(json.dumps(pytorchjob_json, indent=2, ensure_ascii=False), flush=True)
    k8s_client.create_crd(group=CRD_INFO['group'], version=CRD_INFO['version'],
                          plural=CRD_INFO['plural'], namespace=KFJ_NAMESPACE, body=pytorchjob_json)
    create_monitor_service(name)
    time.sleep(10)

    print('begin monitoring pytorchjob', flush=True)
    monitoring_thread = threading.Thread(target=monitoring, args=(k8s_client, name, KFJ_NAMESPACE))
    monitoring_thread.start()

    while True:
        pods = k8s_client.get_pods(
            namespace=KFJ_NAMESPACE, labels={"component": name, "type": "pytorchjob"})
        for pod in pods:
            try:
                logs = k8s_client.download_pod_log(
                    name=pod["name"], namespace=KFJ_NAMESPACE,
                    container="pytorch", since_seconds=20)
                if logs:
                    print("[%s]\n%s" % (pod["name"], logs), flush=True)
            except Exception as exc:
                print("pod log not ready (%s): %s" % (pod["name"], exc), flush=True)
        time.sleep(10)

        pytorchjob = k8s_client.get_one_crd(
            group=CRD_INFO['group'], version=CRD_INFO['version'],
            plural=CRD_INFO['plural'], namespace=KFJ_NAMESPACE, name=name)
        if pytorchjob and pytorchjob['status'] in ("Succeeded", "Failed"):
            break

    pytorchjob = k8s_client.get_one_crd(
        group=CRD_INFO['group'], version=CRD_INFO['version'],
        plural=CRD_INFO['plural'], namespace=KFJ_NAMESPACE, name=name)
    print("pytorchjob %s finished, status: %s" % (name, pytorchjob.get('status', 'unknown')))
    delete_monitor_service(name)

    if pytorchjob.get('status') != 'Succeeded':
        exit(1)


def shell_command(argv, env=None):
    """生成可直接交给 bash -c 的安全命令。"""
    exports = ["export %s=%s" % item for item in (env or {}).items()]
    command = " \\\n  ".join(shlex.quote(str(value)) for value in argv)
    command = command.replace("'--node_rank=${RANK:-0}'", "--node_rank=${RANK:-0}")
    command = command.replace("'--master_addr=${MASTER_ADDR}'", "--master_addr=${MASTER_ADDR}")
    return "\n".join(exports + ["exec " + command])


def append_grpo_argv(argv, args):
    """Append GRPO-only arguments without affecting DPO/SFT."""
    argv.extend(["--num_generations", str(args.num_generations),
                 "--num_iterations", str(args.num_iterations),
                 "--log_completions", args.log_completions])
    reward_funcs = [value.strip() for value in args.reward_funcs.split(',') if value.strip()]
    if reward_funcs:
        argv.append("--reward_funcs")
        argv.extend(reward_funcs)
    reward_weights = [value.strip() for value in args.reward_weights.split(',') if value.strip()]
    if reward_weights:
        argv.append("--reward_weights")
        argv.extend(reward_weights)
    if args.max_completion_length > 0:
        argv.extend(["--max_completion_length", str(args.max_completion_length)])
    if args.enable_thinking != 'auto':
        argv.extend(["--enable_thinking", args.enable_thinking])
    if args.temperature >= 0:
        argv.extend(["--temperature", str(args.temperature)])
    if args.top_p >= 0:
        argv.extend(["--top_p", str(args.top_p)])
    if args.generation_batch_size > 0:
        argv.extend(["--generation_batch_size", str(args.generation_batch_size)])
    if args.steps_per_generation > 0:
        argv.extend(["--steps_per_generation", str(args.steps_per_generation)])
    return argv


def append_hf_enterprise_argv(argv, args):
    """Append opt-in HF options. Inert defaults preserve the V3 command."""
    if args.gradient_accumulation_steps > 0:
        argv.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    if args.max_steps > 0:
        argv.extend(["--max_steps", str(args.max_steps)])
    if args.val_dataset:
        argv.extend(["--val_dataset", args.val_dataset])
    if args.split_dataset_ratio >= 0:
        argv.extend(["--split_dataset_ratio", str(args.split_dataset_ratio)])
    if args.logging_steps > 0:
        argv.extend(["--logging_steps", str(args.logging_steps)])
    if args.eval_steps > 0:
        argv.extend(["--eval_steps", str(args.eval_steps)])
    if args.save_steps > 0:
        argv.extend(["--save_steps", str(args.save_steps)])
    if args.save_total_limit > 0:
        argv.extend(["--save_total_limit", str(args.save_total_limit)])
    if args.resume_from_checkpoint:
        argv.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    if args.seed >= 0:
        argv.extend(["--seed", str(args.seed)])
    if args.warmup_ratio >= 0:
        argv.extend(["--warmup_ratio", str(args.warmup_ratio)])
    if args.weight_decay >= 0:
        argv.extend(["--weight_decay", str(args.weight_decay)])
    return argv


def append_hf_rlhf_argv(argv, args):
    """Append HF/DeepSpeed RLHF controls only when explicitly configured."""
    if args.ref_model:
        argv.extend(["--ref_model", args.ref_model])
    if args.ref_adapters:
        argv.extend(["--ref_adapters", args.ref_adapters])
    if args.beta >= 0:
        argv.extend(["--beta", str(args.beta)])
    append_rlhf_algorithm_argv(argv, args)
    return argv


def append_megatron_rlhf_argv(argv, args):
    """Map generic form fields to Megatron-SWIFT 3.12 argument names."""
    if args.ref_model:
        argv.extend(["--ref_load", args.ref_model])
    if args.ref_adapters:
        argv.extend(["--ref_adapter_load", args.ref_adapters])
    if args.beta >= 0:
        argv.extend(["--beta", str(args.beta)])
    append_rlhf_algorithm_argv(argv, args)
    return argv


def append_rlhf_algorithm_argv(argv, args):
    if args.loss_type:
        argv.extend(["--loss_type", args.loss_type])
    if args.label_smoothing >= 0:
        argv.extend(["--label_smoothing", str(args.label_smoothing)])
    return argv


def hf_training_argv(args, subcommand):
    """Build the unchanged Phase-1 ms-swift/Hugging Face command."""
    nproc = args.nproc_per_node or int(gpu_num)
    env = {"MODELSCOPE_CACHE": "/mnt/%s/.cache/modelscope" % KFJ_CREATOR}
    env.update({"NNODES": str(args.num_worker), "NPROC_PER_NODE": str(nproc),
                "NODE_RANK": "${RANK:-0}", "MASTER_ADDR": "${MASTER_ADDR}",
                "MASTER_PORT": str(args.master_port)})
    argv = ["swift", subcommand]
    tuner_type = 'lora' if args.train_type == 'qlora' else args.train_type
    argv.extend(["--model", args.model, "--dataset", args.dataset,
                 "--output_dir", args.save_path, "--train_type", tuner_type,
                 "--num_train_epochs", str(args.num_epochs), "--learning_rate", str(args.learning_rate),
                 "--max_length", str(args.max_length), "--torch_dtype", "bfloat16",
                 "--gradient_checkpointing", "true", "--save_only_model", args.save_only_model])
    argv.extend(["--per_device_train_batch_size", str(args.batch_size),
                 "--attn_impl", "flash_attn"])
    if args.distributed in ('zero2', 'zero3'):
        argv.extend(["--deepspeed", args.distributed])
    if args.train_type in ('lora', 'qlora'):
        argv.extend(["--lora_rank", str(args.lora_rank)])
        if args.lora_alpha > 0:
            argv.extend(["--lora_alpha", str(args.lora_alpha)])
        if args.lora_dropout >= 0:
            argv.extend(["--lora_dropout", str(args.lora_dropout)])
    if args.train_type == 'qlora':
        argv.extend(["--quant_method", "bnb", "--quant_bits", "4"])
    append_hf_enterprise_argv(argv, args)
    if subcommand == 'rlhf':
        argv.extend(["--rlhf_type", args.rlhf_type])
        append_hf_rlhf_argv(argv, args)
        if args.rlhf_type == 'grpo':
            append_grpo_argv(argv, args)
    return argv, env


def megatron_training_argv(args, subcommand):
    """Build the native Megatron-SWIFT 3.12.x command."""
    nproc = args.nproc_per_node or int(gpu_num)
    world_size = args.num_worker * nproc
    global_batch_size = args.global_batch_size or (args.batch_size * world_size)
    env = {
        "MODELSCOPE_CACHE": "/mnt/%s/.cache/modelscope" % KFJ_CREATOR,
        "NNODES": str(args.num_worker), "NPROC_PER_NODE": str(nproc),
        "NODE_RANK": "${RANK:-0}", "MASTER_ADDR": "${MASTER_ADDR}",
        "MASTER_PORT": str(args.master_port), "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    argv = [
        "megatron", subcommand,
        "--model", args.model, "--dataset", args.dataset,
        "--save", args.save_path, "--train_type", args.train_type,
        "--tensor_model_parallel_size", str(args.tensor_model_parallel_size),
        "--pipeline_model_parallel_size", str(args.pipeline_model_parallel_size),
        "--micro_batch_size", str(args.batch_size),
        "--global_batch_size", str(global_batch_size),
        "--train_iters", str(args.train_iters), "--lr", str(args.learning_rate),
        "--max_length", str(args.max_length), "--bf16", "true",
        "--recompute_granularity", args.recompute_granularity,
        "--use_flash_attn", "true",
    ]
    if args.save_only_model == 'true':
        argv.extend(["--no_save_optim", "true", "--no_save_rng", "true"])
    if args.train_type == 'lora':
        argv.extend(["--lora_rank", str(args.lora_rank)])
        if args.lora_alpha > 0:
            argv.extend(["--lora_alpha", str(args.lora_alpha)])
        if args.lora_dropout >= 0:
            argv.extend(["--lora_dropout", str(args.lora_dropout)])
    if args.val_dataset:
        argv.extend(["--val_dataset", args.val_dataset])
    if args.split_dataset_ratio >= 0:
        argv.extend(["--split_dataset_ratio", str(args.split_dataset_ratio)])
    if args.logging_steps > 0:
        argv.extend(["--log_interval", str(args.logging_steps)])
    if args.eval_steps > 0:
        argv.extend(["--eval_interval", str(args.eval_steps)])
    if args.save_steps > 0:
        argv.extend(["--save_interval", str(args.save_steps)])
    if args.resume_from_checkpoint:
        if args.train_type == 'lora':
            argv.extend(["--adapter_load", args.resume_from_checkpoint, "--finetune", "false"])
        else:
            argv.extend(["--load", args.resume_from_checkpoint])
    if args.seed >= 0:
        argv.extend(["--seed", str(args.seed)])
    if args.weight_decay >= 0:
        argv.extend(["--weight_decay", str(args.weight_decay)])
    if subcommand == 'rlhf':
        argv.extend(["--rlhf_type", args.rlhf_type])
        append_megatron_rlhf_argv(argv, args)
        if args.rlhf_type == 'grpo':
            append_grpo_argv(argv, args)
    return argv, env


def training_argv(args, subcommand):
    if args.distributed == 'megatron':
        return megatron_training_argv(args, subcommand)
    return hf_training_argv(args, subcommand)


def build_sft_command(args):
    argv, env = training_argv(args, 'sft')
    return shell_command(argv, env)


def build_rlhf_command(args):
    argv, env = training_argv(args, 'rlhf')
    return shell_command(argv, env)


def validate_args(args):
    """参数校验"""
    errors = []

    if args.num_worker < 1:
        errors.append("--num_worker must >= 1, got %d" % args.num_worker)

    nproc = args.nproc_per_node or int(gpu_num)
    if nproc < 1 or nproc > int(gpu_num):
        errors.append("--nproc_per_node (%d) must be 1 ~ %d" % (nproc, int(gpu_num)))

    if args.distributed not in ('none', 'zero2', 'zero3', 'megatron'):
        errors.append("--distributed must be none/zero2/zero3/megatron, got '%s'" % args.distributed)

    if args.train_type not in ('lora', 'full', 'qlora'):
        errors.append("--train_type must be lora/full/qlora, got '%s'" % args.train_type)

    if args.train_type == 'full' and args.distributed == 'none' and int(gpu_num) == 1:
        print("WARNING: 全参微调 + 单卡 + 无 DeepSpeed 可能 OOM，建议 --distributed zero3 或增加 GPU")

    if args.mode not in ('sft', 'rlhf'):
        errors.append("--mode must be sft/rlhf, got '%s'" % args.mode)

    if args.mode == 'rlhf' and args.rlhf_type not in ('dpo', 'grpo', 'ppo'):
        errors.append("--rlhf_type must be dpo/grpo/ppo, got '%s'" % args.rlhf_type)
    if args.mode == 'rlhf' and args.rlhf_type == 'ppo':
        errors.append("PPO is not available in the enterprise form yet: reward/value model configuration is required")

    if not 1 <= args.master_port <= 65535:
        errors.append("--master_port must be between 1 and 65535")
    if args.batch_size < 1:
        errors.append("--batch_size must >= 1")
    if args.gradient_accumulation_steps < 0:
        errors.append("--gradient_accumulation_steps must >= 0 (0=framework auto)")
    if args.num_epochs <= 0:
        errors.append("--num_epochs must > 0")
    if args.max_steps < 0:
        errors.append("--max_steps must >= 0")
    if args.learning_rate <= 0:
        errors.append("--learning_rate must > 0")
    if args.max_length < 1:
        errors.append("--max_length must >= 1")
    if args.split_dataset_ratio != -1 and not 0 <= args.split_dataset_ratio < 1:
        errors.append("--split_dataset_ratio must be -1 or in [0, 1)")
    if args.warmup_ratio != -1 and not 0 <= args.warmup_ratio <= 1:
        errors.append("--warmup_ratio must be -1 or in [0, 1]")
    if args.weight_decay < -1:
        errors.append("--weight_decay must >= -1")
    for option, value in (("--logging_steps", args.logging_steps),
                          ("--eval_steps", args.eval_steps),
                          ("--save_steps", args.save_steps),
                          ("--save_total_limit", args.save_total_limit)):
        if value < 0:
            errors.append("%s must >= 0" % option)
    if args.train_type in ('lora', 'qlora'):
        if args.lora_rank < 1:
            errors.append("--lora_rank must >= 1")
        if args.lora_alpha < 0:
            errors.append("--lora_alpha must >= 0")
        if args.lora_dropout != -1 and not 0 <= args.lora_dropout < 1:
            errors.append("--lora_dropout must be -1 or in [0, 1)")
    if args.resume_from_checkpoint and args.save_only_model == 'true':
        print("WARNING: save_only_model=true restores weights only; use false for optimizer/RNG state")
    if args.beta < -1:
        errors.append("--beta must >= -1 (-1=framework default)")
    if args.label_smoothing != -1 and not 0 <= args.label_smoothing <= 1:
        errors.append("--label_smoothing must be -1 or in [0, 1]")

    world_size = args.num_worker * nproc
    if args.mode == 'rlhf' and args.rlhf_type == 'grpo':
        grad_acc = args.gradient_accumulation_steps or 1
        training_global_batch = args.batch_size * world_size * grad_acc
        if args.distributed == 'megatron':
            training_global_batch = args.global_batch_size or (args.batch_size * world_size)
        if args.generation_batch_size > 0:
            generation_batch_size = args.generation_batch_size
        elif args.steps_per_generation > 0:
            generation_batch_size = args.steps_per_generation * args.batch_size * world_size
            if args.distributed == 'megatron':
                generation_batch_size = args.steps_per_generation * training_global_batch
        else:
            generation_batch_size = training_global_batch
        if args.num_generations < 2:
            errors.append("--num_generations must >= 2 for GRPO")
        elif generation_batch_size % args.num_generations != 0:
            errors.append("GRPO generation batch size (%d) must be divisible by num_generations (%d)" %
                          (generation_batch_size, args.num_generations))
        if not args.reward_funcs.strip():
            errors.append("--reward_funcs is required for GRPO when no reward model is configured")
        weights = [value.strip() for value in args.reward_weights.split(',') if value.strip()]
        funcs = [value.strip() for value in args.reward_funcs.split(',') if value.strip()]
        supported_reward_funcs = {'accuracy', 'format', 'cosine', 'repetition', 'soft_overlong'}
        unknown_funcs = sorted(set(funcs) - supported_reward_funcs)
        if unknown_funcs:
            errors.append("unsupported --reward_funcs: %s" % ','.join(unknown_funcs))
        if len(funcs) != len(set(funcs)):
            errors.append("--reward_funcs must not contain duplicates")
        if weights and len(weights) != len(funcs):
            errors.append("--reward_weights count must match --reward_funcs count")
        if weights:
            try:
                parsed_weights = [float(value) for value in weights]
                if any(value < 0 for value in parsed_weights):
                    errors.append("--reward_weights values must be >= 0")
                if parsed_weights and sum(parsed_weights) <= 0:
                    errors.append("--reward_weights must contain at least one positive value")
            except ValueError:
                errors.append("--reward_weights must be comma-separated numbers")
        if args.num_iterations < 1:
            errors.append("--num_iterations must >= 1 for GRPO")
        if args.generation_batch_size < 0:
            errors.append("--generation_batch_size must >= 0")
        if args.steps_per_generation < 0:
            errors.append("--steps_per_generation must >= 0")
        if args.generation_batch_size > 0 and args.steps_per_generation > 0:
            errors.append("--generation_batch_size and --steps_per_generation are mutually exclusive")
        if args.generation_batch_size > 0 and args.generation_batch_size % training_global_batch != 0:
            errors.append("generation_batch_size (%d) must be a multiple of training global batch (%d)" %
                          (args.generation_batch_size, training_global_batch))
        if (args.distributed != 'megatron' and args.steps_per_generation > 0
                and args.steps_per_generation % grad_acc != 0):
            errors.append("steps_per_generation (%d) must be a multiple of gradient accumulation (%d)" %
                          (args.steps_per_generation, grad_acc))
        if args.max_completion_length < 0:
            errors.append("--max_completion_length must >= 0")
        if args.temperature < -1:
            errors.append("--temperature must >= -1 (-1=framework default)")
        if args.top_p < -1 or args.top_p > 1:
            errors.append("--top_p must be -1 or in [0, 1]")
        if args.max_completion_length and args.max_completion_length > args.max_length:
            print("WARNING: max_completion_length exceeds max_length; verify the model context window")

    if args.distributed == 'megatron':
        model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        if args.train_type == 'qlora':
            errors.append("--train_type qlora is not supported with --distributed megatron")
        if model_parallel_size > world_size or world_size % model_parallel_size != 0:
            errors.append("TP * PP (%d) must divide world size (%d)" % (model_parallel_size, world_size))
        data_parallel_size = world_size // model_parallel_size if model_parallel_size else 0
        if args.train_iters < 1:
            errors.append("--train_iters must >= 1 with --distributed megatron")
        if args.global_batch_size < 0:
            errors.append("--global_batch_size must >= 0 (0=auto)")
        effective_global_batch = args.global_batch_size or (args.batch_size * world_size)
        divisor = args.batch_size * data_parallel_size
        if divisor and effective_global_batch % divisor != 0:
            errors.append("global batch size (%d) must be divisible by micro batch size * DP (%d)" %
                          (effective_global_batch, divisor))
        if args.max_steps > 0:
            errors.append("--max_steps is for HF/DeepSpeed; use --train_iters with Megatron")
        if args.gradient_accumulation_steps > 0:
            errors.append("--gradient_accumulation_steps is derived from global_batch_size in Megatron")
        if args.warmup_ratio >= 0:
            errors.append("--warmup_ratio is not exposed for Megatron yet; leave it at -1")
        if args.save_total_limit > 0:
            errors.append("--save_total_limit is not supported by the Megatron launcher")
    if not args.model:
        errors.append("--model is required")
    if not args.dataset:
        errors.append("--dataset is required")
    if not args.save_path:
        errors.append("--save_path is required")

    if errors:
        print("Parameter validation failed:")
        for e in errors:
            print("  - %s" % e)
        sys.exit(1)

    print("Parameter validation passed. GPU count: %d, nproc_per_node: %d" % (int(gpu_num), nproc))


def arg_parser():
    parser = argparse.ArgumentParser(description='msswift distributed training launcher')

    # 公共参数
    parser.add_argument('--mode', type=str, default='sft', choices=['sft', 'rlhf'],
                        help='Task mode: sft / rlhf')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                        help='ModelScope model name or PVC path')
    parser.add_argument('--dataset', type=str, required=True,
                        help='ModelScope dataset name or local JSONL/PVC path')
    parser.add_argument('--val_dataset', type=str, default='',
                        help='Optional validation dataset name or path')
    parser.add_argument('--split_dataset_ratio', type=float, default=-1,
                        help='Validation split ratio; -1 preserves the framework default')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Output PVC path')

    # 分布式配置
    parser.add_argument('--num_worker', type=int, default=1, help='Number of PyTorchJob pods')
    parser.add_argument('--nproc_per_node', type=int, default=0, help='Processes per pod (default: GPU count)')
    parser.add_argument('--master_port', type=int, default=23456, help='Distributed communication port')
    parser.add_argument('--distributed', type=str, default='zero2',
                        choices=['none', 'zero2', 'zero3', 'megatron'],
                        help='Distributed strategy')
    parser.add_argument('--tensor_model_parallel_size', type=int, default=1, help='TP size (megatron only)')
    parser.add_argument('--pipeline_model_parallel_size', type=int, default=1, help='PP size (megatron only)')

    # 训练配置
    parser.add_argument('--train_type', type=str, default='lora', choices=['lora', 'full', 'qlora'])
    parser.add_argument('--num_epochs', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=1, help='Per-device batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=0,
                        help='HF maximum training steps; 0 uses num_epochs')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed; -1 preserves the framework default')
    parser.add_argument('--warmup_ratio', type=float, default=-1,
                        help='Warmup ratio; -1 preserves the framework default')
    parser.add_argument('--weight_decay', type=float, default=-1,
                        help='Weight decay; -1 preserves the framework default')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (lora/qlora only)')
    parser.add_argument('--lora_alpha', type=int, default=0,
                        help='LoRA alpha (integer); 0 preserves the framework default')
    parser.add_argument('--lora_dropout', type=float, default=-1,
                        help='LoRA dropout; -1 preserves the framework default')

    # Evaluation, checkpoint and observability options. Zero/empty keeps V3 behavior.
    parser.add_argument('--logging_steps', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--save_total_limit', type=int, default=0)
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    parser.add_argument('--save_only_model', type=str, default='true', choices=['true', 'false'])

    # Phase-2 Megatron-only options. Phase-1 ignores these values.
    parser.add_argument('--train_iters', type=int, default=100, help='Megatron training iterations')
    parser.add_argument('--global_batch_size', type=int, default=0,
                        help='Megatron global batch size (0=auto)')
    parser.add_argument('--recompute_granularity', type=str, default='selective',
                        choices=['selective', 'full', 'none'],
                        help='Megatron activation recompute strategy')

    # RLHF 专属. PPO remains parseable for old workflows but is rejected with a clear error.
    parser.add_argument('--rlhf_type', type=str, default='dpo', choices=['dpo', 'grpo', 'ppo', 'kto'])
    parser.add_argument('--ref_model', type=str, default='')
    parser.add_argument('--ref_adapters', type=str, default='')
    parser.add_argument('--beta', type=float, default=-1,
                        help='RLHF beta; -1 preserves the framework/algorithm default')
    parser.add_argument('--loss_type', type=str, default='')
    parser.add_argument('--label_smoothing', type=float, default=-1)
    # KTO fields are accepted for form compatibility. RLHF+KTO remains
    # rejected by validate_args until the algorithm is implemented.
    parser.add_argument('--desirable_weight', type=float, default=1.0)
    parser.add_argument('--undesirable_weight', type=float, default=1.0)
    parser.add_argument('--num_generations', type=int, default=2,
                        help='GRPO completions per prompt')
    parser.add_argument('--reward_funcs', type=str, default='format,repetition',
                        help='Comma-separated GRPO reward functions')
    parser.add_argument('--reward_weights', type=str, default='',
                        help='Comma-separated GRPO reward weights')
    parser.add_argument('--num_iterations', type=int, default=1,
                        help='GRPO update iterations per generation batch')
    parser.add_argument('--log_completions', type=str, default='false', choices=['true', 'false'],
                        help='Log GRPO generated completions')
    parser.add_argument('--max_completion_length', type=int, default=0,
                        help='GRPO completion length; 0 preserves the framework default')
    parser.add_argument('--enable_thinking', type=str, default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Qwen thinking mode; auto preserves the template default')
    parser.add_argument('--temperature', type=float, default=-1,
                        help='GRPO sampling temperature; 0 preserves the framework default')
    parser.add_argument('--top_p', type=float, default=-1,
                        help='GRPO top-p; -1 preserves the framework default')
    parser.add_argument('--generation_batch_size', type=int, default=0,
                        help='GRPO rollout batch; 0 derives from training batch')
    parser.add_argument('--steps_per_generation', type=int, default=0,
                        help='GRPO optimization steps per rollout; 0 uses framework default')

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    # 默认 nproc_per_node = GPU 数量
    if not args.nproc_per_node:
        args.nproc_per_node = int(gpu_num)

    print("=" * 60)
    print("msswift launcher")
    print("=" * 60)
    print("  mode:       %s" % args.mode)
    print("  model:      %s" % args.model)
    print("  dataset:    %s" % args.dataset)
    print("  save_path:  %s" % args.save_path)
    print("  num_worker: %d" % args.num_worker)
    print("  nproc:      %d" % args.nproc_per_node)
    print("  distributed:%s" % args.distributed)
    print("  train_type: %s" % args.train_type)
    print("=" * 60)

    validate_args(args)

    # 生成训练命令
    if args.mode == 'sft':
        train_command = build_sft_command(args)
    else:
        train_command = build_rlhf_command(args)

    print("Training command:")
    print(train_command)
    print("=" * 60)

    # 提交 PyTorchJob
    job_name = default_job_name()
    print("Job name: %s" % job_name)
    launch_pytorchjob(name=job_name, num_workers=args.num_worker,
                      image=None, command=train_command)

    # 写 meta.json
    meta = {
        "completed_at": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "mode": args.mode,
        "model": args.model,
        "dataset": args.dataset,
        "save_path": args.save_path,
        "train_type": args.train_type,
        "distributed": args.distributed,
        "num_worker": args.num_worker,
        "nproc_per_node": args.nproc_per_node,
        "gpu_count": int(gpu_num),
        "world_size": args.num_worker * args.nproc_per_node,
        "batch_size_per_device": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_ddp_batch_size": (
            args.batch_size * args.num_worker * args.nproc_per_node *
            args.gradient_accumulation_steps
        ),
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "val_dataset": args.val_dataset,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "rlhf_type": args.rlhf_type if args.mode == 'rlhf' else None,
    }
    meta_path = os.path.join(args.save_path, 'meta.json')
    try:
        os.makedirs(args.save_path, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("meta.json written: %s" % meta_path)
    except Exception as e:
        print("meta.json write failed: %s" % e)

    print("=" * 60)
    print("msswift job completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()






