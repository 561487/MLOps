#!/usr/bin/env python3
import argparse
import copy
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

from job.pkgs.k8s.affinity import build_pod_anti_affinity


CRD_INFO = {
    "group": "kubeflow.org",
    "version": "v1",
    "kind": "PyTorchJob",
    "plural": "pytorchjobs",
    "timeout": 60 * 60 * 24 * 2,
}

DEFAULT_WORKER_IMAGE = (
    "10.121.177.20:8082/mlops/deepspeed:20260716-cuda121-ds0144-hf4442-r3"
)
DEFAULT_ENTRYPOINT = "/app/deepspeed_entrypoint.sh"


def _require_positive(name, value):
    if value <= 0:
        raise ValueError("%s must be greater than 0" % name)


def _validate_auto_or_positive_int(name, value):
    if value == "auto":
        return
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be 'auto' or a positive integer" % name) from exc
    _require_positive(name, parsed)


def validate_args(args):
    _require_positive("--num_worker", args.num_worker)
    _require_positive("--num_gpus", args.num_gpus)
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master_port must be between 1 and 65535")

    _require_positive("--per_device_train_batch_size", args.per_device_train_batch_size)
    _require_positive("--gradient_accumulation_steps", args.gradient_accumulation_steps)
    _require_positive("--max_seq_length", args.max_seq_length)
    _require_positive("--logging_steps", args.logging_steps)
    _require_positive("--save_steps", args.save_steps)
    _require_positive("--lora_rank", args.lora_rank)
    _require_positive("--lora_alpha", args.lora_alpha)
    if args.max_steps < 0:
        raise ValueError("--max_steps must be greater than or equal to 0")
    if args.num_train_epochs <= 0 and args.max_steps == 0:
        raise ValueError("--num_train_epochs must be greater than 0 when --max_steps is 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be greater than 0")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be greater than or equal to 0")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout must be in the range [0, 1)")
    if args.gradient_clipping < 0:
        raise ValueError("--gradient_clipping must be greater than or equal to 0")

    _validate_auto_or_positive_int("--train_batch_size", args.train_batch_size)
    _validate_auto_or_positive_int(
        "--train_micro_batch_size_per_gpu",
        args.train_micro_batch_size_per_gpu,
    )

    if args.offload_param != "none" and args.zero_stage != 3:
        raise ValueError("--offload_param requires --zero_stage 3")
    if args.offload_optimizer != "none" and args.zero_stage == 0:
        raise ValueError("--offload_optimizer requires --zero_stage 1, 2, or 3")

    if args.train_mode == "custom":
        if not args.command.strip():
            raise ValueError("--command is required in custom mode")
        return

    if not args.dataset_path:
        raise ValueError("--dataset_path is required for finetune and pretrain modes")
    if args.train_mode == "finetune" and not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required for finetune mode")
    if (
        args.train_mode == "pretrain"
        and not args.model_name_or_path
        and not args.model_config_path
    ):
        raise ValueError(
            "--model_name_or_path or --model_config_path is required for pretrain mode"
        )
    if not args.tokenizer_name_or_path and not args.model_name_or_path:
        raise ValueError(
            "--tokenizer_name_or_path is required when --model_name_or_path is empty"
        )


def parse_node_selector(raw):
    selectors = {}
    for item in re.split(r",|;|\n|\t", raw or ""):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        selectors[key.strip()] = value.strip()
    return selectors


def default_job_name(pipeline_name):
    name = "deepspeed-" + pipeline_name.replace("_", "-") + "-" + uuid.uuid4().hex[:4]
    return name[:54]


def run_shell(shell):
    print("begin run shell: %s" % shell, flush=True)
    cmd = subprocess.Popen(
        shell,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        bufsize=1,
    )
    while True:
        line = cmd.stdout.readline()
        status = subprocess.Popen.poll(cmd)
        if line:
            print(line, end="", flush=True)
        if status in (0, -9, -15, 143, 127):
            print("shell finish %s" % status, flush=True)
            break
        if status is not None:
            print("shell finish %s" % status, flush=True)
            break
    return cmd.returncode


def get_runtime_context(k8s_client, args):
    namespace = os.getenv("KFJ_NAMESPACE", "")
    creator = os.getenv("KFJ_CREATOR", "")
    volume_mount = os.getenv("KFJ_TASK_VOLUME_MOUNT", "")
    k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(
        volume_mount, creator
    )
    gpu_resource = os.getenv("KFJ_TASK_RESOURCE_GPU", "0")
    gpu_num, gpu_type, _ = k8s_client.get_gpu(gpu_resource)
    gpu_num = int(args.num_gpus) if args.num_gpus is not None else int(gpu_num)
    node_selector = parse_node_selector(
        os.getenv("KFJ_TASK_NODE_SELECTOR", "cpu=true,train=true")
    )
    if gpu_type:
        node_selector["gpu-type"] = gpu_type
    default_pod_resources = os.getenv("DEFAULT_POD_RESOURCES", "")
    return {
        "namespace": namespace,
        "task_id": os.getenv("KFJ_TASK_ID", ""),
        "task_name": os.getenv("KFJ_TASK_NAME", ""),
        "project": os.getenv("KFJ_TASK_PROJECT_NAME", "public"),
        "pipeline_id": os.getenv("KFJ_PIPELINE_ID", ""),
        "run_id": os.getenv("KFJ_RUN_ID", ""),
        "creator": creator,
        "runner": os.getenv("KFJ_RUNNER", ""),
        "pipeline_name": os.getenv("KFJ_PIPELINE_NAME", ""),
        "task_images": os.getenv("KFJ_TASK_IMAGES", ""),
        "resource_cpu": os.getenv("KFJ_TASK_RESOURCE_CPU", "2"),
        "resource_memory": os.getenv("KFJ_TASK_RESOURCE_MEMORY", "4G"),
        "gpu_resource_name": os.getenv("GPU_RESOURCE_NAME", ""),
        "gpu_resource": gpu_resource,
        "gpu_num": gpu_num,
        "rdma_resource_name": os.getenv("RDMA_RESOURCE_NAME", ""),
        "rdma_resource": os.getenv("KFJ_TASK_RESOURCE_RDMA", "0"),
        "hubsecret": [
            {"name": item}
            for item in os.getenv("HUBSECRET", "hubsecret").split(",")
            if item
        ],
        "scheduler": os.getenv("SCHEDULER", "volcano"),
        "image_pull_policy": os.getenv("IMAGE_PULL_POLICY", "IfNotPresent"),
        "default_pod_resources": json.loads(default_pod_resources)
        if default_pod_resources
        else {},
        "volumes": k8s_volumes,
        "volume_mounts": k8s_volume_mounts,
        "node_selector": node_selector,
        "entrypoint": args.entrypoint,
    }


def build_worker_command(args):
    command = [
        "bash",
        args.entrypoint,
        "--mode",
        args.train_mode,
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset_path",
        args.dataset_path,
        "--tokenizer_name_or_path",
        args.tokenizer_name_or_path,
        "--model_config_path",
        args.model_config_path,
        "--output_dir",
        args.output_dir,
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--max_steps",
        str(args.max_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_seq_length",
        str(args.max_seq_length),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--use_lora",
        str(args.use_lora).lower(),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--lora_target",
        args.lora_target,
        "--zero_stage",
        str(args.zero_stage),
        "--precision",
        args.precision,
        "--train_batch_size",
        args.train_batch_size,
        "--train_micro_batch_size_per_gpu",
        args.train_micro_batch_size_per_gpu,
        "--offload_optimizer",
        args.offload_optimizer,
        "--offload_param",
        args.offload_param,
        "--nvme_path",
        args.nvme_path,
        "--gradient_clipping",
        str(args.gradient_clipping),
        "--ds_config_override",
        args.ds_config_override,
        "--working_dir",
        args.working_dir,
        "--command",
        args.command,
        "--append_config_to_command",
        str(args.append_config_to_command).lower(),
        "--config_arg_name",
        args.config_arg_name,
    ]
    return " ".join(shlex.quote(item) for item in command)


def _base_pod_spec(name, args, ctx, command):
    labels = {
        "pipeline-id": ctx["pipeline_id"],
        "pipeline-name": ctx["pipeline_name"],
        "task-id": ctx["task_id"],
        "task-name": ctx["task_name"],
        "username": ctx["runner"],
        "component": name,
        "type": "deepspeed",
        "run-id": ctx["run_id"],
    }
    container = {
        "name": "pytorch",
        "image": args.image or ctx["task_images"] or DEFAULT_WORKER_IMAGE,
        "imagePullPolicy": ctx["image_pull_policy"],
        "workingDir": args.working_dir,
        "env": [
            {"name": "NCCL_DEBUG", "value": "INFO"},
            {"name": "GPU_NUM", "value": str(max(ctx["gpu_num"], 0))},
            {"name": "MASTER_PORT", "value": str(args.master_port)},
        ],
        "ports": [
            {
                "name": "pytorchjob-port",
                "containerPort": int(args.master_port),
            }
        ],
        "command": ["bash", "-c", command],
        "volumeMounts": ctx["volume_mounts"],
        "resources": {
            "requests": {
                **{
                    "cpu": ctx["resource_cpu"],
                    "memory": ctx["resource_memory"],
                },
                **ctx["default_pod_resources"],
            },
            "limits": {
                **{
                    "cpu": ctx["resource_cpu"],
                    "memory": ctx["resource_memory"],
                },
                **ctx["default_pod_resources"],
            },
        },
    }
    pod_spec = {
        "replicas": 1,
        "restartPolicy": "Never",
        "template": {
            "metadata": {
                "labels": labels,
                "annotations": {"project": ctx["project"]},
            },
            "spec": {
                "schedulerName": ctx["scheduler"],
                "restartPolicy": "Never",
                "volumes": ctx["volumes"],
                "imagePullSecrets": ctx["hubsecret"],
                "nodeSelector": copy.deepcopy(ctx["node_selector"]),
                "affinity": {
                    "podAntiAffinity": build_pod_anti_affinity(
                        {"component": name, "type": "deepspeed"},
                        args.num_worker,
                    )
                },
                "containers": [container],
            },
        },
    }
    return pod_spec


def apply_accelerator_resources(pod_spec, ctx):
    container = pod_spec["template"]["spec"]["containers"][0]
    node_selector = pod_spec["template"]["spec"]["nodeSelector"]
    gpu_num = ctx["gpu_num"]
    if gpu_num >= 1 and ctx["gpu_resource_name"]:
        container["resources"]["requests"][ctx["gpu_resource_name"]] = gpu_num
        container["resources"]["limits"][ctx["gpu_resource_name"]] = gpu_num
        node_selector.pop("cpu", None)
        node_selector["gpu"] = "true"
    elif gpu_num == 0:
        container["env"].append(
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"}
        )

    if (
        ctx["rdma_resource_name"]
        and ctx["rdma_resource"]
        and int(ctx["rdma_resource"]) > 0
    ):
        amount = int(ctx["rdma_resource"])
        container["resources"]["requests"][ctx["rdma_resource_name"]] = amount
        container["resources"]["limits"][ctx["rdma_resource_name"]] = amount
        container["securityContext"] = {"capabilities": {"add": ["IPC_LOCK"]}}


def make_pytorchjob(name, args, ctx, worker_command):
    master_spec = _base_pod_spec(name, args, ctx, worker_command)
    apply_accelerator_resources(master_spec, ctx)
    replica_specs = {"Master": master_spec}
    worker_replicas = int(args.num_worker) - 1
    if worker_replicas > 0:
        worker_spec = copy.deepcopy(master_spec)
        worker_spec["replicas"] = worker_replicas
        replica_specs["Worker"] = worker_spec
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "namespace": ctx["namespace"],
            "name": name,
            "labels": {
                "run-id": ctx["run_id"],
                "run-username": ctx["runner"],
                "pipeline-username": ctx["creator"],
                "pipeline-id": ctx["pipeline_id"],
                "pipeline-name": ctx["pipeline_name"],
                "task-id": ctx["task_id"],
                "task-name": ctx["task_name"],
            },
            "annotations": {"project": ctx["project"]},
        },
        "spec": {
            "backoffLimit": int(args.num_worker),
            "cleanPodPolicy": "None",
            "pytorchReplicaSpecs": replica_specs,
        },
    }


def monitoring(k8s_client, name, namespace, timed_out):
    import psutil

    time.sleep(10)

    def stern_pids():
        pids = []
        for proc in psutil.process_iter():
            if "stern" in proc.name():
                pids.append(proc.pid)
        return pids

    start_time = datetime.datetime.now()
    check_time = start_time
    while True:
        pytorchjob = k8s_client.get_one_crd(
            group=CRD_INFO["group"],
            version=CRD_INFO["version"],
            plural=CRD_INFO["plural"],
            namespace=namespace,
            name=name,
        )
        if pytorchjob:
            print("pytorchjob status %s" % pytorchjob["status"], flush=True)
            status = pytorchjob.get("status", "").lower()
            if status != "running":
                events = k8s_client.get_job_event(namespace=namespace, job_name=name)
                if events:
                    print(events[-1].get("message", ""), flush=True)
        else:
            print("pytorchjob not exist", flush=True)

        if pytorchjob and pytorchjob["status"] in ("Succeeded", "Failed"):
            for pid in stern_pids():
                psutil.Process(int(pid)).terminate()
                print("kill process %s" % pid, flush=True)
            break

        if (
            datetime.datetime.now() - start_time
        ).total_seconds() > CRD_INFO["timeout"]:
            print(
                "pytorchjob %s timed out after %s seconds"
                % (name, CRD_INFO["timeout"]),
                flush=True,
            )
            timed_out.set()
            for pid in stern_pids():
                psutil.Process(int(pid)).terminate()
                print("kill process %s" % pid, flush=True)
            break

        if (datetime.datetime.now() - check_time).total_seconds() > 3600:
            for pid in stern_pids():
                psutil.Process(int(pid)).terminate()
                print("kill process %s" % pid, flush=True)
            check_time = datetime.datetime.now()
        time.sleep(60)


def launch_pytorchjob(args):
    from job.pkgs.k8s.py_k8s import K8s

    k8s_client = K8s()
    ctx = get_runtime_context(k8s_client, args)
    name = args.job_name or default_job_name(ctx["pipeline_name"] or "pipeline")
    worker_command = build_worker_command(args)
    print("worker command: %s" % worker_command, flush=True)

    if ctx["run_id"]:
        print("delete old deepspeed pytorchjob, run-id %s" % ctx["run_id"], flush=True)
        k8s_client.delete_crd(
            group=CRD_INFO["group"],
            version=CRD_INFO["version"],
            plural=CRD_INFO["plural"],
            namespace=ctx["namespace"],
            labels={"run-id": ctx["run_id"]},
        )
        time.sleep(10)

    k8s_client.delete_crd(
        group=CRD_INFO["group"],
        version=CRD_INFO["version"],
        plural=CRD_INFO["plural"],
        namespace=ctx["namespace"],
        name=name,
    )
    time.sleep(10)

    pytorchjob_json = make_pytorchjob(name, args, ctx, worker_command)
    print("create new deepspeed pytorchjob %s" % name, flush=True)
    k8s_client.create_crd(
        group=CRD_INFO["group"],
        version=CRD_INFO["version"],
        plural=CRD_INFO["plural"],
        namespace=ctx["namespace"],
        body=pytorchjob_json,
    )
    time.sleep(10)

    timed_out = threading.Event()
    monitoring_thread = threading.Thread(
        target=monitoring,
        args=(k8s_client, name, ctx["namespace"], timed_out),
    )
    monitoring_thread.start()
    while not timed_out.is_set():
        print("begin follow log\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", flush=True)
        command = (
            "stern %s --namespace %s --exclude-container init-pytorch "
            "--since 10s --template '{{.PodName}} {{.Message}} {{\"\\n\"}}' "
            % (name, ctx["namespace"])
        )
        run_shell(command)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nend follow log", flush=True)
        if timed_out.is_set():
            break
        time.sleep(10)
        pytorchjob = k8s_client.get_one_crd(
            group=CRD_INFO["group"],
            version=CRD_INFO["version"],
            plural=CRD_INFO["plural"],
            namespace=ctx["namespace"],
            name=name,
        )
        if pytorchjob and pytorchjob["status"] in ("Succeeded", "Failed"):
            break

    if timed_out.is_set():
        print("pytorchJob %s failed: timeout" % name, flush=True)
        sys.exit(1)

    pytorchjob = k8s_client.get_one_crd(
        group=CRD_INFO["group"],
        version=CRD_INFO["version"],
        plural=CRD_INFO["plural"],
        namespace=ctx["namespace"],
        name=name,
    )
    if not pytorchjob:
        print("pytorchJob %s disappeared before completion" % name, flush=True)
        sys.exit(1)
    print("pytorchJob %s finished, status %s" % (name, pytorchjob["status"]))
    if pytorchjob["status"] != "Succeeded":
        sys.exit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("DeepSpeed PyTorchJob launcher")
    parser.add_argument("--job_name", default="")
    parser.add_argument("--train_mode", choices=["finetune", "pretrain", "custom"], default="finetune")
    parser.add_argument("--image", default=DEFAULT_WORKER_IMAGE)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--working_dir", default="/app")
    parser.add_argument("--output_dir", default="/mnt/deepspeed-output")
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--model_config_path", default="")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--use_lora", default="false")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target", default="")
    parser.add_argument("--zero_stage", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--train_batch_size", default="auto")
    parser.add_argument("--train_micro_batch_size_per_gpu", default="auto")
    parser.add_argument("--offload_optimizer", choices=["none", "cpu", "nvme"], default="none")
    parser.add_argument("--offload_param", choices=["none", "cpu", "nvme"], default="none")
    parser.add_argument("--nvme_path", default="/local_nvme")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--ds_config_override", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--append_config_to_command", default="true")
    parser.add_argument(
        "--config_arg_name",
        default="deepspeed",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print("%s args: %s" % (__file__, args), flush=True)
    launch_pytorchjob(args)


if __name__ == "__main__":
    main()
