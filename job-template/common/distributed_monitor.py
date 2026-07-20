#!/usr/bin/env python3
"""
分布式训练监控辅助模块。

提供 primary/worker 判断，确保：
- 只有主进程注册 monitor 记录
- 只有主进程解析 experiment_id 和更新状态
- worker 进程不重复创建 SwanLab 实验

支持：
- PyTorch 分布式 (RANK / WORLD_RANK / LOCAL_RANK)
- TFJob (TF_CONFIG → chief/worker/ps)
- 通用环境变量 (OMPI_COMM_WORLD_RANK, RANK)
- 平台显式标记 (MLOPS_MONITOR_ROLE=primary/worker)

用法:
    from common.distributed_monitor import is_primary, get_role

    if not is_primary():
        print("[monitor] worker process, skipping monitor registration")
        # 跳过 monitor.start / wrapper register
"""
import json
import os


def _env_int(name, default=None):
    val = os.environ.get(name, "")
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_role():
    """
    返回当前进程角色: "primary" | "worker" | "unknown"

    优先级:
    1. MLOPS_MONITOR_ROLE 显式标记
    2. TF_CONFIG → task type
    3. PyTorch RANK / WORLD_RANK
    4. OMPI_COMM_WORLD_RANK
    5. 默认 primary
    """
    # 1. 平台显式标记
    explicit = os.environ.get("MLOPS_MONITOR_ROLE", "").strip().lower()
    if explicit in ("primary", "worker"):
        return explicit

    # 2. TFJob
    tf_config = os.environ.get("TF_CONFIG", "")
    if tf_config:
        try:
            cfg = json.loads(tf_config)
            task_type = cfg.get("task", {}).get("type", "")
            if task_type in ("chief", "master", "evaluator"):
                return "primary"
            if task_type in ("worker", "ps"):
                return "worker"
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. PyTorch 分布式
    rank = _env_int("RANK")
    world_rank = _env_int("WORLD_RANK")
    local_rank = _env_int("LOCAL_RANK")
    world_size = _env_int("WORLD_SIZE")

    if world_size is not None and world_size > 1:
        if rank is not None:
            return "primary" if rank == 0 else "worker"
        if world_rank is not None:
            return "primary" if world_rank == 0 else "worker"
        if local_rank is not None and local_rank > 0:
            return "worker"

    # 4. OpenMPI
    ompi_rank = _env_int("OMPI_COMM_WORLD_RANK")
    if ompi_rank is not None:
        return "primary" if ompi_rank == 0 else "worker"

    # 5. Ray: driver has RAY_ADDRESS set but no specific rank; worker tasks don't set it
    # Safe fallback: treat as primary unless WORLD_SIZE > 1 with non-zero RANK
    return "primary"


def is_primary():
    """当前进程是否为主进程（应负责监控注册和状态同步）。"""
    role = get_role()
    if role == "primary":
        return True
    print(f"[distributed_monitor] role={role}, skipping monitor operations")
    return False


def get_distributed_context():
    """返回分布式上下文信息，用于日志和调试。"""
    return {
        "role": get_role(),
        "RANK": _env_int("RANK"),
        "WORLD_RANK": _env_int("WORLD_RANK"),
        "LOCAL_RANK": _env_int("LOCAL_RANK"),
        "WORLD_SIZE": _env_int("WORLD_SIZE"),
        "TF_CONFIG": bool(os.environ.get("TF_CONFIG", "")),
        "MLOPS_MONITOR_ROLE": os.environ.get("MLOPS_MONITOR_ROLE", ""),
    }
