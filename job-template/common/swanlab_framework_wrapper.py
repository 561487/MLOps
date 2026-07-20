#!/usr/bin/env python3
"""
SwanLab Framework Wrapper — 大模型微调训练监控生命周期管理。

用于 LLaMA-Factory / ModelScope Swift / Transformers / TRL 等框架。
训练脚本通过 --report_to swanlab 使用框架原生 SwanLab callback，
本 wrapper 负责 monitor 记录的注册、experiment_id 解析、状态同步。

用法:
    python3 /app/common/swanlab_framework_wrapper.py -- llamafactory-cli train ... --report_to swanlab

    # 也可通过环境变量指定额外参数:
    WRAPPER_PROJECT=my-project python3 /app/common/swanlab_framework_wrapper.py -- ...
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

SWANLAB_HOST = os.environ.get("SWANLAB_WEB_HOST", "http://10.121.177.20:30092")
SWANLAB_LOGDIR = os.environ.get("SWANLAB_LOGDIR", "/mnt/storage/swanlab/swanlog")
REGISTER_URL = os.environ.get("MLOPS_MONITOR_REGISTER_URL", "")
MONITOR_ENABLED = os.environ.get("MLOPS_TRAINING_MONITOR_ENABLE", "").lower() == "true"

# How often to check for experiment_id resolution (seconds)
_URL_CHECK_INTERVAL = 10
# Max time to wait for experiment_id resolution after training completes (seconds)
_URL_FINAL_WAIT = 60


def _log(msg):
    print(f"[swanlab_wrapper] {msg}", flush=True)


def _warn(msg):
    print(f"[swanlab_wrapper] WARNING: {msg}", file=sys.stderr, flush=True)


def _http_get(url, timeout=5):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _http_post(url, data, timeout=10):
    try:
        import urllib.request
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        _warn(f"POST {url} failed: {e}")
        return None


def _find_run_dir():
    """Find the latest run-* directory in the swanlog dir."""
    logdir = SWANLAB_LOGDIR
    if not os.path.isdir(logdir):
        return None
    dirs = sorted(
        [d for d in os.listdir(logdir) if d.startswith("run-") and os.path.isdir(os.path.join(logdir, d))],
        key=lambda d: os.path.getmtime(os.path.join(logdir, d)),
        reverse=True,
    )
    return dirs[0] if dirs else None


def _resolve_experiment_id(run_dir_name):
    """Query Dashboard API to resolve numeric experiment_id."""
    data = _http_get(f"{SWANLAB_HOST}/api/v1/project")
    if not data:
        return None
    experiments = []
    if isinstance(data, dict):
        experiments = data.get("experiments", data.get("data", {}).get("experiments", []))
    for exp in experiments:
        if isinstance(exp, dict) and exp.get("run_id") == run_dir_name:
            return exp.get("id")
    return None


class MonitorContext:
    """Context that tracks the monitor record and resolves experiment_id."""

    def __init__(self):
        self.run_dir_name = ""
        self.experiment_id = None
        self.monitor_url = f"{SWANLAB_HOST}/"
        self.monitor_status = "RUNNING"
        self.record_id = None
        self._stop_event = threading.Event()
        self._thread = None

    def _build_payload(self):
        return {
            "run_id": os.environ.get("MLOPS_PIPELINE_RUN_ID", ""),
            "workflow_name": os.environ.get("MLOPS_WORKFLOW_NAME", ""),
            "task_id": os.environ.get("MLOPS_TASK_ID", ""),
            "task_name": os.environ.get("MLOPS_TASK_NAME", ""),
            "node_name": os.environ.get("MLOPS_NODE_NAME", ""),
            "pod_name": os.environ.get("K8S_POD_NAME", ""),
            "job_template_name": os.environ.get("MLOPS_JOB_TEMPLATE_NAME", ""),
            "monitor_type": "swanlab",
            "monitor_url": self.monitor_url,
            "monitor_status": self.monitor_status,
            "creator": os.environ.get("USERNAME", ""),
            "experiment_run_id": self.run_dir_name,
        }

    def register(self):
        """Register the monitor record (initial URL = homepage)."""
        if not REGISTER_URL:
            _warn("MLOPS_MONITOR_REGISTER_URL not set, skipping register")
            return False
        self.monitor_url = f"{SWANLAB_HOST}/"
        self.monitor_status = "RUNNING"
        resp = _http_post(REGISTER_URL, self._build_payload())
        if resp and resp.get("success"):
            self.record_id = resp.get("id")
            _log(f"registered: id={self.record_id}, action={resp.get('action')}")
            return True
        _warn(f"register failed: {resp}")
        return False

    def update(self):
        """Update the monitor record with current URL and status."""
        if not REGISTER_URL:
            return
        payload = self._build_payload()
        payload["experiment_run_id"] = self.run_dir_name
        _http_post(REGISTER_URL, payload)

    def _watch_loop(self):
        """Background thread: periodically resolve experiment_id."""
        while not self._stop_event.is_set():
            if self.experiment_id is None:
                rn = _find_run_dir()
                if rn:
                    self.run_dir_name = rn
                    eid = _resolve_experiment_id(rn)
                    if eid is not None:
                        self.experiment_id = eid
                        self.monitor_url = f"{SWANLAB_HOST}/experiment/{eid}/chart"
                        _log(f"experiment_id resolved: {rn} -> {eid}")
                        self.update()
            self._stop_event.wait(_URL_CHECK_INTERVAL)

    def start_watching(self):
        self._thread = threading.Thread(target=self._watch_loop, daemon=True,
                                        name="swanlab-wrapper-watch")
        self._thread.start()

    def stop_watching(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def finalize(self, exit_code):
        """Update final status. Block up to _URL_FINAL_WAIT to resolve experiment_id."""
        self.stop_watching()
        self.monitor_status = "SUCCEEDED" if exit_code == 0 else "FAILED"

        # Final attempt to resolve experiment_id if not yet resolved
        if self.experiment_id is None:
            rn = _find_run_dir()
            if rn:
                self.run_dir_name = rn
            deadline = time.time() + _URL_FINAL_WAIT
            while time.time() < deadline and self.experiment_id is None:
                if rn:
                    eid = _resolve_experiment_id(rn)
                    if eid is not None:
                        self.experiment_id = eid
                        self.monitor_url = f"{SWANLAB_HOST}/experiment/{eid}/chart"
                        _log(f"experiment_id resolved at finalize: {rn} -> {eid}")
                        break
                time.sleep(3)

        self.update()
        _log(f"finalize: status={self.monitor_status}, url={self.monitor_url}")


def parse_args():
    """Parse wrapper args: everything after '--' is the training command."""
    args = sys.argv[1:]
    separator_idx = None
    for i, a in enumerate(args):
        if a == "--":
            separator_idx = i
            break
    if separator_idx is None:
        _log("no '--' separator found, treating all args as training command")
        return [], args
    return args[:separator_idx], args[separator_idx + 1:]


def main():
    if not MONITOR_ENABLED:
        _log("MLOPS_TRAINING_MONITOR_ENABLE is not true, running command directly")
        _, cmd = parse_args()
        if not cmd:
            _log("no command to execute")
            return 1
        return subprocess.run(cmd).returncode

    # 分布式：只允许 primary 进程执行注册和状态更新
    try:
        from common.distributed_monitor import is_primary, get_role as _get_dist_role
        if not is_primary():
            _role = _get_dist_role()
            _log(f"worker process (role={_role}), running command without monitor")
            _, cmd = parse_args()
            if not cmd:
                return 1
            return subprocess.run(cmd).returncode
    except ImportError:
        pass

    wrapper_args, cmd = parse_args()

    if not cmd:
        _log("no training command provided after '--'")
        _log("usage: python3 swanlab_framework_wrapper.py -- <training command>")
        return 1

    _log(f"training command: {' '.join(cmd)}")

    ctx = MonitorContext()

    # 1. Register before training starts
    if not ctx.register():
        _warn("register failed, continuing training anyway")

    # 2. Start background experiment_id resolution
    ctx.start_watching()

    # 3. Execute training command with SIGTERM/SIGINT forwarding
    _log("starting training process...")
    _proc = None
    _orig_sigterm = signal.signal(signal.SIGTERM, lambda s, f: _proc.terminate() if _proc else None)
    _orig_sigint = signal.signal(signal.SIGINT, lambda s, f: _proc.terminate() if _proc else None)
    try:
        _proc = subprocess.Popen(cmd)
        exit_code = _proc.wait()
    except Exception as e:
        _warn(f"training process error: {e}")
        exit_code = 1
    finally:
        signal.signal(signal.SIGTERM, _orig_sigterm)
        signal.signal(signal.SIGINT, _orig_sigint)

    _log(f"training process exited with code {exit_code}")

    # 4. Finalize monitor record
    try:
        ctx.finalize(exit_code)
    except Exception as e:
        _warn(f"finalize error: {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
