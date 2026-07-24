"""
训练监控公共模块 —— SwanLab 接入 + 硬件监控线程。

用法:
    from common.training_monitor import TrainingMonitor
    monitor = TrainingMonitor()
    monitor.start()
    monitor.log({"loss": 0.5, "accuracy": 0.9})
    monitor.finish("SUCCEEDED")
"""

import json
import os
import re as _re
import sys
import subprocess
import threading

import numpy as np
import time
import traceback
import warnings

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

SWANLAB_HOST = os.environ.get("SWANLAB_WEB_HOST", "http://10.121.177.20:30092")

_K8S_SWANLAB_SAFE_KEYS = frozenset({
    "SWANLAB_MODE", "SWANLAB_API_HOST", "SWANLAB_WEB_HOST",
    "SWANLAB_API_KEY", "SWANLAB_WORKSPACE",
    "SWANLAB_LOGDIR", "SWANLAB_PROJ_NAME", "SWANLAB_EXP_NAME",
    "SWANLAB_GROUP", "SWANLAB_TAGS",
    "SWANLAB_PROBE_HARDWARE", "SWANLAB_PROBE_MONITOR", "SWANLAB_PROBE_MONITOR_INTERVAL",
})


def _sanitize_k8s_env():
    removed = []
    for k in list(os.environ):
        if k.startswith("SWANLAB_") and k not in _K8S_SWANLAB_SAFE_KEYS:
            removed.append((k, os.environ.pop(k, None)))
    if removed:
        print(f"[training_monitor] cleaned K8s env conflicts: {[r[0] for r in removed]}")


def _urlopen_get(url, timeout=5):
    try:
        import requests
        return requests.get(url, timeout=timeout).text
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8")


class TrainingMonitor:

    # basic mode allowed metric keys (exact match or prefix match for wildcard patterns)
    _BASIC_KEYS = frozenset({
        "trial/index", "trial/score", "trial/std_score", "trial/duration_seconds",
        "trial/accuracy", "trial/f1_macro", "trial/r2", "trial/rmse", "trial/mae",
        "best/score", "best/trial_index", "best/cv_score",
        "test/accuracy", "test/f1_macro", "test/f1_weighted",
        "test/precision_macro", "test/recall_macro",
        "test/rmse", "test/mae", "test/r2",
        "dataset/train_samples", "dataset/test_samples", "dataset/num_features",
        "config/cv", "config/n_trials",
        "hardware/cpu_percent", "hardware/memory_percent", "hardware/memory_used_gb",
        "hardware/gpu_0_util_percent", "hardware/gpu_0_memory_used_mb",
        "hardware/gpu_0_temperature_c", "hardware/gpu_0_power_w",
    })
    # prefix-based allowlist for dynamic keys
    _BASIC_PREFIXES = ("fold/", "best_params/", "train/", "config/", "gpu/", "hardware/", "cluster/",
                       "metrics/", "error/", "experiment/", "trial/", "dataset/")

    def __init__(self):
        self._swanlab = None
        self._run = None
        self._enabled = False
        self._monitor_url = ""
        self._monitor_status = "INIT"
        self._experiment_name = ""
        self._log_dir = ""
        self._run_dir_name = ""
        self._experiment_id = None
        self._hw_thread = None
        self._hw_stop = None
        self._hw_interval = 10
        self._log_count = 0
        self._finished = False
        self._last_url_check = 0.0
        self._metric_level = os.environ.get("SWANLAB_METRIC_LEVEL", "basic")
        self._check_enabled()

    def _check_enabled(self):
        self._enabled = os.environ.get("MLOPS_TRAINING_MONITOR_ENABLE", "").lower() == "true"
        if self._enabled:
            try:
                from common.distributed_monitor import is_primary
                if not is_primary():
                    self._info("worker process detected, disabling monitor")
                    self._enabled = False
            except ImportError:
                pass  # distributed_monitor not available, assume single-node

    def _warn(self, msg):
        print(f"[training_monitor] WARNING: {msg}", file=sys.stderr)

    def _info(self, msg):
        print(f"[training_monitor] INFO: {msg}")

    # ---- Hardware monitoring ----

    def _hardware_collect(self):
        m = {}
        if _HAS_PSUTIL:
            try:
                m["hardware/cpu_percent"] = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                m["hardware/memory_percent"] = mem.percent
                m["hardware/memory_used_gb"] = round(mem.used / (1024**3), 2)
            except Exception:
                pass
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], timeout=8)
            for line in out.decode().strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                idx = parts[0]
                try:
                    util, mu, mt, temp, power = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                m[f"hardware/gpu_{idx}_util_percent"] = util
                m[f"hardware/gpu_{idx}_memory_used_mb"] = mu
                m[f"hardware/gpu_{idx}_temperature_c"] = temp
                m[f"hardware/gpu_{idx}_power_w"] = power
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return m

    def _hw_loop(self):
        self._info(f"hw monitor started, interval={self._hw_interval}s")
        # immediate first sample
        try:
            mm = self._hardware_collect()
            if mm:
                self.log(mm)
        except Exception as e:
            self._warn(f"hw first sample error: {e}")
        while not self._hw_stop.is_set():
            self._hw_stop.wait(self._hw_interval)
            if self._hw_stop.is_set():
                break
            try:
                mm = self._hardware_collect()
                if mm:
                    self.log(mm)
            except Exception as e:
                self._warn(f"hw sample error: {e}")
        self._info("hw monitor stopped")

    def _start_hw_thread(self):
        if not _HAS_PSUTIL:
            self._warn("psutil missing, hw monitor disabled")
        if os.environ.get("SWANLAB_PROBE_MONITOR", "true").lower() != "true":
            return
        try:
            self._hw_interval = int(os.environ.get("SWANLAB_PROBE_MONITOR_INTERVAL", "10"))
        except ValueError:
            self._hw_interval = 10
        self._hw_stop = threading.Event()
        self._hw_thread = threading.Thread(target=self._hw_loop, daemon=True, name="swanlab-hw")
        self._hw_thread.start()

    def _stop_hw_thread(self):
        if self._hw_stop:
            self._hw_stop.set()
        if self._hw_thread and self._hw_thread.is_alive():
            self._hw_thread.join(timeout=5)

    # ---- Dashboard API experiment_id resolution ----

    def _resolve_experiment_id(self, max_retries=3, interval=1):
        if not self._run_dir_name:
            return None
        for attempt in range(max_retries):
            try:
                text = _urlopen_get(f"{SWANLAB_HOST}/api/v1/project", timeout=5)
                data = json.loads(text)
                experiments = []
                if isinstance(data, dict):
                    experiments = data.get("experiments", data.get("data", {}).get("experiments", []))
                for exp in experiments:
                    if isinstance(exp, dict) and exp.get("run_id") == self._run_dir_name:
                        return exp.get("id")
                for exp in experiments:
                    if isinstance(exp, dict) and exp.get("name") == self._experiment_name:
                        return exp.get("id")
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(interval)
        return None

    def _try_update_experiment_url(self):
        eid = self._resolve_experiment_id(max_retries=1, interval=0)
        if eid is not None:
            self._experiment_id = eid
            self._monitor_url = f"{SWANLAB_HOST}/experiment/{eid}/chart"
            self._info(f"experiment_id resolved: {self._run_dir_name} → {eid}")
            self._register()
            return True
        return False

    def _check_url_periodic(self):
        """Time+count based periodic URL resolution. Ensures URL is checked even
        when all metrics are filtered in basic mode."""
        if self._experiment_id is not None:
            return
        now = time.time()
        # every 3 log calls OR every 15 seconds, whichever comes first
        if self._log_count % 3 == 0 or (now - self._last_url_check) >= 15:
            self._last_url_check = now
            self._try_update_experiment_url()

    def _retry_experiment_url_blocking(self, max_retries=30, interval=2):
        if self._experiment_id is not None:
            return
        # Cloud 模式下 monitor_url 已在 init 时通过 run.url 确定，无需轮询 Dashboard API
        swanlab_mode = os.environ.get("SWANLAB_MODE", "local").lower()
        if swanlab_mode == "online":
            swanlab_mode = "cloud"
        if swanlab_mode == "cloud":
            self._info("cloud mode: skipping experiment_id polling, url already resolved")
            return
        for attempt in range(max_retries):
            eid = self._resolve_experiment_id(max_retries=1, interval=0)
            if eid is not None:
                self._experiment_id = eid
                self._monitor_url = f"{SWANLAB_HOST}/experiment/{eid}/chart"
                self._info(f"experiment_id matched (attempt {attempt+1}): {self._run_dir_name} → {eid}")
                self._register()
                return
            if attempt < max_retries - 1:
                time.sleep(interval)
        self._warn(f"experiment_id not matched for {self._run_dir_name}, url stays as is")

    # ---- Public API ----

    def _resolve_run_url(self, run):
        """Try multiple ways to get the experiment URL from the run object."""
        # 1) run.get_url() (callable, newer SDK)
        get_url = getattr(run, "get_url", None)
        if callable(get_url):
            try:
                val = get_url()
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    return val
            except Exception:
                pass
        # 2) run.url (property, SDK 0.8.x)
        val = getattr(run, "url", None)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val
        # 3) run.public.cloud.experiment_url (legacy)
        public = getattr(run, "public", None)
        if public is not None:
            cloud = getattr(public, "cloud", None)
            if cloud is not None:
                val = getattr(cloud, "experiment_url", None)
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    return val
        return None

    def start(self):
        if not self._enabled:
            return
        _sanitize_k8s_env()

        # ---- Import swanlab ----
        try:
            import swanlab
            self._swanlab = swanlab
        except ImportError:
            self._warn("swanlab not installed (import failed)")
            self._enabled = False
            return
        except Exception:
            self._warn(f"SwanLab import failed:\n{traceback.format_exc()}")
            self._enabled = False
            return

        swanlab_mode = os.environ.get("SWANLAB_MODE", "local").lower()
        # "online" is the official SDK mode name; treat as synonym for "cloud"
        if swanlab_mode == "online":
            swanlab_mode = "cloud"
        self._experiment_name = os.environ.get("SWANLAB_EXP_NAME", "training-experiment")
        group = os.environ.get("SWANLAB_GROUP", "")
        tags = [t.strip() for t in os.environ.get("SWANLAB_TAGS", "mlops,training").split(",") if t.strip()]

        mlops_config = {
            "mlops_pipeline_run_id": os.environ.get("MLOPS_PIPELINE_RUN_ID", ""),
            "mlops_task_name": os.environ.get("MLOPS_TASK_NAME", ""),
            "mlops_node_name": os.environ.get("MLOPS_NODE_NAME", ""),
            "mlops_job_template_name": os.environ.get("MLOPS_JOB_TEMPLATE_NAME", ""),
        }

        try:
            if swanlab_mode == "cloud":
                # ---- Cloud / Self-hosted Online 模式 ----
                api_host = os.environ.get("SWANLAB_API_HOST", "")
                api_key = os.environ.get("SWANLAB_API_KEY", "")
                # 使用官方环境变量 SWANLAB_PROJ_NAME（不是 SWANLAB_PROJECT，
                # 后者会被 pydantic-settings 当作嵌套配置解析导致 init 失败）
                project_name = os.environ.get("SWANLAB_PROJ_NAME", "mlops-training")
                workspace = os.environ.get("SWANLAB_WORKSPACE", "")

                # 环境诊断日志（不打印 API Key 原文）
                api_key_len = len(api_key) if api_key else 0
                reg_url = os.environ.get("MLOPS_MONITOR_REGISTER_URL", "")
                self._info(f"env: mode={swanlab_mode}, api_host={api_host}, project={project_name}, "
                           f"workspace={workspace}, exp={self._experiment_name}, "
                           f"api_key_len={api_key_len}, register_url={reg_url}")

                if not api_host:
                    raise ValueError("SWANLAB_API_HOST is required for cloud mode")
                if not api_key:
                    raise ValueError("SWANLAB_API_KEY is required for cloud mode")

                self._info(f"SwanLab cloud init: host={api_host}, project={project_name}, "
                           f"workspace={workspace or '(default)'}, experiment={self._experiment_name}")

                # SDK 0.8.x: swanlab.init() 不接受 host/api_key 参数
                # 必须先通过 swanlab.login() 配置 API 连接信息
                if hasattr(self._swanlab, "login"):
                    self._swanlab.login(api_key=api_key, host=api_host, save=False)
                    self._info("SwanLab login done")

                init_kw = {
                    "mode": "cloud",  # SwanLab maps 'cloud' to 'online'
                    "project": project_name,
                    "experiment_name": self._experiment_name,
                    "group": group or None,
                    "tags": tags,
                    "config": mlops_config,
                }
                if workspace:
                    init_kw["workspace"] = workspace

                self._run = self._swanlab.init(**init_kw)
                self._monitor_status = "RUNNING"
                self._info("SwanLab experiment created (cloud mode)")

                if hasattr(self._run, "dir") and self._run.dir:
                    self._run_dir_name = os.path.basename(str(self._run.dir))
                    self._info(f"run_dir_name: {self._run_dir_name}")

                # Resolve monitor URL using SDK-native methods
                resolved_url = self._resolve_run_url(self._run)
                self._monitor_url = resolved_url or SWANLAB_HOST
                # Normalize: ensure /chart suffix for experiment URLs
                _mu = self._monitor_url
                if _re.search(r'/runs/[^/]+$', _mu):
                    self._monitor_url = _mu + '/chart'
                self._info(f"monitor_url: {self._monitor_url}")

                self._register()
                self._start_hw_thread()

            else:
                # ---- Local / Watch 模式（保持现有逻辑不变） ----
                proj = os.environ.get("SWANLAB_PROJ_NAME", "mlops-training")
                ld = os.environ.get("SWANLAB_LOGDIR", "")
                if ld:
                    os.makedirs(ld, exist_ok=True)
                    self._log_dir = ld
                else:
                    self._log_dir = None

                self._info(f"SwanLab local init: project={proj}, experiment={self._experiment_name}, "
                           f"log_dir={self._log_dir or 'default'}")

                kw = {
                    "project": proj,
                    "experiment_name": self._experiment_name,
                    "group": group or None,
                    "tags": tags,
                    "mode": "local",
                    "config": mlops_config,
                }
                if self._log_dir:
                    kw["log_dir"] = self._log_dir

                self._run = self._swanlab.init(**kw)
                self._monitor_status = "RUNNING"
                self._info("SwanLab experiment created (local mode)")

                if hasattr(self._run, "dir") and self._run.dir:
                    self._run_dir_name = os.path.basename(str(self._run.dir))
                    self._info(f"run_dir_name: {self._run_dir_name}")

                self._monitor_url = f"{SWANLAB_HOST}/"
                self._info(f"monitor_url (initial): {self._monitor_url}")
                self._register()
                self._start_hw_thread()
                self._try_update_experiment_url()

        except Exception:
            self._warn(f"SwanLab initialization failed:\n{traceback.format_exc()}")
            self._enabled = False
            # 即使初始化失败也尝试注册，便于排查
            try:
                self._register()
            except Exception:
                pass

    def log(self, metrics=None, step=None):
        if not self._enabled or self._run is None:
            return
        if not metrics:
            return
        # basic mode: filter out non-essential keys and non-numeric values
        if self._metric_level != "full":
            filtered = {}
            for k, v in metrics.items():
                if not isinstance(v, (int, float)) and not (
                    hasattr(v, "dtype") and np.issubdtype(np.asarray(v).dtype, np.number)):
                    continue
                if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                    continue
                if k in self._BASIC_KEYS or k.startswith(self._BASIC_PREFIXES):
                    filtered[k] = v
            if not filtered:
                self._info(f"all metrics filtered (basic), original_keys={list(metrics.keys())[:10]}")
                self._check_url_periodic()
                return
            metrics = filtered
        try:
            self._run.log(metrics, step=step)
        except Exception as e:
            self._warn(f"swanlab.log failed: {e}")
        self._log_count += 1
        self._check_url_periodic()

    def fail(self, error=None):
        """训练失败时调用。等同于 finish("FAILED")。"""
        if error:
            self._info(f"fail called: {error}")
        self.finish("FAILED")

    def finish(self, status="SUCCEEDED"):
        if self._finished:
            return
        self._info(f"finish called, status={status}")
        self._monitor_status = status
        self._stop_hw_thread()
        self._finished = True

        if not self._enabled:
            self._info("monitor not enabled, skipping swanlab.finish")
            return

        self._retry_experiment_url_blocking(max_retries=30, interval=2)

        try:
            if self._run is not None:
                self._run.finish()
                self._info("swanlab.finish called")
        except Exception as e:
            self._warn(f"swanlab.finish failed: {e}")

        self._register()
        self._info(f"monitor_status updated to {status}")

    # ---- Registration ----

    def _register(self):
        url = os.environ.get("MLOPS_MONITOR_REGISTER_URL", "")
        if not url:
            return
        mu = self._monitor_url or f"{SWANLAB_HOST}/"

        # Normalize: ensure /chart suffix for SwanLab run URLs
        if _re.search(r'/runs/[^/]+$', mu):
            mu += '/chart'
            self._monitor_url = mu

        payload = {
            "mlops_run_id": os.environ.get("MLOPS_PIPELINE_RUN_ID", ""),
            "run_id": os.environ.get("MLOPS_PIPELINE_RUN_ID", ""),
            "pipeline_id": os.environ.get("MLOPS_PIPELINE_ID", ""),
            "workflow_name": os.environ.get("MLOPS_WORKFLOW_NAME", ""),
            "task_id": os.environ.get("MLOPS_TASK_ID", ""),
            "task_name": os.environ.get("MLOPS_TASK_NAME", ""),
            "node_name": os.environ.get("MLOPS_NODE_NAME", ""),
            "pod_name": os.environ.get("K8S_POD_NAME", ""),
            "job_template_name": os.environ.get("MLOPS_JOB_TEMPLATE_NAME", ""),
            "monitor_type": os.environ.get("MLOPS_TRAINING_MONITOR_TYPE", "swanlab"),
            "monitor_url": mu,
            "monitor_status": self._monitor_status,
            "creator": os.environ.get("USERNAME", ""),
            "experiment_run_id": self._run_dir_name,
            "swanlab_run_id": self._run_dir_name or "",
            "swanlab_project": os.environ.get("SWANLAB_PROJ_NAME", "mlops-training"),
        }
        try:
            import requests
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                d = r.json()
                ok = d.get("status") == "ok" or d.get("success")
                if ok:
                    self._info(f"register ok: action={d.get('action')}, id={d.get('id')}")
                else:
                    self._warn(f"register returned fail: {d.get('message','')}")
            else:
                body = r.text[:500] if r.text else "(empty)"
                self._warn(f"register HTTP {r.status_code}, body={body}")
        except ImportError:
            try:
                import urllib.request
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:
                self._warn(f"register urllib failed: {e}")
        except Exception as e:
            self._warn(f"register error: {e}")
