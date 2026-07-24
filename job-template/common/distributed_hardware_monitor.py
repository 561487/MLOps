"""Fail-open all-node GPU monitoring for distributed training jobs."""

import json
import os
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _integer(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class DistributedHardwareMonitor:
    """One collector per Pod; the master owns the only SwanLab reporter."""

    def __init__(self, reporter):
        self.reporter = reporter
        self.rank = _integer("RANK", _integer("NODE_RANK", 0))
        explicit_role = os.environ.get("MLOPS_MONITOR_ROLE", "").lower()
        self.is_master = explicit_role == "primary" or (not explicit_role and self.rank == 0)
        self.node_name = "master_0" if self.is_master else "worker_%d" % max(0, self.rank - 1)
        self.interval = max(2, _integer("SWANLAB_PROBE_MONITOR_INTERVAL", 10))
        self.port = _integer("MLOPS_MONITOR_MASTER_PORT", 29501)
        self.master_addr = os.environ.get(
            "MLOPS_MONITOR_MASTER_ADDR", os.environ.get("MASTER_ADDR", ""))
        self.token = os.environ.get("MLOPS_MONITOR_TOKEN", "")
        self.expected_nodes = max(1, _integer("MLOPS_MONITOR_EXPECTED_NODES", 1))
        self.stop_event = threading.Event()
        self.thread = None
        self.server = None
        self.server_thread = None
        self.latest = {}
        self.latest_lock = threading.Lock()
        self._last_warning = {}

    def _warn(self, key, message):
        now = time.time()
        if now - self._last_warning.get(key, 0) >= 60:
            print("[distributed_hardware_monitor] WARNING: %s" % message, flush=True)
            self._last_warning[key] = now

    def _collect(self):
        metrics = {}
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], stderr=subprocess.STDOUT, timeout=8).decode("utf-8")
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                index = int(parts[0])
                utilization, used, total, temperature, power = map(float, parts[1:6])
            except (TypeError, ValueError):
                continue
            prefix = "hardware/%s/gpu_%d" % (self.node_name, index)
            metrics[prefix + "/util_percent"] = utilization
            metrics[prefix + "/memory_used_mb"] = used
            metrics[prefix + "/memory_total_mb"] = total
            metrics[prefix + "/memory_percent"] = used * 100.0 / total if total else 0.0
            metrics[prefix + "/temperature_c"] = temperature
            metrics[prefix + "/power_w"] = power
        return metrics

    def _cluster_metrics(self):
        now = time.time()
        with self.latest_lock:
            active = {
                node: item for node, item in self.latest.items()
                if now - item["timestamp"] <= max(30, self.interval * 3)
            }
        utilization = []
        memory = []
        master_utilization = []
        worker_utilization = []
        for node, item in active.items():
            for key, value in item["metrics"].items():
                if key.endswith("/util_percent"):
                    utilization.append(float(value))
                    target = master_utilization if node.startswith("master_") else worker_utilization
                    target.append(float(value))
                elif key.endswith("/memory_used_mb"):
                    memory.append(float(value))
        result = {
            "cluster/alive_nodes": len(active),
            "cluster/expected_nodes": self.expected_nodes,
        }
        if utilization:
            result.update({
                "cluster/gpu_util_avg": sum(utilization) / len(utilization),
                "cluster/gpu_util_min": min(utilization),
                "cluster/gpu_util_max": max(utilization),
                "cluster/gpu_util_std": statistics.pstdev(utilization),
            })
        if memory:
            average = sum(memory) / len(memory)
            result.update({
                "cluster/memory_avg_mb": average,
                "cluster/memory_min_mb": min(memory),
                "cluster/memory_max_mb": max(memory),
                "cluster/memory_imbalance_ratio":
                    (max(memory) - min(memory)) / average if average else 0.0,
            })
        if master_utilization:
            master_average = sum(master_utilization) / len(master_utilization)
            result["cluster/master_gpu_util_avg"] = master_average
            if worker_utilization:
                worker_average = sum(worker_utilization) / len(worker_utilization)
                result["cluster/worker_gpu_util_avg"] = worker_average
                result["cluster/master_worker_util_gap"] = abs(master_average - worker_average)
        return result

    def _accept(self, payload):
        node = str(payload.get("node", "unknown"))
        raw_metrics = payload.get("metrics") or {}
        metrics = {
            str(key): value for key, value in raw_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        with self.latest_lock:
            self.latest[node] = {"timestamp": time.time(), "metrics": metrics}
        if metrics:
            self.reporter.log(metrics)
        self.reporter.log(self._cluster_metrics())

    def _start_server(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/health":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                if self.path != "/metrics/node":
                    self.send_error(404)
                    return
                if owner.token and self.headers.get("X-Monitor-Token", "") != owner.token:
                    self.send_error(403)
                    return
                try:
                    size = min(int(self.headers.get("Content-Length", "0")), 1024 * 1024)
                    owner._accept(json.loads(self.rfile.read(size).decode("utf-8")))
                    self.send_response(204)
                    self.end_headers()
                except Exception:
                    self.send_error(400)

            def log_message(self, *_args):
                return

        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                name="swift-monitor-aggregator",
                daemon=True)
            self.server_thread.start()
            print("[distributed_hardware_monitor] aggregator port=%d" % self.port, flush=True)
        except Exception as error:
            self._warn("server", "cannot start aggregator: %s" % error)

    def _send_worker_sample(self, metrics):
        if not self.master_addr:
            self._warn("address", "MASTER_ADDR is empty; dropping worker sample")
            return
        payload = json.dumps({
            "node": self.node_name,
            "timestamp": time.time(),
            "metrics": metrics,
        }).encode("utf-8")
        url = "http://%s:%d/metrics/node" % (self.master_addr, self.port)
        try:
            import urllib.request
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["X-Monitor-Token"] = self.token
            request = urllib.request.Request(
                url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=2):
                return
        except Exception as error:
            self._warn("upload", "worker sample upload failed: %s" % error)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                metrics = self._collect()
                if self.is_master:
                    self._accept({"node": self.node_name, "metrics": metrics})
                else:
                    self._send_worker_sample(metrics)
            except Exception as error:
                self._warn("collect", "hardware collection failed: %s" % error)
            self.stop_event.wait(self.interval)

    def start(self):
        if os.environ.get(
                "MLOPS_DISTRIBUTED_HARDWARE_ENABLE", "true").lower() != "true":
            return
        if self.is_master:
            self._start_server()
        self.thread = threading.Thread(
            target=self._loop, name="swift-node-gpu-monitor", daemon=True)
        self.thread.start()
        print("[distributed_hardware_monitor] started node=%s rank=%d" %
              (self.node_name, self.rank), flush=True)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.server:
            self.server.shutdown()
            self.server.server_close()
