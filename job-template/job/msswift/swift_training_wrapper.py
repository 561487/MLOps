#!/usr/bin/env python3
"""Run ms-swift with fail-open, all-node SwanLab monitoring."""

import ast
import json
import os
import subprocess
import sys


class _NoopReporter:
    def log(self, *_args, **_kwargs):
        return


def _parse_metrics(line):
    text = line.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        value = json.loads(text)
    except Exception:
        try:
            value = ast.literal_eval(text)
        except Exception:
            return {}
    if not isinstance(value, dict):
        return {}
    metrics = {}
    for key, item in value.items():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            name = str(key)
            metrics[name if name.startswith(("train/", "rlhf/", "metrics/"))
                    else "train/" + name] = item
    return metrics


def _rank():
    try:
        return int(os.environ.get("RANK", os.environ.get("NODE_RANK", "0")) or 0)
    except (TypeError, ValueError):
        return 0


def main():
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("swift_training_wrapper: missing command", file=sys.stderr)
        return 2

    rank = _rank()
    explicit_role = os.environ.get("MLOPS_MONITOR_ROLE", "").lower()
    is_master = explicit_role == "primary" or (not explicit_role and rank == 0)
    os.environ["MLOPS_MONITOR_ROLE"] = "primary" if is_master else "worker"

    monitor = None
    hardware = None
    if is_master:
        try:
            # DistributedHardwareMonitor owns GPU sampling, preventing duplicates.
            os.environ["SWANLAB_PROBE_MONITOR"] = "false"
            from common.training_monitor import TrainingMonitor
            monitor = TrainingMonitor()
            monitor.start()
            monitor.log({
                "config/world_size": int(os.environ.get("WORLD_SIZE", "1") or 1),
                "config/expected_nodes": int(
                    os.environ.get("MLOPS_MONITOR_EXPECTED_NODES", "1") or 1),
            })
        except Exception as error:
            print("[swift_monitor] WARNING: monitor init failed: %s" % error, flush=True)

    try:
        from common.distributed_hardware_monitor import DistributedHardwareMonitor
        hardware = DistributedHardwareMonitor(monitor or _NoopReporter())
        hardware.start()
    except Exception as error:
        print("[swift_monitor] WARNING: hardware monitor init failed: %s" % error, flush=True)

    return_code = 1
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            if monitor:
                metrics = _parse_metrics(line)
                if metrics:
                    monitor.log(metrics)
        return_code = process.wait()
    except Exception as error:
        print("[swift_monitor] command failed: %s" % error, file=sys.stderr, flush=True)
        return_code = 1
    finally:
        if hardware:
            try:
                hardware.stop()
            except Exception:
                pass
        if monitor:
            try:
                monitor.finish("SUCCEEDED" if return_code == 0 else "FAILED")
            except Exception as error:
                print("[swift_monitor] WARNING: finish failed: %s" % error, flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
