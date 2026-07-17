#!/usr/bin/env python3
"""LightGBM training operator with GPU support and SwanLab monitoring."""

import argparse
import json
import os
import sys
import subprocess
import traceback

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor


def detect_gpu():
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=10
        )
        return True
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="LightGBM training operator (CPU/GPU)")

    # Data
    p.add_argument("--input_csv", default="", help="训练数据 CSV 路径")
    p.add_argument("--label_col", default="label", help="标签列名")
    p.add_argument("--task_type", default="classification",
                   choices=["classification", "regression"],
                   help="任务类型")
    p.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    p.add_argument("--random_state", type=int, default=42, help="随机种子")

    # Model params
    p.add_argument("--n_estimators", type=int, default=100, help="树数量")
    p.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    p.add_argument("--num_leaves", type=int, default=31, help="叶子节点数")
    p.add_argument("--max_depth", type=int, default=-1, help="最大深度")

    # Hidden GPU params (compat only, not shown in frontend)
    p.add_argument("--resource_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--use_gpu", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--device_type", default="", help=argparse.SUPPRESS)
    p.add_argument("--gpu_platform_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--gpu_device_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--max_bin", type=int, default=255, help=argparse.SUPPRESS)
    p.add_argument("--gpu_use_dp", type=str, default="false", help=argparse.SUPPRESS)
    p.add_argument("--allow_gpu_fallback", type=str, default="false", help=argparse.SUPPRESS)

    # Output
    p.add_argument("--output_model_path", default="", help="模型输出路径")
    p.add_argument("--output_metrics_path", default="", help="指标输出路径")

    return p.parse_args()


def main():
    args = parse_args()

    # Import TrainingMonitor (needs PYTHONPATH=/app)
    try:
        from common.training_monitor import TrainingMonitor
    except ImportError:
        TrainingMonitor = None

    monitor = TrainingMonitor() if TrainingMonitor else None
    if monitor:
        monitor.start()

    log_prefix = "[lightgbm]"
    _info = lambda msg: print(f"{log_prefix} INFO: {msg}", flush=True)
    _warn = lambda msg: print(f"{log_prefix} WARNING: {msg}", flush=True, file=sys.stderr)

    _info(f"========== LightGBM Operator Start ==========")
    _info(f"input_csv={args.input_csv}")
    _info(f"label_col={args.label_col}")
    _info(f"task_type={args.task_type}")
    _info(f"device_type={args.device_type}")

    try:
        import lightgbm as lgb
        _info(f"lightgbm version: {lgb.__version__}")
    except ImportError:
        _info("lightgbm __version__ not available")

    gpu_detected = detect_gpu()
    _info(f"nvidia-smi: {'OK' if gpu_detected else 'not detected'}")

    # ---- Validate inputs ----
    if not args.input_csv:
        raise ValueError("input_csv 不能为空")
    if not args.output_model_path:
        raise ValueError("output_model_path 不能为空")
    if not args.output_metrics_path:
        raise ValueError("output_metrics_path 不能为空")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"训练数据不存在: {args.input_csv}")

    # ---- Load data ----
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"训练数据为空: {args.input_csv}")
    if args.label_col not in df.columns:
        raise ValueError(f"标签列 {args.label_col} 不存在，字段: {list(df.columns)}")

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col])
    if X.shape[1] == 0:
        raise ValueError(f"无特征列")

    X = pd.get_dummies(X, dummy_na=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    _info(f"data: {X_train.shape[0]} train / {X_test.shape[0]} test, {X_train.shape[1]} features")

    if monitor:
        monitor.log({
            "train/n_samples": int(X_train.shape[0]),
            "train/n_features": int(X_train.shape[1]),
        })

    # ---- Build LightGBM params ----
    lgb_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "verbose": -1,
    }

    # Resolve GPU settings from platform env + nvidia-smi
    fallback_occurred = False

    platform_gpu_str = os.environ.get("KFJ_TASK_RESOURCE_GPU", "0").replace("+", "").strip()
    try:
        platform_gpu = int(float(platform_gpu_str))
    except ValueError:
        platform_gpu = 0

    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    driver_caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "")

    # OpenCL diagnostics
    opencl_icd_dir = "/etc/OpenCL/vendors"
    opencl_icd_exists = os.path.isdir(opencl_icd_dir) and bool(os.listdir(opencl_icd_dir))
    libopencl_found = False
    for libpath in ["/usr/lib/x86_64-linux-gnu/libOpenCL.so.1", "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
                    "/usr/lib64/libOpenCL.so.1", "/usr/local/cuda/lib64/libOpenCL.so.1"]:
        if os.path.exists(libpath):
            libopencl_found = True
            break
    clinfo_ok = False
    try:
        subprocess.check_output(["clinfo", "-l"], timeout=5, stderr=subprocess.DEVNULL)
        clinfo_ok = True
    except Exception:
        pass

    _info(f"platform GPU request={platform_gpu}")
    _info(f"NVIDIA_VISIBLE_DEVICES={nvidia_visible or '(not set)'}")
    _info(f"NVIDIA_DRIVER_CAPABILITIES={driver_caps or '(not set)'}")
    _info(f"CUDA_VISIBLE_DEVICES={cuda_visible or '(not set)'}")
    _info(f"nvidia-smi: {'OK' if gpu_detected else 'not detected'}")
    _info(f"{opencl_icd_dir}: {'found' if opencl_icd_exists else 'missing'}")
    _info(f"libOpenCL: {'found' if libopencl_found else 'not found'}")
    _info(f"clinfo: {'OK' if clinfo_ok else 'not available'}")

    # Determine device_type:
    # 1) Explicit --device_type compat arg
    # 2) Platform GPU申请>0 + nvidia-smi OK → gpu (OpenCL)
    # 3) Otherwise → cpu
    if args.device_type and args.device_type in ("gpu", "cuda"):
        device_type = args.device_type
    elif platform_gpu > 0 and gpu_detected:
        device_type = "gpu"   # OpenCL GPU backend, NOT cuda
    else:
        device_type = "cpu"

    _info(f"resolved device_type={device_type}")

    if device_type in ("gpu", "cuda") and not gpu_detected:
        if args.allow_gpu_fallback == "true":
            _warn(f"GPU not available, falling back to CPU")
            device_type = "cpu"
            fallback_occurred = True
        else:
            raise RuntimeError(
                f"device_type={device_type} but nvidia-smi not detected. "
                f"GPU may not be allocated to this Pod."
            )

    if device_type in ("gpu", "cuda"):
        _info(f"GPU mode: device={device_type}, device_id={args.gpu_device_id}")
        lgb_params["device_type"] = device_type
        lgb_params["gpu_platform_id"] = args.gpu_platform_id
        lgb_params["gpu_device_id"] = args.gpu_device_id
        lgb_params["max_bin"] = args.max_bin
        if args.gpu_use_dp == "true":
            lgb_params["gpu_use_dp"] = True

    if monitor:
        monitor.log({
            "gpu/requested": int(platform_gpu),
            "gpu/nvidia_smi_ok": 1 if gpu_detected else 0,
            "gpu/opencl_available": 1 if (libopencl_found and opencl_icd_exists) else 0,
            "gpu/fallback": 1 if fallback_occurred else 0,
            "config/device_type_str": str(device_type),
            "config/n_estimators": args.n_estimators,
            "config/learning_rate": args.learning_rate,
            "config/num_leaves": args.num_leaves,
            "config/max_depth": args.max_depth,
            "config/test_size": args.test_size,
            "gpu/enabled": device_type in ("gpu", "cuda"),
            "gpu/fallback": fallback_occurred,
        })

    # ---- Train ----
    _info(f"Training with device={device_type}...")
    try:
        if args.task_type == "classification":
            model = LGBMClassifier(**lgb_params)
        else:
            model = LGBMRegressor(**lgb_params)

        model.fit(X_train, y_train)
    except Exception as e:
        _info(f"LightGBM fit failed: {e}")
        err_msg = str(e)
        if device_type in ("gpu", "cuda") and args.allow_gpu_fallback == "true":
            _warn(f"GPU training failed ({e}), retrying with CPU...")
            lgb_params.pop("device_type", None)
            lgb_params.pop("gpu_platform_id", None)
            lgb_params.pop("gpu_device_id", None)
            lgb_params.pop("gpu_use_dp", None)
            device_type = "cpu"
            fallback_occurred = True
            if args.task_type == "classification":
                model = LGBMClassifier(**lgb_params)
            else:
                model = LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train)
            if monitor:
                monitor.log({"gpu/fallback": 1, "gpu/enabled": 0, "gpu/train_success": 0})
        else:
            if monitor and device_type in ("gpu", "cuda"):
                is_opencl_err = "opencl" in err_msg.lower() or "no opencl" in err_msg.lower()
                monitor.log({
                    "gpu/train_success": 0,
                    "gpu/opencl_available": 0 if is_opencl_err else 1,
                    "error/opencl_missing": 1 if is_opencl_err else 0,
                })
            raise

    _info("Training completed")

    # GPU training succeeded: override OpenCL diagnostic
    # (libOpenCL may not show in static checks but works via nvidia-container-toolkit)
    if device_type in ("gpu", "cuda") and monitor:
        monitor.log({
            "gpu/opencl_available": 1,
            "gpu/train_success": 1,
        })

    # ---- Predict & evaluate ----
    pred = model.predict(X_test)

    if args.task_type == "classification":
        pred_int = pred.astype(int) if hasattr(pred, "astype") else pred
        metrics = {
            "metrics/accuracy": float(accuracy_score(y_test, pred_int)),
            "metrics/f1_macro": float(f1_score(y_test, pred_int, average="macro", zero_division=0)),
            "metrics/f1_weighted": float(f1_score(y_test, pred_int, average="weighted", zero_division=0)),
        }
    else:
        metrics = {
            "metrics/r2": float(r2_score(y_test, pred)),
            "metrics/mae": float(mean_absolute_error(y_test, pred)),
            "metrics/mse": float(mean_squared_error(y_test, pred)),
            "metrics/rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        }

    _info(f"Metrics: {json.dumps(metrics, ensure_ascii=False)}")
    if monitor:
        monitor.log(metrics)

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output_model_path) if os.path.dirname(args.output_model_path) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metrics_path) if os.path.dirname(args.output_metrics_path) else ".", exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "label_col": args.label_col,
        "task_type": args.task_type,
        "device_type": device_type,
        "lgb_params": {k: v for k, v in lgb_params.items() if not k.startswith("_")},
    }
    joblib.dump(artifact, args.output_model_path)

    report = {
        "task_type": args.task_type,
        "device_type": device_type,
        "gpu_detected": gpu_detected,
        "fallback_occurred": fallback_occurred,
        "lgb_params": {str(k): str(v) for k, v in lgb_params.items() if not k.startswith("_")},
        "metrics": metrics,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "label_col": args.label_col,
        "input_csv": args.input_csv,
    }
    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _info("========== LightGBM Operator Finished ==========")
    _info(f"model: {args.output_model_path}")
    _info(f"metrics: {args.output_metrics_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if monitor:
        monitor.finish("SUCCEEDED")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[lightgbm] ERROR: {type(exc).__name__}: {exc}", flush=True, file=sys.stderr)
        traceback.print_exc()
        # Try to notify monitor of failure
        try:
            from common.training_monitor import TrainingMonitor
            m = TrainingMonitor()
            m.log({"error": str(exc)[:200]})
            m.finish("FAILED")
        except Exception:
            pass
        sys.exit(1)
