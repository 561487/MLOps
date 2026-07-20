#!/usr/bin/env python3
"""GBDT training launcher — sklearn CPU + XGBoost CUDA GPU + SwanLab Cloud monitoring."""

import argparse, json, os, sys, subprocess, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib, numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

SUPPORTED_TASK_TYPES = {"classification", "regression"}
DEVICE_CHOICES = {"auto", "cpu", "cuda"}
GPU_BACKEND = "xgboost"  # XGBoost CUDA GPU, same as hyperparam-search xgboost_gpu


# ── GPU detection ──
def detect_gpu():
    try:
        subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total",
                                 "--format=csv,noheader,nounits"], timeout=10)
        return True
    except Exception:
        return False


def resolve_device(requested, resource_gpu=0):
    gpu_ok = detect_gpu()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not gpu_ok or resource_gpu <= 0:
            raise RuntimeError(
                f"device=cuda but GPU not available (nvidia-smi={gpu_ok}, resource_gpu={resource_gpu})")
        return "cuda"
    # auto
    if gpu_ok and resource_gpu > 0:
        return "cuda"
    return "cpu"


# ── Args ──
def parse_args():
    p = argparse.ArgumentParser(description="GBDT training (sklearn CPU / LightGBM GPU)")
    p.add_argument("--input_csv", default="", help="CSV path")
    p.add_argument("--label_col", default="label")
    p.add_argument("--task_type", default="classification", choices=sorted(SUPPORTED_TASK_TYPES))
    p.add_argument("--drop_cols", default="")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--max_features", default="")
    p.add_argument("--device", default="auto", choices=sorted(DEVICE_CHOICES),
                   help="auto|cpu|cuda")
    p.add_argument("--output_model_path", default="")
    p.add_argument("--output_metrics_path", default="")
    return p.parse_args()


# ── Helpers ──
def _info(msg): print(f"[gbdt] INFO: {msg}", flush=True)
def _warn(msg): print(f"[gbdt] WARNING: {msg}", flush=True, file=sys.stderr)

def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

def _ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def ensure_file(path, name):
    if not path: raise ValueError(f"{name} 不能为空")
    if not os.path.isfile(path): raise FileNotFoundError(f"{name} 不存在: {path}")

def ensure_dir(path, name):
    if not path: raise ValueError(f"{name} 不能为空")
    d = os.path.dirname(os.path.abspath(path))
    if d: os.makedirs(d, exist_ok=True)


# ── Main ──
def main():
    args = parse_args()

    # SwanLab monitor
    try:
        from common.training_monitor import TrainingMonitor
        monitor = TrainingMonitor()
        monitor.start()
    except Exception:
        monitor = None

    _info(f"========== GBDT Start ==========")
    _info(f"input={args.input_csv}  task={args.task_type}  device={args.device}")

    # Resolve device
    gpu_ok = detect_gpu()
    rg = os.environ.get("KFJ_TASK_RESOURCE_GPU", "0").replace("+", "").strip()
    try: rg_val = int(float(rg))
    except ValueError: rg_val = 0
    _info(f"nvidia-smi: {'OK' if gpu_ok else 'not detected'},  resource_gpu={rg_val}")
    actual_device = resolve_device(args.device, rg_val)
    _info(f"resolved device={actual_device}")

    try:
        ensure_file(args.input_csv, "input_csv")
        ensure_dir(args.output_model_path, "output_model_path")
        ensure_dir(args.output_metrics_path, "output_metrics_path")

        # Load data
        df = pd.read_csv(args.input_csv)
        if df.empty: raise ValueError("CSV 为空")
        if args.label_col not in df.columns:
            raise ValueError(f"label_col 不存在: {args.label_col}")

        before_rows = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.label_col])
        drop_cols = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]
        y = df[args.label_col]
        X = df.drop(columns=[args.label_col] + [c for c in drop_cols if c in df.columns and c != args.label_col])

        if args.task_type == "regression":
            y = pd.to_numeric(y, errors="coerce")
            mask = y.notna(); X = X.loc[mask]; y = y.loc[mask]

        feats = list(X.columns); num_feats = list(X.select_dtypes(include=[np.number]).columns)
        cat_feats = [c for c in feats if c not in num_feats]

        stratify_y = y if args.task_type == "classification" and len(set(y)) > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify_y)

        _info(f"data: {len(X_train)} train / {len(X_test)} test, {len(feats)} features")

        # Config log
        if monitor:
            monitor.log({
                "config/model_type": "gbdt",
                "config/task_type": args.task_type,
                "config/requested_device": args.device,
                "config/actual_device": actual_device,
                "config/n_estimators": args.n_estimators,
                "config/learning_rate": args.learning_rate,
                "config/max_depth": args.max_depth,
                "config/train_samples": len(X_train),
                "config/test_samples": len(X_test),
                "config/feature_count": len(feats),
                "gpu/requested": rg_val,
                "gpu/nvidia_smi_ok": 1 if gpu_ok else 0,
            })

        # Preprocessor
        transformers = []
        if num_feats: transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_feats))
        if cat_feats: transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe())]), cat_feats))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Build model
        max_features = args.max_features.strip()
        if max_features in ("", "none", "null"): max_features = None
        elif max_features in ("sqrt", "log2"): pass
        else:
            try: max_features = float(max_features) if "." in max_features else int(max_features)
            except ValueError: max_features = None

        if actual_device == "cuda":
            # GPU: XGBoost CUDA, same pattern as hyperparam-search xgboost_gpu
            if not _HAS_XGB:
                raise ImportError("xgboost not installed (required for GPU)")
            import xgboost as xgb
            xgb_ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            if xgb_ver >= (2, 1):
                xgb_kw = {"tree_method": "hist", "device": "cuda"}
            elif xgb_ver >= (1, 5):
                xgb_kw = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
            else:
                xgb_kw = {"tree_method": "gpu_hist"}
            _info(f"GBDT GPU backend: {GPU_BACKEND} (xgboost {xgb.__version__}, {xgb_kw})")
            if args.task_type == "classification":
                model = XGBClassifier(random_state=args.random_state, n_estimators=args.n_estimators,
                                      learning_rate=args.learning_rate, max_depth=args.max_depth,
                                      subsample=args.subsample, eval_metric="logloss", **xgb_kw)
            else:
                model = XGBRegressor(random_state=args.random_state, n_estimators=args.n_estimators,
                                     learning_rate=args.learning_rate, max_depth=args.max_depth,
                                     subsample=args.subsample, **xgb_kw)
        else:
            model_params = {
                "n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
                "max_depth": args.max_depth, "subsample": args.subsample,
                "min_samples_split": args.min_samples_split, "min_samples_leaf": args.min_samples_leaf,
                "random_state": args.random_state, "max_features": max_features,
            }
            if args.task_type == "classification":
                model = GradientBoostingClassifier(**model_params)
            else:
                model = GradientBoostingRegressor(**model_params)
            _info("GBDT CPU backend: sklearn GradientBoosting")

        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        _info("Training...")
        pipe.fit(X_train, y_train)
        _info("Training completed")

        if actual_device == "cuda" and monitor:
            monitor.log({"gpu/opencl_available": 1, "gpu/train_success": 1})

        # Eval
        y_pred = pipe.predict(X_test)
        if args.task_type == "classification":
            pred_int = y_pred.astype(int) if hasattr(y_pred, "astype") else y_pred
            metrics = {
                "metrics/accuracy": float(accuracy_score(y_test, pred_int)),
                "metrics/f1_macro": float(f1_score(y_test, pred_int, average="macro", zero_division=0)),
                "metrics/f1_weighted": float(f1_score(y_test, pred_int, average="weighted", zero_division=0)),
                "metrics/precision_macro": float(precision_score(y_test, pred_int, average="macro", zero_division=0)),
                "metrics/recall_macro": float(recall_score(y_test, pred_int, average="macro", zero_division=0)),
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                "metrics/r2": float(r2_score(y_test, y_pred)),
                "metrics/mae": float(mean_absolute_error(y_test, y_pred)),
                "metrics/mse": float(mse),
                "metrics/rmse": float(np.sqrt(mse)),
            }
        if monitor: monitor.log(metrics)

        # Save
        artifact = {"pipeline": pipe, "task_type": args.task_type, "label_col": args.label_col,
                    "feature_columns": feats, "backend": "xgboost" if actual_device == "cuda" else "sklearn",
                    "actual_device": actual_device}
        joblib.dump(artifact, args.output_model_path)

        report = {"status": "success", "algorithm": "GBDT",
                  "backend": "xgboost" if actual_device == "cuda" else "sklearn",
                  "requested_device": args.device, "actual_device": actual_device,
                  "created_at": datetime.now().isoformat(timespec="seconds"),
                  "dataset": {"train_rows": len(X_train), "test_rows": len(X_test),
                              "feature_count": len(feats)},
                  "params": {"task_type": args.task_type, "n_estimators": args.n_estimators,
                             "learning_rate": args.learning_rate, "max_depth": args.max_depth},
                  "metrics": json_safe(metrics)}
        with open(args.output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(report), f, ensure_ascii=False, indent=2)

        _info("GBDT SUCCEEDED")
        print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
        if monitor: monitor.finish("SUCCEEDED")
        return 0

    except Exception as e:
        _info(f"GBDT FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        if monitor:
            try: monitor.log({"error": str(e)[:200]}); monitor.finish("FAILED")
            except Exception: pass
        raise


if __name__ == "__main__":
    try: sys.exit(main())
    except Exception: sys.exit(1)
