#!/usr/bin/env python3
"""Hyperparameter Search with real-time per-trial SwanLab monitoring + GPU support."""

import argparse, json, os, sys, subprocess, time
from datetime import datetime

import joblib, numpy as np, pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler,
    KFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, SVR

from common.training_monitor import TrainingMonitor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

SUPPORTED_TASK_TYPES = {"classification", "regression"}
SUPPORTED_SEARCH_METHODS = {"grid", "random"}
MODEL_TYPES = [
    "gbdt", "random_forest", "extra_trees", "ada_boost",
    "svm", "logistic_regression", "ridge",
    "xgboost", "xgboost_gpu", "lightgbm",
]
CLF_ONLY = {"logistic_regression"}
REG_ONLY = {"ridge"}
GPU_MODELS = {"xgboost_gpu"}


def detect_gpu():
    """返回统一 GPU 检测字段。gpu_available=True 表示任意方式确认 GPU 可用。"""
    r = {
        "gpu_available": False,
        "detected_gpu_count": 0,
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "",
        "nvidia_smi_ok": False,
        "gpus": [],
    }
    r["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    r["nvidia_visible_devices"] = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    env_gpu = False
    for v in (r["cuda_visible_devices"], r["nvidia_visible_devices"]):
        if v and v not in ("", "void", "none", "-1"):
            env_gpu = True
            break
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total",
                                        "--format=csv,noheader,nounits"], timeout=10)
        lines = [l.strip() for l in out.decode().strip().split("\n") if l.strip()]
        r["detected_gpu_count"] = len(lines)
        r["nvidia_smi_ok"] = True
        for line in lines:
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                r["gpus"].append({"index": p[0], "name": p[1], "memory_mb": p[2]})
    except Exception:
        pass
    # gpu_available 只要 nvidia-smi 检测到 GPU 或环境变量有 GPU 即为可用
    r["gpu_available"] = (r["detected_gpu_count"] > 0 and r["nvidia_smi_ok"]) or env_gpu
    # 向后兼容旧字段别名
    r["gpu_count"] = r["detected_gpu_count"]
    r["gp"] = r["gpu_available"]
    r["nvidia_smi"] = r["nvidia_smi_ok"]
    return r


def build_estimator(model_type, task_type, random_state, gp_info=None):
    mt = model_type.lower()

    if mt == "xgboost_gpu":
        if not _HAS_XGB:
            raise ImportError("xgboost 未安装")
        if not gp_info or not gp_info["gpu_available"]:
            detail = "gp_info=None"
            if gp_info:
                detail = (f"requested_gpu={gp_info.get('requested_gpu',1)}, "
                          f"detected_gpu_count={gp_info['detected_gpu_count']}, "
                          f"nvidia_smi_ok={gp_info['nvidia_smi_ok']}, "
                          f"cuda_visible={gp_info['cuda_visible_devices']!r}, "
                          f"nvidia_visible={gp_info['nvidia_visible_devices']!r}")
            raise RuntimeError(f"model_type=xgboost_gpu 需要 GPU 资源，但 GPU 不可用: {detail}")
        import xgboost as xgb
        ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
        if ver >= (2, 1):
            kw = {"tree_method": "hist", "device": "cuda"}
        elif ver >= (1, 5):
            kw = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
        else:
            kw = {"tree_method": "gpu_hist"}
        log(f"[GPU] xgboost {xgb.__version__} tree_method={kw.get('tree_method')} device={kw.get('device','N/A')}")
        log(f"[GPU] detected GPUs={gp_info['detected_gpu_count']}")
        for g in gp_info.get("gpus", gp_info.get("gpu_info", [])):
            log(f"[GPU]   GPU {g['index']}: {g['name']} ({g['memory_mb']} MB)")
        if task_type == "classification":
            return XGBClassifier(random_state=random_state, eval_metric="logloss", **kw)
        return XGBRegressor(random_state=random_state, **kw)

    if task_type == "classification":
        if mt in REG_ONLY: raise ValueError(f"{model_type} 不支持 classification")
        if mt == "gbdt":     return GradientBoostingClassifier(random_state=random_state)
        if mt == "random_forest": return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees":   return ExtraTreesClassifier(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost":     return AdaBoostClassifier(random_state=random_state)
        if mt == "svm":      return SVC(probability=True, random_state=random_state)
        if mt == "logistic_regression": return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB: raise ImportError("xgboost 未安装")
            return XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB: raise ImportError("lightgbm 未安装")
            return LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
    else:
        if mt in CLF_ONLY: raise ValueError(f"{model_type} 不支持 regression")
        if mt == "gbdt":     return GradientBoostingRegressor(random_state=random_state)
        if mt == "random_forest": return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees":   return ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost":     return AdaBoostRegressor(random_state=random_state)
        if mt == "svm":      return SVR()
        if mt == "ridge":    return Ridge(random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB: raise ImportError("xgboost 未安装")
            return XGBRegressor(random_state=random_state, n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB: raise ImportError("lightgbm 未安装")
            return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    raise ValueError(f"未知 model_type: {model_type}")


DEFAULT_PARAM_GRIDS = {
    "gbdt": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
             "model__max_depth": [2, 3, 5], "model__subsample": [0.8, 1.0]},
    "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10],
                      "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]},
    "extra_trees": {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10],
                    "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]},
    "ada_boost": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "svm": {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0], "model__solver": ["lbfgs", "liblinear"]},
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "xgboost": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 5, 7], "model__subsample": [0.8, 1.0]},
    "xgboost_gpu": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7], "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0]},
    "lightgbm": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.05, 0.1],
                 "model__num_leaves": [15, 31, 63], "model__max_depth": [-1, 5, 10]},
}


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if obj is None: return None
    return obj

def make_ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def parse_param_grid(raw):
    if not raw or not raw.strip(): return {}
    try: d = json.loads(raw)
    except Exception as e: raise ValueError(f"param_grid JSON 解析失败: {e}")
    if not isinstance(d, dict): raise ValueError("param_grid 必须是 JSON 对象")
    return {("model__" + k if not k.startswith("model__") else k): v for k, v in d.items()}


def parse_args():
    p = argparse.ArgumentParser("Multi-model hyperparameter search with GPU + SwanLab")
    p.add_argument("--input_csv", default="")
    p.add_argument("--label_col", default="label")
    p.add_argument("--task_type", default="classification", choices=sorted(SUPPORTED_TASK_TYPES))
    p.add_argument("--model_type", default="gbdt", choices=MODEL_TYPES)
    p.add_argument("--search_method", default="random", choices=sorted(SUPPORTED_SEARCH_METHODS))
    p.add_argument("--param_grid", default="")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--scoring", default="auto")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--output_model_path", default="")
    p.add_argument("--output_metrics_path", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 手动 trial 循环核心
# ---------------------------------------------------------------------------

def _extract_model_params(param_dict):
    """从 model__xxx 格式提取模型参数名。"""
    return {k.replace("model__", ""): v for k, v in param_dict.items()}


def run_trial(pipe, params, X_train, y_train, cv_splitter, scoring, task_type):
    """
    执行一次 trial：对给定参数做 CV，返回 fold scores 和 mean score。
    """
    pipe = clone(pipe)
    pipe.set_params(**params)

    fold_scores = []
    for fi, (ti, vi) in enumerate(cv_splitter.split(X_train, y_train)):
        Xt, Xv = X_train.iloc[ti], X_train.iloc[vi]
        yt, yv = y_train[ti], y_train[vi]
        pipe.fit(Xt, yt)
        yp = pipe.predict(Xv)

        if task_type == "classification":
            if scoring == "accuracy":
                s = accuracy_score(yv, yp)
            elif scoring == "f1_macro":
                s = f1_score(yv, yp, average="macro", zero_division=0)
            else:
                s = accuracy_score(yv, yp)
        else:
            if scoring == "r2":
                s = r2_score(yv, yp)
            elif scoring == "neg_mean_squared_error":
                s = -mean_squared_error(yv, yp)
            else:
                s = r2_score(yv, yp)
        fold_scores.append((fi, s))

    mean_score = float(np.mean([fs[1] for fs in fold_scores]))
    std_score = float(np.std([fs[1] for fs in fold_scores]))
    return fold_scores, mean_score, std_score


def main():
    args = parse_args()
    monitor = TrainingMonitor()
    monitor.start()

    # Metric filtering is now handled centrally by TrainingMonitor.log()
    # based on SWANLAB_METRIC_LEVEL env var (default: basic)

    log("========== Hyperparameter Search Start ==========")
    log(f"model_type={args.model_type} task_type={args.task_type} search_method={args.search_method}")
    for required in [("input_csv", args.input_csv), ("output_model_path", args.output_model_path),
                     ("output_metrics_path", args.output_metrics_path)]:
        if not required[1]:
            log(f"ERROR: {required[0]} 不能为空"); monitor.finish("FAILED"); return 1

    if not os.path.isfile(args.input_csv):
        log(f"ERROR: input_csv 不存在: {args.input_csv}"); monitor.finish("FAILED"); return 1
    for p in [args.output_model_path, args.output_metrics_path]:
        d = os.path.dirname(os.path.abspath(p))
        if d: os.makedirs(d, exist_ok=True)

    # GPU detection
    gp_info = detect_gpu()
    rg = os.environ.get("KFJ_TASK_RESOURCE_GPU", "0").replace("+", "").strip()
    try: rg_val = float(rg)
    except ValueError: rg_val = 0
    gp_info["requested_gpu"] = int(rg_val)
    # GPU status logged as info, not charted as metrics
    log(f"[GPU] requested={gp_info['requested_gpu']}, detected={gp_info['detected_gpu_count']}, "
        f"available={gp_info['gpu_available']}, nvidia_smi={gp_info['nvidia_smi_ok']}")

    if args.model_type in GPU_MODELS:
        if not gp_info["gpu_available"]:
            msg = (f"model_type={args.model_type} 需要 GPU>=1，"
                   f"requested={gp_info['requested_gpu']}, "
                   f"detected={gp_info['detected_gpu_count']}, "
                   f"nvidia_smi={gp_info['nvidia_smi_ok']}, "
                   f"CUDA_VISIBLE={gp_info['cuda_visible_devices']!r}, "
                   f"NVIDIA_VISIBLE={gp_info['nvidia_visible_devices']!r}")
            log(f"ERROR: {msg}")
            monitor.log({"error": msg})
            monitor.finish("FAILED")
            return 1
        log(f"[GPU] requested={gp_info['requested_gpu']}, detected={gp_info['detected_gpu_count']}, "
            f"available={gp_info['gpu_available']}, nvidia_smi={gp_info['nvidia_smi_ok']}")

    # Load data
    df = pd.read_csv(args.input_csv)
    if df.empty: log("ERROR: CSV 为空"); monitor.finish("FAILED"); return 1
    if args.label_col not in df.columns:
        log(f"ERROR: label_col '{args.label_col}' 不存在"); monitor.finish("FAILED"); return 1
    df = df.dropna(subset=[args.label_col])
    if df.empty: log("ERROR: 删除空标签后数据为空"); monitor.finish("FAILED"); return 1
    log(f"数据: {len(df)} 行, {len(df.columns)} 列")

    y_raw = df[args.label_col]
    X = df.drop(columns=[args.label_col])
    if args.task_type == "classification":
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna(); X = X.loc[valid]; y = y.loc[valid]
    if len(X) < 2: log("ERROR: 有效样本不足"); monitor.finish("FAILED"); return 1

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    strat = y if args.task_type == "classification" and len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=strat)

    # Reset index for iloc-based CV splitting
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = np.array(y_train) if hasattr(y_train, "reset_index") else y_train
    if hasattr(y_train, "reset_index"):
        y_train = y_train.reset_index(drop=True)
    y_train = np.array(y_train)

    monitor.log({
        "dataset/train_samples": len(X_train), "dataset/test_samples": len(X_test),
        "dataset/num_features": len(X.columns),
    })

    # Build preprocessor + estimator
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("ohe", make_ohe())]), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = build_estimator(args.model_type, args.task_type, args.random_state, gp_info)
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    # Param grid
    user_grid = parse_param_grid(args.param_grid)
    param_grid = user_grid if user_grid else DEFAULT_PARAM_GRIDS.get(args.model_type, DEFAULT_PARAM_GRIDS["gbdt"])
    scoring = args.scoring if args.scoring != "auto" else ("accuracy" if args.task_type == "classification" else "r2")

    # Generate param combos
    if args.search_method == "grid":
        combos = list(ParameterGrid(param_grid))
    else:
        combos = list(ParameterSampler(param_grid, n_iter=min(args.n_iter, len(ParameterGrid(param_grid))),
                                        random_state=args.random_state))
    n_trials = len(combos)
    log(f"搜索空间: {n_trials} 个组合, cv={args.cv}, scoring={scoring}")

    # CV splitter
    if args.task_type == "classification" and len(set(y_train)) >= args.cv:
        cv_splitter = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    else:
        cv_splitter = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

    monitor.log({
        "config/cv": args.cv,
        "config/n_trials": n_trials,
    })

    # ---- Manual trial loop ----
    best_score = -float("inf")
    best_params = None
    best_idx = -1
    trial_records = []

    for ti, params in enumerate(combos):
        t0 = time.time()
        fold_scores, mean_score, std_score = run_trial(
            pipe, params, X_train, y_train, cv_splitter, scoring, args.task_type)
        elapsed = time.time() - t0

        is_best = mean_score > best_score
        if is_best:
            best_score = mean_score
            best_params = params
            best_idx = ti

        # Build metrics dict
        mp = _extract_model_params(params)
        m = {
            "trial/index": ti, "trial/score": mean_score, "trial/std_score": std_score,
            "trial/duration_seconds": round(elapsed, 2),
            "best/score": best_score, "best/trial_index": best_idx,
        }
        for pk, pv in mp.items():
            m[f"params/{pk}"] = float(pv) if isinstance(pv, (int, float)) else str(pv)
        for fi, fs in fold_scores:
            m[f"fold/{fi}/score"] = fs

        # Task-specific
        if args.task_type == "classification":
            m["trial/accuracy"] = mean_score
            m["trial/f1_macro"] = mean_score  # approximation from CV mean
        else:
            m["trial/r2"] = mean_score
            m["trial/rmse"] = np.sqrt(-mean_score) if scoring == "neg_mean_squared_error" else mean_score

        monitor.log(m, step=ti)
        trial_records.append({"params": mp, "mean_score": mean_score, "std_score": std_score, "is_best": is_best})

        log(f"[Trial {ti+1}/{n_trials}] score={mean_score:.6f} std={std_score:.6f} "
            f"best={best_score:.6f} (trial {best_idx}) elapsed={elapsed:.1f}s "
            f"params={mp}")

    log(f"搜索完成: best_trial={best_idx} best_score={best_score:.6f} best_params={best_params}")

    # Retrain best model on full training set
    log("Retraining best model on full training data...")
    best_pipe = clone(pipe)
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_train, y_train)

    # Test set evaluation
    y_pred = best_pipe.predict(X_test)
    if args.task_type == "classification":
        tm = {
            "test/accuracy": float(accuracy_score(y_test, y_pred)),
            "test/f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "test/f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test/precision_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "test/recall_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        tm = {
            "test/rmse": float(np.sqrt(mse)), "test/mae": float(mean_absolute_error(y_test, y_pred)),
            "test/r2": float(r2_score(y_test, y_pred)),
        }
    monitor.log(tm)
    bp_metrics = {"best/cv_score": best_score}
    for pk, pv in _extract_model_params(best_params).items():
        if isinstance(pv, (int, float)):
            bp_metrics[f"best_params/{pk}"] = float(pv)
    monitor.log(bp_metrics)

    # Save model + metrics
    artifact = {"best_estimator": best_pipe, "best_params": best_params,
                "best_cv_score": best_score, "search_method": args.search_method,
                "task_type": args.task_type, "model_type": args.model_type,
                "label_col": args.label_col, "feature_columns": list(X.columns),
                "trial_records": trial_records}
    joblib.dump(artifact, args.output_model_path)
    log(f"模型已保存: {args.output_model_path}")

    report = {"task_type": args.task_type, "model_type": args.model_type, "search_method": args.search_method,
              "best_params": {str(k): json_safe(v) for k, v in best_params.items()},
              "best_cv_score": float(best_score), "scoring": scoring, "cv": args.cv,
              "n_trials": n_trials, "n_iter": args.n_iter if args.search_method == "random" else None,
              "test_metrics": tm, "label_col": args.label_col, "input_csv": args.input_csv,
              "n_samples": len(X), "n_features": len(X.columns)}
    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)
    log(f"指标已保存: {args.output_metrics_path}")

    log("========== Hyperparameter Search Finished ==========")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)

    monitor.finish("SUCCEEDED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"任务执行失败: {type(exc).__name__}: {exc}")
        traceback = __import__("traceback")
        traceback.print_exc()
        raise
