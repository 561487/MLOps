#!/usr/bin/env python3
"""Hyperparameter Search (GridSearchCV / RandomizedSearchCV) for multi-model support."""

import argparse, json, os, sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib, numpy as np, pandas as pd

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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, SVR

# Optional imports
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

# ---------------------------------------------------------------------------
SUPPORTED_TASK_TYPES = {"classification", "regression"}
SUPPORTED_SEARCH_METHODS = {"grid", "random"}

MODEL_TYPES = [
    "gbdt", "random_forest", "extra_trees", "ada_boost",
    "svm", "logistic_regression", "ridge",
    "xgboost", "lightgbm",
]

CLF_ONLY = {"logistic_regression"}
REG_ONLY = {"ridge"}

# ---------------------------------------------------------------------------
def build_estimator(model_type: str, task_type: str, random_state: int):
    """Factory returning a scikit-learn compatible estimator."""
    mt = model_type.lower()

    if task_type == "classification":
        if mt in REG_ONLY:
            raise ValueError(f"model_type={model_type} 不支持 classification")
        if mt == "gbdt":
            return GradientBoostingClassifier(random_state=random_state)
        if mt == "random_forest":
            return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees":
            return ExtraTreesClassifier(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost":
            return AdaBoostClassifier(random_state=random_state)
        if mt == "svm":
            return SVC(probability=True, random_state=random_state)
        if mt == "logistic_regression":
            return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB:
                raise ImportError("xgboost 未安装，请在镜像中安装 xgboost")
            return XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB:
                raise ImportError("lightgbm 未安装，请在镜像中安装 lightgbm")
            return LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
    else:  # regression
        if mt in CLF_ONLY:
            raise ValueError(f"model_type={model_type} 不支持 regression")
        if mt == "gbdt":
            return GradientBoostingRegressor(random_state=random_state)
        if mt == "random_forest":
            return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees":
            return ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost":
            return AdaBoostRegressor(random_state=random_state)
        if mt == "svm":
            return SVR()
        if mt == "ridge":
            return Ridge(random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB:
                raise ImportError("xgboost 未安装，请在镜像中安装 xgboost")
            return XGBRegressor(random_state=random_state, n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB:
                raise ImportError("lightgbm 未安装，请在镜像中安装 lightgbm")
            return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)

    raise ValueError(f"未知 model_type: {model_type}")


DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "gbdt": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 5],
        "model__subsample": [0.8, 1.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    },
    "extra_trees": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    },
    "ada_boost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    },
    "svm": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    },
    "logistic_regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "xgboost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__max_depth": [-1, 5, 10],
    },
}


# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_file_exists(path, name):
    if not path: raise ValueError(f"{name} 不能为空")
    if not os.path.isfile(path): raise FileNotFoundError(f"{name} 文件不存在: {path}")

def ensure_parent_dir(path, name):
    if not path: raise ValueError(f"{name} 不能为空")
    d = os.path.dirname(os.path.abspath(path))
    if d: os.makedirs(d, exist_ok=True)

def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if obj is None: return None
    return obj

def make_one_hot_encoder():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def parse_param_grid(raw: str) -> Dict[str, List[Any]]:
    if not raw or not raw.strip(): return {}
    try: user_grid = json.loads(raw)
    except json.JSONDecodeError as e: raise ValueError(f"param_grid JSON 解析失败: {e}")
    if not isinstance(user_grid, dict): raise ValueError("param_grid 必须是 JSON 对象")
    out = {}
    for k, v in user_grid.items():
        full = k if k.startswith("model__") else f"model__{k}"
        if not isinstance(v, list): raise ValueError(f"param_grid 中 {k} 的值必须是数组")
        out[full] = v
    return out


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-model hyperparameter search")
    p.add_argument("--input_csv", default="")
    p.add_argument("--label_col", default="label")
    p.add_argument("--task_type", default="classification", choices=sorted(SUPPORTED_TASK_TYPES))
    p.add_argument("--model_type", default="gbdt", choices=MODEL_TYPES, help="模型类型")
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


def main():
    args = parse_args()
    log("========== Hyperparameter Search Start ==========")
    log(f"model_type: {args.model_type}  task_type: {args.task_type}  search_method: {args.search_method}")

    if not args.input_csv: log("ERROR: input_csv 不能为空"); return 1
    if not args.output_model_path: log("ERROR: output_model_path 不能为空"); return 1
    if not args.output_metrics_path: log("ERROR: output_metrics_path 不能为空"); return 1

    ensure_file_exists(args.input_csv, "input_csv")
    ensure_parent_dir(args.output_model_path, "output_model_path")
    ensure_parent_dir(args.output_metrics_path, "output_metrics_path")

    df = pd.read_csv(args.input_csv)
    if df.empty: log("ERROR: CSV 为空"); return 1
    if args.label_col not in df.columns:
        log(f"ERROR: label_col '{args.label_col}' 不存在; 列: {list(df.columns)}"); return 1

    before = len(df)
    df = df.dropna(subset=[args.label_col])
    if df.empty: log("ERROR: 删除空标签后数据为空"); return 1
    log(f"数据行数: {len(df)} (原始 {before})")

    y_raw = df[args.label_col]
    X = df.drop(columns=[args.label_col])
    if args.task_type == "classification":
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna(); X = X.loc[valid]; y = y.loc[valid]
    if len(X) < 2: log("ERROR: 有效样本不足"); return 1

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    strat = y if args.task_type == "classification" and len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=strat)

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_one_hot_encoder())]), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    estimator = build_estimator(args.model_type, args.task_type, args.random_state)
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    user_grid = parse_param_grid(args.param_grid)
    param_grid = user_grid if user_grid else DEFAULT_PARAM_GRIDS.get(args.model_type, DEFAULT_PARAM_GRIDS["gbdt"])
    log(f"搜索空间 ({args.model_type}): {json.dumps({k: list(v) for k, v in param_grid.items()}, ensure_ascii=False, default=str)}")

    scoring = args.scoring if args.scoring != "auto" else ("accuracy" if args.task_type == "classification" else "r2")

    if args.search_method == "grid":
        search = GridSearchCV(pipe, param_grid, cv=args.cv, scoring=scoring, refit=True, n_jobs=-1, verbose=1)
    else:
        search = RandomizedSearchCV(pipe, param_grid, n_iter=args.n_iter, cv=args.cv, scoring=scoring,
                                     refit=True, n_jobs=-1, random_state=args.random_state, verbose=1)

    log("开始超参数搜索 ...")
    search.fit(X_train, y_train)
    log(f"best_cv_score ({scoring}): {search.best_score_:.6f}")

    y_pred = search.best_estimator_.predict(X_test)
    if args.task_type == "classification":
        tm = {"accuracy": float(accuracy_score(y_test, y_pred)),
              "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
              "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))}
    else:
        mse = mean_squared_error(y_test, y_pred)
        tm = {"rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y_test, y_pred)),
              "r2": float(r2_score(y_test, y_pred))}

    artifact = {"best_estimator": search.best_estimator_, "best_params": search.best_params_,
                "best_cv_score": float(search.best_score_), "search_method": args.search_method,
                "task_type": args.task_type, "model_type": args.model_type,
                "label_col": args.label_col, "feature_columns": list(X.columns)}
    joblib.dump(artifact, args.output_model_path)
    log(f"模型已保存: {args.output_model_path}")

    report = {"task_type": args.task_type, "model_type": args.model_type, "search_method": args.search_method,
              "best_params": {str(k): json_safe(v) for k, v in search.best_params_.items()},
              "best_cv_score": float(search.best_score_), "scoring": scoring, "cv": args.cv,
              "n_iter": args.n_iter if args.search_method == "random" else None,
              "test_metrics": tm, "label_col": args.label_col, "input_csv": args.input_csv,
              "output_model_path": args.output_model_path, "n_samples": len(X), "n_features": len(X.columns)}
    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)
    log(f"指标已保存: {args.output_metrics_path}")

    log("========== Hyperparameter Search Finished ==========")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as exc: log(f"任务执行失败: {type(exc).__name__}: {exc}"); raise
