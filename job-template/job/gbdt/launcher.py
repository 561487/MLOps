#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBDT training launcher for MLOps job-template.

Flow:
1. Read CSV from a mounted path.
2. Split train/test dataset.
3. Build preprocessing + GradientBoosting model pipeline.
4. Train model.
5. Save model.pkl and metrics.json.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SUPPORTED_TASK_TYPES = {"classification", "regression"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GBDT model with scikit-learn.")

    # Data config
    parser.add_argument("--input_csv", type=str, default="", help="Input CSV path in container.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name.")
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=sorted(SUPPORTED_TASK_TYPES),
        help="Task type: classification or regression.",
    )
    parser.add_argument(
        "--drop_cols",
        type=str,
        default="",
        help="Comma-separated columns to exclude from features, e.g. id,date.",
    )

    # Split config
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio, range (0, 1).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")

    # GBDT model params
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting stages.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of each regression tree.")
    parser.add_argument("--subsample", type=float, default=1.0, help="Sample fraction for stochastic gradient boosting.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples required to split a node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum samples required at a leaf node.")
    parser.add_argument(
        "--max_features",
        type=str,
        default="",
        help="Max features for each split. Empty means None; supports int, float, sqrt, log2.",
    )

    # Output config
    parser.add_argument("--output_model_path", type=str, default="", help="Output model.pkl path in container.")
    parser.add_argument("--output_metrics_path", type=str, default="", help="Output metrics.json path in container.")

    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_file_exists(path: str, arg_name: str) -> None:
    if not path:
        raise ValueError(f"{arg_name} 不能为空，请填写容器内可访问路径，例如 /mnt/storage/models-storage/...")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{arg_name} 文件不存在: {path}")


def ensure_parent_dir(path: str, arg_name: str) -> None:
    if not path:
        raise ValueError(f"{arg_name} 不能为空，请填写容器内可写路径，例如 /mnt/storage/models-storage/...")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_drop_cols(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_max_features(value: str) -> Optional[Any]:
    value = (value or "").strip()
    if value == "":
        return None
    if value.lower() in {"none", "null"}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    try:
        if "." in value:
            parsed = float(value)
            if parsed <= 0:
                raise ValueError
            return parsed
        parsed_int = int(value)
        if parsed_int <= 0:
            raise ValueError
        return parsed_int
    except ValueError as exc:
        raise ValueError("max_features 仅支持空值、sqrt、log2、正整数或正小数") from exc


def make_one_hot_encoder() -> OneHotEncoder:
    # sklearn >= 1.2 uses sparse_output; older versions use sparse.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, set, np.ndarray)) else False:
        return None
    return obj


def validate_args(args: argparse.Namespace) -> None:
    if args.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task_type 仅支持: {sorted(SUPPORTED_TASK_TYPES)}")
    if not 0 < args.test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if args.n_estimators <= 0:
        raise ValueError("n_estimators 必须为正整数")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if args.max_depth <= 0:
        raise ValueError("max_depth 必须为正整数；GBDT 不建议使用 -1")
    if not 0 < args.subsample <= 1:
        raise ValueError("subsample 必须在 (0, 1] 范围内")
    if args.min_samples_split < 2:
        raise ValueError("min_samples_split 必须 >= 2")
    if args.min_samples_leaf < 1:
        raise ValueError("min_samples_leaf 必须 >= 1")


def build_pipeline(
    task_type: str,
    numeric_features: List[str],
    categorical_features: List[str],
    model_params: Dict[str, Any],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("没有可用特征列，请检查 label_col 和 drop_cols 配置")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if task_type == "classification":
        model = GradientBoostingClassifier(**model_params)
    else:
        model = GradientBoostingRegressor(**model_params)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def choose_stratify_y(y: pd.Series, task_type: str, test_size: float) -> Optional[pd.Series]:
    if task_type != "classification":
        return None
    value_counts = y.value_counts(dropna=False)
    if len(value_counts) < 2:
        raise ValueError("分类任务至少需要 2 个类别")
    # train_test_split with stratify requires every class to have at least 2 samples.
    if value_counts.min() < 2:
        log("检测到部分类别样本数少于 2，关闭 stratify 切分。")
        return None
    # Conservative guard: both train and test should be able to contain each class.
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    if test_count < len(value_counts) or train_count < len(value_counts):
        log("测试集或训练集容量不足以覆盖所有类别，关闭 stratify 切分。")
        return None
    return y


def main() -> int:
    args = parse_args()
    validate_args(args)

    ensure_file_exists(args.input_csv, "input_csv")
    ensure_parent_dir(args.output_model_path, "output_model_path")
    ensure_parent_dir(args.output_metrics_path, "output_metrics_path")

    max_features = parse_max_features(args.max_features)
    model_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "max_features": max_features,
    }

    log(f"读取数据: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError("输入 CSV 为空")
    if args.label_col not in df.columns:
        raise ValueError(f"label_col 不存在: {args.label_col}; 当前列: {list(df.columns)}")

    before_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[args.label_col])
    dropped_label_rows = before_rows - len(df)
    if df.empty:
        raise ValueError("去除空标签后数据为空")

    drop_cols = parse_drop_cols(args.drop_cols)
    effective_drop_cols = [c for c in drop_cols if c in df.columns and c != args.label_col]

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col] + effective_drop_cols)

    if args.task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        if len(y) < 2:
            raise ValueError("回归任务有效标签样本数不足")
    else:
        y = y.astype(str)

    feature_columns = list(X.columns)
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    if len(X) < 2:
        raise ValueError("样本数不足，至少需要 2 行有效数据")

    stratify_y = choose_stratify_y(y, args.task_type, args.test_size)
    log(
        f"数据形状: rows={len(X)}, features={len(feature_columns)}, "
        f"numeric={len(numeric_features)}, categorical={len(categorical_features)}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_y,
    )

    pipeline = build_pipeline(
        task_type=args.task_type,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )

    log("开始训练 GBDT 模型")
    pipeline.fit(X_train, y_train)
    log("模型训练完成，开始评估")

    y_pred = pipeline.predict(X_test)
    if args.task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "classes": sorted(pd.Series(y).unique().tolist()),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

    artifact = {
        "pipeline": pipeline,
        "task_type": args.task_type,
        "label_col": args.label_col,
        "drop_cols": effective_drop_cols,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_params": model_params,
    }

    log(f"保存模型: {args.output_model_path}")
    joblib.dump(artifact, args.output_model_path)

    report = {
        "status": "success",
        "algorithm": "GBDT",
        "framework": "scikit-learn",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": args.input_csv,
        "output_model_path": args.output_model_path,
        "output_metrics_path": args.output_metrics_path,
        "dataset": {
            "rows_before_drop_label": before_rows,
            "rows_after_drop_label": len(df),
            "dropped_label_rows": dropped_label_rows,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        "params": {
            "task_type": args.task_type,
            "label_col": args.label_col,
            "drop_cols": effective_drop_cols,
            "test_size": args.test_size,
            "random_state": args.random_state,
            **model_params,
        },
        "metrics": metrics,
    }

    log(f"保存指标: {args.output_metrics_path}")
    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)

    log("GBDT 任务执行成功")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"任务执行失败: {type(exc).__name__}: {exc}")
        raise
