#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision Tree training launcher for the MLOps job-template platform.

The launcher reads a CSV file, builds a preprocessing/model pipeline, trains
and evaluates either a classifier or a regressor, then saves a reusable model
artifact and a JSON metrics report.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SUPPORTED_TASK_TYPES = {"classification", "regression"}
CLASSIFICATION_CRITERIA = {"gini", "entropy", "log_loss"}
REGRESSION_CRITERIA = {"squared_error", "friedman_mse", "absolute_error", "poisson"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree model with scikit-learn."
    )

    # Data configuration
    parser.add_argument("--input_csv", default="", help="训练数据 CSV 路径")
    parser.add_argument("--label_col", default="label", help="标签列名称")
    parser.add_argument(
        "--drop_cols",
        default="",
        help="不参与训练的列，多个列名使用英文逗号分隔",
    )

    # Training configuration
    parser.add_argument(
        "--task_type",
        default="classification",
        choices=sorted(SUPPORTED_TASK_TYPES),
        help="任务类型：classification 或 regression",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")

    # Decision Tree parameters
    parser.add_argument(
        "--criterion",
        default="auto",
        help="分裂标准；auto 会按任务类型选择 gini 或 squared_error",
    )
    parser.add_argument("--splitter", choices=["best", "random"], default="best")
    parser.add_argument("--max_depth", type=int, default=5, help="最大树深度")
    parser.add_argument(
        "--min_samples_split", type=int, default=2, help="节点分裂所需最小样本数"
    )
    parser.add_argument(
        "--min_samples_leaf", type=int, default=1, help="叶子节点最小样本数"
    )
    parser.add_argument(
        "--max_features",
        default="",
        help="为空表示全部特征；支持 sqrt、log2、正整数或 (0,1] 小数",
    )
    parser.add_argument(
        "--class_weight",
        default="",
        help="分类类别权重；为空表示不加权，也可填写 balanced",
    )
    parser.add_argument("--ccp_alpha", type=float, default=0.0, help="剪枝强度")

    # Output configuration
    parser.add_argument("--output_model_path", default="", help="模型输出 .pkl 路径")
    parser.add_argument("--output_metrics_path", default="", help="指标输出 JSON 路径")

    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def parse_comma_separated(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def parse_max_features(value: str) -> Optional[Any]:
    normalized = (value or "").strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized in {"sqrt", "log2"}:
        return normalized

    try:
        if any(marker in normalized for marker in (".", "e")):
            parsed_float = float(normalized)
            if not 0 < parsed_float <= 1:
                raise ValueError
            return parsed_float
        parsed_int = int(normalized)
        if parsed_int < 1:
            raise ValueError
        return parsed_int
    except ValueError as exc:
        raise ValueError(
            "max_features 仅支持空值、sqrt、log2、正整数或 (0,1] 范围的小数"
        ) from exc


def resolve_criterion(task_type: str, criterion: str) -> str:
    normalized = (criterion or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "gini" if task_type == "classification" else "squared_error"

    supported = (
        CLASSIFICATION_CRITERIA
        if task_type == "classification"
        else REGRESSION_CRITERIA
    )
    if normalized not in supported:
        raise ValueError(
            f"{task_type} 不支持 criterion={normalized}; 可选值: {sorted(supported)}"
        )
    return normalized


def resolve_class_weight(task_type: str, class_weight: str) -> Optional[str]:
    normalized = (class_weight or "").strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if task_type != "classification":
        raise ValueError("class_weight 仅适用于 classification 分类任务")
    if normalized != "balanced":
        raise ValueError("class_weight 当前仅支持空值或 balanced")
    return normalized


def validate_args(args: argparse.Namespace) -> None:
    if args.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task_type 仅支持: {sorted(SUPPORTED_TASK_TYPES)}")
    if not 0 < args.test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if args.max_depth < 1:
        raise ValueError("max_depth 必须 >= 1")
    if args.min_samples_split < 2:
        raise ValueError("min_samples_split 必须 >= 2")
    if args.min_samples_leaf < 1:
        raise ValueError("min_samples_leaf 必须 >= 1")
    if args.ccp_alpha < 0:
        raise ValueError("ccp_alpha 必须 >= 0")
    if not args.input_csv:
        raise ValueError("input_csv 不能为空，请填写容器内可访问的训练 CSV 路径")
    if not args.output_model_path:
        raise ValueError("output_model_path 不能为空，请填写模型输出路径")
    if not args.output_metrics_path:
        raise ValueError("output_metrics_path 不能为空，请填写指标输出路径")
    if os.path.abspath(args.output_model_path) == os.path.abspath(args.output_metrics_path):
        raise ValueError("output_model_path 和 output_metrics_path 不能是同一路径")


def ensure_input_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"训练数据文件不存在: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_one_hot_encoder() -> OneHotEncoder:
    # Keep compatibility with scikit-learn versions before and after 1.2.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_pipeline(
    task_type: str,
    numeric_features: List[str],
    categorical_features: List[str],
    model_params: Dict[str, Any],
) -> Pipeline:
    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                )
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent", keep_empty_features=True
                    ),
                ),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("没有可用特征列，请检查 label_col 和 drop_cols")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    if task_type == "classification":
        model = DecisionTreeClassifier(**model_params)
    else:
        regression_params = {
            key: value for key, value in model_params.items() if key != "class_weight"
        }
        model = DecisionTreeRegressor(**regression_params)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def choose_stratify_y(
    y: pd.Series, task_type: str, test_size: float
) -> Optional[pd.Series]:
    if task_type != "classification":
        return None

    value_counts = y.value_counts(dropna=False)
    if len(value_counts) < 2:
        raise ValueError("分类任务至少需要 2 个类别")
    if value_counts.min() < 2:
        log("部分类别样本少于 2 条，无法分层抽样，将使用普通随机切分")
        return None

    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    if test_count < len(value_counts) or train_count < len(value_counts):
        log("训练集或测试集不足以覆盖全部类别，将使用普通随机切分")
        return None
    return y


def classification_metrics(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> Dict[str, Any]:
    predictions = pipeline.predict(X)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
    }

    model = pipeline.named_steps["model"]
    try:
        probabilities = pipeline.predict_proba(X)
        if len(model.classes_) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])
        elif len(model.classes_) > 2 and len(pd.unique(y)) == len(model.classes_):
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="macro",
            )
    except ValueError as exc:
        # AUC is undefined when an evaluation split contains only one class.
        metrics["roc_auc_note"] = f"未计算 AUC: {exc}"

    return metrics


def regression_metrics(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> Dict[str, Any]:
    predictions = pipeline.predict(X)
    mse = mean_squared_error(y, predictions)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y, predictions),
        "r2": r2_score(y, predictions),
    }


def feature_importance_report(pipeline: Pipeline) -> List[Dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    ranked = sorted(
        zip(names, importances), key=lambda item: float(item[1]), reverse=True
    )
    return [
        {"feature": str(name), "importance": float(importance)}
        for name, importance in ranked
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    ensure_input_file(args.input_csv)
    ensure_parent_dir(args.output_model_path)
    ensure_parent_dir(args.output_metrics_path)

    criterion = resolve_criterion(args.task_type, args.criterion)
    max_features = parse_max_features(args.max_features)
    class_weight = resolve_class_weight(args.task_type, args.class_weight)

    model_params: Dict[str, Any] = {
        "criterion": criterion,
        "splitter": args.splitter,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": max_features,
        "class_weight": class_weight,
        "ccp_alpha": args.ccp_alpha,
        "random_state": args.random_state,
    }

    log(f"读取训练数据: {args.input_csv}")
    dataframe = pd.read_csv(args.input_csv)
    if dataframe.empty:
        raise ValueError("输入 CSV 为空")
    if args.label_col not in dataframe.columns:
        raise ValueError(
            f"标签列不存在: {args.label_col}; 当前字段: {list(dataframe.columns)}"
        )

    rows_before_cleaning = len(dataframe)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna(subset=[args.label_col])
    dropped_label_rows = rows_before_cleaning - len(dataframe)
    if dataframe.empty:
        raise ValueError("去除空标签后没有可用数据")

    requested_drop_cols = parse_comma_separated(args.drop_cols)
    effective_drop_cols = [
        column
        for column in requested_drop_cols
        if column in dataframe.columns and column != args.label_col
    ]
    ignored_drop_cols = [
        column
        for column in requested_drop_cols
        if column not in dataframe.columns or column == args.label_col
    ]

    X = dataframe.drop(columns=[args.label_col] + effective_drop_cols)
    y = dataframe[args.label_col]
    if X.shape[1] == 0:
        raise ValueError("除标签列和忽略列外没有可用特征")
    if len(X) < 2:
        raise ValueError("有效样本数不足，至少需要 2 条数据")

    if args.task_type == "regression":
        numeric_y = pd.to_numeric(y, errors="coerce")
        valid_target_mask = numeric_y.notna()
        X = X.loc[valid_target_mask].copy()
        y = numeric_y.loc[valid_target_mask].copy()
        if len(y) < 2:
            raise ValueError("回归任务中可转换为数值的有效标签不足 2 条")
    else:
        # A uniform string representation supports numeric and textual labels.
        y = y.astype(str)

    feature_columns = list(X.columns)
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    stratify_y = choose_stratify_y(y, args.task_type, args.test_size)
    log(
        "数据准备完成: "
        f"rows={len(X)}, features={len(feature_columns)}, "
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
        args.task_type,
        numeric_features,
        categorical_features,
        model_params,
    )
    log(f"开始训练决策树: task_type={args.task_type}, criterion={criterion}")
    pipeline.fit(X_train, y_train)
    log("模型训练完成，开始计算指标")

    metric_function = (
        classification_metrics
        if args.task_type == "classification"
        else regression_metrics
    )
    train_metrics = metric_function(pipeline, X_train, y_train)
    test_metrics = metric_function(pipeline, X_test, y_test)

    model = pipeline.named_steps["model"]
    tree_summary = {
        "depth": model.get_depth(),
        "leaf_count": model.get_n_leaves(),
        "node_count": model.tree_.node_count,
    }
    feature_importances = feature_importance_report(pipeline)

    artifact = {
        "pipeline": pipeline,
        "algorithm": "decision_tree",
        "framework": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "task_type": args.task_type,
        "label_col": args.label_col,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "drop_cols": effective_drop_cols,
        "model_params": model_params,
        "classes": model.classes_.tolist()
        if args.task_type == "classification"
        else None,
    }

    report = {
        "status": "success",
        "algorithm": "decision_tree",
        "framework": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": args.input_csv,
        "output_model_path": args.output_model_path,
        "output_metrics_path": args.output_metrics_path,
        "dataset": {
            "rows_before_cleaning": rows_before_cleaning,
            "rows_after_cleaning": len(X),
            "dropped_label_rows": dropped_label_rows,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "ignored_drop_cols": ignored_drop_cols,
        },
        "params": {
            "task_type": args.task_type,
            "label_col": args.label_col,
            "drop_cols": effective_drop_cols,
            "test_size": args.test_size,
            "random_state": args.random_state,
            **model_params,
        },
        "tree": tree_summary,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "feature_importances": feature_importances,
    }

    log(f"保存模型: {args.output_model_path}")
    joblib.dump(artifact, args.output_model_path)
    log(f"保存指标: {args.output_metrics_path}")
    with open(args.output_metrics_path, "w", encoding="utf-8") as output_file:
        json.dump(json_safe(report), output_file, ensure_ascii=False, indent=2)

    log("决策树任务执行成功")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"任务执行失败: {type(exc).__name__}: {exc}")
        raise
