#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdaBoost training launcher for the MLOps job-template platform."""

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
SUPPORTED_ALGORITHMS = {"SAMME", "SAMME.R"}
SUPPORTED_LOSSES = {"linear", "square", "exponential"}
DIAGNOSTIC_DETAIL_LIMIT = 200


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AdaBoost classifier or regressor with scikit-learn."
    )

    parser.add_argument("--input_csv", default="", help="训练数据 CSV 路径")
    parser.add_argument("--label_col", default="label", help="标签列名称")
    parser.add_argument(
        "--drop_cols",
        default="",
        help="不参与训练的列，多个列名使用英文逗号分隔",
    )
    parser.add_argument(
        "--task_type",
        default="classification",
        choices=sorted(SUPPORTED_TASK_TYPES),
        help="任务类型：classification 或 regression",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")

    parser.add_argument(
        "--n_estimators", type=int, default=100, help="弱学习器最大数量"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.0, help="Boosting 学习率"
    )
    parser.add_argument(
        "--algorithm",
        default="auto",
        help="分类算法：auto、SAMME 或 SAMME.R",
    )
    parser.add_argument(
        "--loss",
        default="auto",
        help="回归损失：auto、linear、square 或 exponential",
    )
    parser.add_argument(
        "--base_max_depth",
        default="auto",
        help="弱决策树深度；auto 对分类取 1、对回归取 3",
    )
    parser.add_argument(
        "--base_min_samples_split",
        type=int,
        default=2,
        help="弱决策树节点分裂所需最小样本数",
    )
    parser.add_argument(
        "--base_min_samples_leaf",
        type=int,
        default=1,
        help="弱决策树叶子节点最小样本数",
    )
    parser.add_argument(
        "--base_max_features",
        default="",
        help="为空表示全部特征；支持 sqrt、log2、正整数或 (0,1] 小数",
    )

    parser.add_argument("--output_model_path", default="", help="模型输出 .pkl 路径")
    parser.add_argument(
        "--output_metrics_path", default="", help="指标输出 JSON 路径"
    )
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
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
            "base_max_features 仅支持空值、sqrt、log2、正整数或 (0,1] 小数"
        ) from exc


def resolve_base_max_depth(task_type: str, value: str) -> int:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return 1 if task_type == "classification" else 3
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError("base_max_depth 仅支持 auto 或正整数") from exc
    if parsed < 1:
        raise ValueError("base_max_depth 必须为 auto 或 >= 1 的整数")
    return parsed


def resolve_algorithm(task_type: str, value: str) -> Optional[str]:
    normalized = str(value or "auto").strip().upper()
    if task_type == "regression":
        if normalized not in {"", "AUTO"}:
            raise ValueError("algorithm 仅适用于 classification 分类任务")
        return None
    if normalized in {"", "AUTO"}:
        return "SAMME.R"
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"algorithm 仅支持: {sorted(SUPPORTED_ALGORITHMS)}")
    return normalized


def resolve_loss(task_type: str, value: str) -> Optional[str]:
    normalized = str(value or "auto").strip().lower()
    if task_type == "classification":
        if normalized not in {"", "auto"}:
            raise ValueError("loss 仅适用于 regression 回归任务")
        return None
    if normalized in {"", "auto"}:
        return "linear"
    if normalized not in SUPPORTED_LOSSES:
        raise ValueError(f"loss 仅支持: {sorted(SUPPORTED_LOSSES)}")
    return normalized


def validate_args(args: argparse.Namespace) -> None:
    if args.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task_type 仅支持: {sorted(SUPPORTED_TASK_TYPES)}")
    if not 0 < args.test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if args.random_state < 0:
        raise ValueError("random_state 必须 >= 0")
    if args.n_estimators < 1:
        raise ValueError("n_estimators 必须 >= 1")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate 必须 > 0")
    if args.base_min_samples_split < 2:
        raise ValueError("base_min_samples_split 必须 >= 2")
    if args.base_min_samples_leaf < 1:
        raise ValueError("base_min_samples_leaf 必须 >= 1")
    if not args.input_csv:
        raise ValueError("input_csv 不能为空，请填写容器内可访问的训练 CSV 路径")
    if not args.output_model_path:
        raise ValueError("output_model_path 不能为空，请填写模型输出路径")
    if not args.output_metrics_path:
        raise ValueError("output_metrics_path 不能为空，请填写指标输出路径")
    if os.path.abspath(args.output_model_path) == os.path.abspath(
        args.output_metrics_path
    ):
        raise ValueError("output_model_path 和 output_metrics_path 不能是同一路径")


def resolve_model_params(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "algorithm": resolve_algorithm(args.task_type, args.algorithm),
        "loss": resolve_loss(args.task_type, args.loss),
        "base_max_depth": resolve_base_max_depth(
            args.task_type, args.base_max_depth
        ),
        "base_min_samples_split": args.base_min_samples_split,
        "base_min_samples_leaf": args.base_min_samples_leaf,
        "base_max_features": parse_max_features(args.base_max_features),
        "random_state": args.random_state,
    }


def ensure_input_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"训练数据文件不存在: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_one_hot_encoder() -> OneHotEncoder:
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

    base_params = {
        "max_depth": model_params["base_max_depth"],
        "min_samples_split": model_params["base_min_samples_split"],
        "min_samples_leaf": model_params["base_min_samples_leaf"],
        "max_features": model_params["base_max_features"],
        "random_state": model_params["random_state"],
    }
    if task_type == "classification":
        base_estimator = DecisionTreeClassifier(**base_params)
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=model_params["n_estimators"],
            learning_rate=model_params["learning_rate"],
            algorithm=model_params["algorithm"],
            random_state=model_params["random_state"],
        )
    else:
        base_estimator = DecisionTreeRegressor(**base_params)
        model = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=model_params["n_estimators"],
            learning_rate=model_params["learning_rate"],
            loss=model_params["loss"],
            random_state=model_params["random_state"],
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    warnings: List[str] = []
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    if test_count < 1 or train_count < 1:
        raise ValueError("有效样本数不足以按当前 test_size 切分训练集和测试集")

    if task_type != "classification":
        split = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return (*split, warnings)

    value_counts = y.value_counts(dropna=False)
    class_count = len(value_counts)
    if class_count < 2:
        raise ValueError("分类任务至少需要 2 个类别")
    if train_count < class_count:
        raise ValueError(
            "当前 test_size 导致训练集样本数少于类别数，无法保证每个类别参与训练"
        )

    can_stratify = (
        value_counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )
    if can_stratify:
        split = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return (*split, warnings)

    warning = (
        "数据条件不满足分层抽样，已使用保底随机切分并确保每个类别至少有一条样本留在训练集"
    )
    warnings.append(warning)
    log(warning)

    random_generator = np.random.RandomState(random_state)
    mandatory_train_positions: List[int] = []
    y_values = y.to_numpy()
    for class_value in pd.unique(y):
        positions = np.flatnonzero(y_values == class_value)
        mandatory_train_positions.append(int(random_generator.choice(positions)))

    mandatory_set = set(mandatory_train_positions)
    remaining_positions = np.array(
        [position for position in range(len(y)) if position not in mandatory_set],
        dtype=int,
    )
    random_generator.shuffle(remaining_positions)
    test_positions = remaining_positions[:test_count]
    extra_train_positions = remaining_positions[test_count:]
    train_positions = np.array(
        mandatory_train_positions + extra_train_positions.tolist(), dtype=int
    )
    random_generator.shuffle(train_positions)

    return (
        X.iloc[train_positions].copy(),
        X.iloc[test_positions].copy(),
        y.iloc[train_positions].copy(),
        y.iloc[test_positions].copy(),
        warnings,
    )


def classification_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    warnings: List[str],
) -> Dict[str, Any]:
    predictions = pipeline.predict(X)
    model = pipeline.named_steps["model"]
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=model.classes_
        ).tolist(),
    }

    try:
        probabilities = pipeline.predict_proba(X)
        if len(model.classes_) == 2:
            if len(pd.unique(y)) < 2:
                raise ValueError("评估数据只包含一个类别")
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])
        elif len(pd.unique(y)) == len(model.classes_):
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="macro",
            )
        else:
            raise ValueError("评估数据未覆盖训练模型的全部类别")
    except ValueError as exc:
        warnings.append(f"{split_name} 未计算 ROC-AUC: {exc}")
    return metrics


def regression_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    warnings: List[str],
) -> Dict[str, Any]:
    predictions = pipeline.predict(X)
    mse = mean_squared_error(y, predictions)
    result = {
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y, predictions),
        "r2": r2_score(y, predictions),
    }
    if not np.isfinite(result["r2"]):
        warnings.append(f"{split_name} 样本不足，R2 无法计算")
    return result


def feature_importance_report(
    pipeline: Pipeline, warnings: List[str]
) -> List[Dict[str, Any]]:
    try:
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
    except (AttributeError, ValueError) as exc:
        warnings.append(f"未生成特征重要性: {exc}")
        return []


def summarize_values(values: Any, actual_count: int) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)[:actual_count]
    finite_values = array[np.isfinite(array)]
    return {
        "count": len(array),
        "min": float(np.min(finite_values)) if len(finite_values) else None,
        "max": float(np.max(finite_values)) if len(finite_values) else None,
        "mean": float(np.mean(finite_values)) if len(finite_values) else None,
        "values": array[:DIAGNOSTIC_DETAIL_LIMIT].tolist(),
        "values_truncated": len(array) > DIAGNOSTIC_DETAIL_LIMIT,
    }


def boosting_diagnostics(pipeline: Pipeline, requested_count: int) -> Dict[str, Any]:
    model = pipeline.named_steps["model"]
    actual_count = len(model.estimators_)
    return {
        "requested_estimators": requested_count,
        "fitted_estimators": actual_count,
        "stopped_early": actual_count < requested_count,
        "estimator_weights": summarize_values(model.estimator_weights_, actual_count),
        "estimator_errors": summarize_values(model.estimator_errors_, actual_count),
    }


def atomic_joblib_dump(value: Any, output_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".adaboost-model-", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    try:
        joblib.dump(value, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_json_dump(value: Any, output_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".adaboost-metrics-", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                json_safe(value),
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def train(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_params = resolve_model_params(args)
    ensure_input_file(args.input_csv)
    ensure_parent_dir(args.output_model_path)
    ensure_parent_dir(args.output_metrics_path)

    warnings: List[str] = []
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
    dataframe = dataframe.dropna(subset=[args.label_col]).copy()
    dropped_label_rows = rows_before_cleaning - len(dataframe)
    if dropped_label_rows:
        warnings.append(f"已删除 {dropped_label_rows} 条空标签数据")
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
    if ignored_drop_cols:
        warnings.append(f"以下忽略列不存在或为标签列，未执行删除: {ignored_drop_cols}")

    X = dataframe.drop(columns=[args.label_col] + effective_drop_cols)
    y = dataframe[args.label_col]
    if X.shape[1] == 0:
        raise ValueError("除标签列和忽略列外没有可用特征")

    dropped_invalid_target_rows = 0
    if args.task_type == "regression":
        numeric_y = pd.to_numeric(y, errors="coerce")
        valid_target_mask = numeric_y.notna()
        dropped_invalid_target_rows = int((~valid_target_mask).sum())
        X = X.loc[valid_target_mask].copy()
        y = numeric_y.loc[valid_target_mask].copy()
        if dropped_invalid_target_rows:
            warnings.append(
                f"已删除 {dropped_invalid_target_rows} 条无法转换为数值的回归标签数据"
            )
        if len(y) < 2:
            raise ValueError("回归任务中可转换为数值的有效标签不足 2 条")
    else:
        y = y.astype(str)
        if y.nunique(dropna=False) < 2:
            raise ValueError("分类任务至少需要 2 个类别")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if len(X) < 2:
        raise ValueError("有效样本数不足，至少需要 2 条数据")

    feature_columns = list(X.columns)
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    log(
        "数据准备完成: "
        f"rows={len(X)}, features={len(feature_columns)}, "
        f"numeric={len(numeric_features)}, categorical={len(categorical_features)}"
    )
    X_train, X_test, y_train, y_test, split_warnings = split_dataset(
        X,
        y,
        args.task_type,
        args.test_size,
        args.random_state,
    )
    warnings.extend(split_warnings)

    pipeline = build_pipeline(
        args.task_type,
        numeric_features,
        categorical_features,
        model_params,
    )
    log(
        "开始训练 AdaBoost: "
        f"task_type={args.task_type}, n_estimators={args.n_estimators}, "
        f"learning_rate={args.learning_rate}"
    )
    pipeline.fit(X_train, y_train)
    log("模型训练完成，开始计算指标")

    metric_function = (
        classification_metrics
        if args.task_type == "classification"
        else regression_metrics
    )
    train_metrics = metric_function(pipeline, X_train, y_train, "训练集", warnings)
    test_metrics = metric_function(pipeline, X_test, y_test, "测试集", warnings)
    model = pipeline.named_steps["model"]
    feature_importances = feature_importance_report(pipeline, warnings)
    diagnostics = boosting_diagnostics(pipeline, args.n_estimators)

    created_at = datetime.now().isoformat(timespec="seconds")
    classes = (
        model.classes_.tolist() if args.task_type == "classification" else None
    )
    data_summary = {
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": len(X),
        "dropped_label_rows": dropped_label_rows,
        "dropped_invalid_target_rows": dropped_invalid_target_rows,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
    }
    library_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    artifact = {
        "artifact_version": "1.0",
        "algorithm": "adaboost",
        "framework": "scikit-learn",
        "pipeline": pipeline,
        "task_type": args.task_type,
        "label_col": args.label_col,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "drop_cols": effective_drop_cols,
        "classes": classes,
        "model_params": model_params,
        "data_summary": data_summary,
        "library_versions": library_versions,
        "trained_at": created_at,
    }

    report = {
        "schema_version": "1.0",
        "status": "success",
        "algorithm": "adaboost",
        "framework": "scikit-learn",
        "task_type": args.task_type,
        "created_at": created_at,
        "input_csv": args.input_csv,
        "data": {
            **data_summary,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "effective_drop_cols": effective_drop_cols,
            "ignored_drop_cols": ignored_drop_cols,
            "classes": classes,
        },
        "model_params": model_params,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "boosting": diagnostics,
        "feature_importances": feature_importances,
        "warnings": warnings,
        "library_versions": library_versions,
        "artifacts": {
            "model_path": args.output_model_path,
            "metrics_path": args.output_metrics_path,
        },
    }
    return artifact, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    artifact, report = train(args)
    log(f"保存模型: {args.output_model_path}")
    atomic_joblib_dump(artifact, args.output_model_path)
    log(f"保存指标: {args.output_metrics_path}")
    atomic_json_dump(report, args.output_metrics_path)
    log("AdaBoost 任务执行成功")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"任务执行失败: {type(exc).__name__}: {exc}")
        raise
