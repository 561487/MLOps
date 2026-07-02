"""
标准化/归一化/正则化核心逻辑
支持 Z-Score、MinMax、MaxAbs、RobustScaler、L2归一化
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def zscore_normalize(df, columns):
    """
    Z-Score 标准化：(x - mean) / std，结果为均值为0、标准差为1的分布

    Args:
        df: pandas DataFrame
        columns: 要标准化的列名列表

    Returns:
        标准化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) < 2:
            logger.warning(f"列 '{col}' 有效数据不足，跳过")
            continue
        mean = col_data.mean()
        std = col_data.std()
        if std < 1e-10:
            result[col] = result[col] - mean
            logger.info(f"Z-Score 列'{col}': std≈0, 仅做中心化, mean={mean:.4f}")
        else:
            result[col] = (result[col] - mean) / std
            logger.info(f"Z-Score 列'{col}': mean={mean:.4f}, std={std:.4f}")
    return result


def minmax_normalize(df, columns, feature_range=(0, 1)):
    """
    MinMax 归一化：(x - min) / (max - min) * (max_range - min_range) + min_range

    Args:
        df: pandas DataFrame
        columns: 要归一化的列名列表
        feature_range: 目标范围，默认 (0, 1)

    Returns:
        归一化后的 DataFrame
    """
    result = df.copy()
    f_min, f_max = feature_range
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) < 2:
            logger.warning(f"列 '{col}' 有效数据不足，跳过")
            continue
        x_min = col_data.min()
        x_max = col_data.max()
        if x_max - x_min < 1e-10:
            result[col] = f_min
            logger.info(f"MinMax 列'{col}': max≈min={x_min:.4f}, 全部置为{f_min}")
        else:
            result[col] = (result[col] - x_min) / (x_max - x_min) * (f_max - f_min) + f_min
            logger.info(f"MinMax 列'{col}': [{x_min:.4f}, {x_max:.4f}] -> [{f_min}, {f_max}]")
    return result


def maxabs_normalize(df, columns):
    """
    MaxAbs 归一化：x / max(|x|)，结果范围 [-1, 1]，保留稀疏性

    Args:
        df: pandas DataFrame
        columns: 要归一化的列名列表

    Returns:
        归一化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            logger.warning(f"列 '{col}' 无有效数据，跳过")
            continue
        max_abs = col_data.abs().max()
        if max_abs < 1e-10:
            result[col] = 0
            logger.info(f"MaxAbs 列'{col}': max_abs≈0, 全部置为0")
        else:
            result[col] = result[col] / max_abs
            logger.info(f"MaxAbs 列'{col}': max_abs={max_abs:.4f}")
    return result


def robust_normalize(df, columns):
    """
    RobustScaler 标准化：(x - median) / IQR，对异常值不敏感

    Args:
        df: pandas DataFrame
        columns: 要标准化的列名列表

    Returns:
        标准化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) < 2:
            logger.warning(f"列 '{col}' 有效数据不足，跳过")
            continue
        median = col_data.median()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr < 1e-10:
            result[col] = result[col] - median
            logger.info(f"Robust 列'{col}': IQR≈0, 仅做中心化, median={median:.4f}")
        else:
            result[col] = (result[col] - median) / iqr
            logger.info(f"Robust 列'{col}': median={median:.4f}, IQR={iqr:.4f}")
    return result


def l2_normalize(df, columns):
    """
    L2 归一化：x / ||x||_2，使每列的 L2 范数为 1

    Args:
        df: pandas DataFrame
        columns: 要归一化的列名列表

    Returns:
        归一化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            continue
        l2_norm = np.sqrt((col_data ** 2).sum())
        if l2_norm < 1e-10:
            result[col] = 0
            logger.info(f"L2 列'{col}': norm≈0, 全部置为0")
        else:
            result[col] = result[col] / l2_norm
            logger.info(f"L2归一化 列'{col}': norm={l2_norm:.4f}")
    return result


NORMALIZE_FUNCTIONS = {
    'zscore': zscore_normalize,
    'minmax': minmax_normalize,
    'maxabs': maxabs_normalize,
    'robust': robust_normalize,
    'l2': l2_normalize,
}
