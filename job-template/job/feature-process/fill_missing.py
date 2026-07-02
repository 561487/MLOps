"""
填充缺失值核心逻辑
支持均值/中位数/众数/常量/前后向/线性插值填充
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fill_mean(df, columns):
    """均值填充"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            logger.info(f"均值填充 跳过非数值列'{col}'")
            continue
        val = result[col].mean()
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(val)
        logger.info(f"均值填充 列'{col}': 填充 {n_missing} 个缺失值, mean={val:.4f}")
    return result


def fill_median(df, columns):
    """中位数填充"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            logger.info(f"中位数填充 跳过非数值列'{col}'")
            continue
        val = result[col].median()
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(val)
        logger.info(f"中位数填充 列'{col}': 填充 {n_missing} 个缺失值, median={val:.4f}")
    return result


def fill_mode(df, columns):
    """众数填充"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        mode_vals = result[col].mode()
        val = mode_vals[0] if len(mode_vals) > 0 else ''
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(val)
        logger.info(f"众数填充 列'{col}': 填充 {n_missing} 个缺失值, mode={val}")
    return result


def fill_constant(df, columns, value=0):
    """常量填充"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(value)
        logger.info(f"常量填充 列'{col}': 填充 {n_missing} 个缺失值, value={value}")
    return result


def fill_ffill(df, columns):
    """前向填充（用前一个非空值填充）"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(method='ffill')
        # 如果开头仍有缺失，用后向填充补上
        if result[col].isna().any():
            result[col] = result[col].fillna(method='bfill')
        logger.info(f"前向填充 列'{col}': 填充 {n_missing} 个缺失值")
    return result


def fill_bfill(df, columns):
    """后向填充（用后一个非空值填充）"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        n_missing = result[col].isna().sum()
        result[col] = result[col].fillna(method='bfill')
        # 如果末尾仍有缺失，用前向填充补上
        if result[col].isna().any():
            result[col] = result[col].fillna(method='ffill')
        logger.info(f"后向填充 列'{col}': 填充 {n_missing} 个缺失值")
    return result


def fill_interpolate(df, columns, method='linear'):
    """线性插值填充"""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            logger.info(f"插值填充 跳过非数值列'{col}'")
            continue
        n_missing = result[col].isna().sum()
        result[col] = result[col].interpolate(method=method, limit_direction='both')
        # 插值后仍有缺失的用前后向补充
        if result[col].isna().any():
            result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
        logger.info(f"插值填充 列'{col}': 填充 {n_missing} 个缺失值, method={method}")
    return result


def fill_missing(df, columns=None, fill_method='mean', fill_value=0):
    """
    综合填充缺失值

    Args:
        df: pandas DataFrame
        columns: 要填充的列名列表，None 为所有列
        fill_method: 填充方法
        fill_value: 常量填充时的值

    Returns:
        填充后的 DataFrame
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()
    else:
        columns = [c for c in columns if c in result.columns]

    if not columns:
        logger.warning("没有有效的列")
        return result

    total_missing = sum(result[col].isna().sum() for col in columns)
    logger.info(f"开始填充: {len(columns)} 列, 共 {total_missing} 个缺失值")

    fill_funcs = {
        'mean': fill_mean,
        'median': fill_median,
        'mode': fill_mode,
        'constant': lambda df, cols: fill_constant(df, cols, fill_value),
        'ffill': fill_ffill,
        'bfill': fill_bfill,
        'interpolate': fill_interpolate,
    }

    result = fill_funcs[fill_method](result, columns)

    remaining = sum(result[col].isna().sum() for col in columns)
    logger.info(f"填充完成: 剩余 {remaining} 个缺失值")
    return result


FILL_FUNCTIONS = {
    'mean': fill_mean,
    'median': fill_median,
    'mode': fill_mode,
    'constant': fill_constant,
    'ffill': fill_ffill,
    'bfill': fill_bfill,
    'interpolate': fill_interpolate,
}
