"""
删除缺失率过高的值核心逻辑
支持按列缺失率删列、按行缺失率删行
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def drop_columns_by_missing(df, threshold=0.5, columns=None):
    """
    按列缺失率删除：缺失比例超过阈值的列

    Args:
        df: pandas DataFrame
        threshold: 缺失率阈值（0~1）
        columns: 要检查的列名列表，None 为所有列

    Returns:
        (删减后的 DataFrame, 被删除的列名列表)
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()
    else:
        columns = [c for c in columns if c in result.columns]

    dropped = []
    for col in columns:
        missing_rate = result[col].isna().mean()
        if missing_rate > threshold:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"缺失率过高 列'{col}': {missing_rate:.2%} > {threshold:.0%}，已删除")

    logger.info(f"列缺失删除: 删除 {len(dropped)} 列, 保留 {len(result.columns)} 列")
    if dropped:
        logger.info(f"被删除列: {dropped}")
    return result, dropped


def drop_rows_by_missing(df, threshold=0.5, columns=None):
    """
    按行缺失率删除：缺失比例超过阈值的行

    Args:
        df: pandas DataFrame
        threshold: 缺失率阈值（0~1）
        columns: 要参与计算的列名列表，None 为所有列

    Returns:
        (删减后的 DataFrame, 删除的行数)
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()
    else:
        columns = [c for c in columns if c in result.columns]

    if not columns:
        logger.warning("没有有效的列用于行缺失计算")
        return result, 0

    missing_rate = result[columns].isna().mean(axis=1)
    mask = missing_rate <= threshold
    n_dropped = (~mask).sum()

    result = result[mask].reset_index(drop=True)
    logger.info(f"行缺失删除: 删除 {n_dropped} 行, 保留 {len(result)} 行 "
                f"({n_dropped/len(df):.2%})")
    return result, n_dropped


def drop_high_missing(df, axis='both', col_threshold=0.5, row_threshold=0.5, columns=None):
    """
    综合删除缺失率过高的值：先删列再删行

    Args:
        df: pandas DataFrame
        axis: 'column'(仅删列) / 'row'(仅删行) / 'both'(先删列再删行)
        col_threshold: 列缺失率阈值
        row_threshold: 行缺失率阈值
        columns: 要参与计算的列

    Returns:
        删减后的 DataFrame
    """
    result = df.copy()
    total_dropped_cols = 0
    total_dropped_rows = 0

    if axis in ('column', 'both'):
        result, dropped_cols = drop_columns_by_missing(result, col_threshold, columns)
        total_dropped_cols = len(dropped_cols)

    if axis in ('row', 'both'):
        result, n_dropped = drop_rows_by_missing(result, row_threshold, columns)
        total_dropped_rows = n_dropped

    logger.info(f"缺失删除完成: 删除 {total_dropped_cols} 列 + {total_dropped_rows} 行, "
                f"最终 {len(result)} 行 × {len(result.columns)} 列")
    return result


MISSING_FUNCTIONS = {
    'column': drop_columns_by_missing,
    'row': drop_rows_by_missing,
    'both': drop_high_missing,
}
