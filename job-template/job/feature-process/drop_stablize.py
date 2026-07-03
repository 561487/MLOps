"""
去除值过于单一的变量核心逻辑
支持按缺失率、主导值比例、唯一值数、方差阈值删除列
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def drop_by_missing_rate(df, columns=None, threshold=0.5):
    """
    按缺失率删除：缺失比例超过阈值的列

    Args:
        df: pandas DataFrame
        columns: 要检查的列名列表，None 表示所有列
        threshold: 缺失率阈值（0~1），超过即删除

    Returns:
        (删减后的 DataFrame, 被删除的列名列表)
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()

    dropped = []
    for col in columns:
        if col not in result.columns:
            continue
        missing_rate = result[col].isna().mean()
        if missing_rate > threshold:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"缺失率过高 列'{col}': {missing_rate:.2%} > {threshold:.0%}，已删除")

    logger.info(f"缺失率删除: 删除 {len(dropped)} 列, 保留 {len(result.columns)} 列")
    return result, dropped


def drop_by_dominant_ratio(df, columns=None, threshold=0.95):
    """
    按主导值比例删除：最频繁值占比超过阈值的列（即值过于单一）

    Args:
        df: pandas DataFrame
        columns: 要检查的列名列表
        threshold: 主导值比例阈值（0~1），超过即删除

    Returns:
        (删减后的 DataFrame, 被删除的列名列表)
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()

    dropped = []
    for col in columns:
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"全为空值 列'{col}'，已删除")
            continue
        top_count = col_data.value_counts().iloc[0]
        top_ratio = top_count / len(col_data)
        if top_ratio > threshold:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"值过于单一 列'{col}': 主导值占比 {top_ratio:.2%} > {threshold:.0%}，已删除")

    logger.info(f"主导值删除: 删除 {len(dropped)} 列, 保留 {len(result.columns)} 列")
    return result, dropped


def drop_by_unique_count(df, columns=None, min_unique=2):
    """
    按唯一值数删除：唯一值数不足的列

    Args:
        df: pandas DataFrame
        columns: 要检查的列名列表
        min_unique: 最少唯一值数，低于此值则删除

    Returns:
        (删减后的 DataFrame, 被删除的列名列表)
    """
    result = df.copy()
    if columns is None:
        columns = result.columns.tolist()

    dropped = []
    for col in columns:
        if col not in result.columns:
            continue
        n_unique = result[col].dropna().nunique()
        if n_unique < min_unique:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"唯一值不足 列'{col}': {n_unique} < {min_unique}，已删除")

    logger.info(f"唯一值删除: 删除 {len(dropped)} 列, 保留 {len(result.columns)} 列")
    return result, dropped


def drop_by_variance(df, columns=None, threshold=1e-10):
    """
    按方差删除：方差过低的列（接近常量）

    Args:
        df: pandas DataFrame
        columns: 要检查的列名列表（仅数值列有效）
        threshold: 方差阈值，低于此值则删除

    Returns:
        (删减后的 DataFrame, 被删除的列名列表)
    """
    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    dropped = []
    for col in columns:
        if col not in result.columns:
            continue
        if not np.issubdtype(result[col].dtype, np.number):
            continue
        col_data = result[col].dropna()
        if len(col_data) < 2:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"有效值不足 列'{col}': {len(col_data)} < 2，已删除")
            continue
        var = col_data.var()
        if var < threshold:
            result.drop(columns=[col], inplace=True)
            dropped.append(col)
            logger.info(f"方差过低 列'{col}': {var:.2e} < {threshold:.0e}，已删除")

    logger.info(f"方差删除: 删除 {len(dropped)} 列, 保留 {len(result.columns)} 列")
    return result, dropped


def drop_stablize(df, columns=None, missing_threshold=0.5,
                  dominant_threshold=0.95, min_unique=2,
                  variance_threshold=1e-10):
    """
    综合去除值过于单一的变量：合并以上四种策略

    Args:
        df: pandas DataFrame
        columns: 要检查的列名列表，None 为所有列
        missing_threshold: 缺失率阈值
        dominant_threshold: 主导值比例阈值
        min_unique: 最少唯一值数
        variance_threshold: 方差阈值（仅数值列）

    Returns:
        删减后的 DataFrame
    """
    result = df.copy()
    all_dropped = []

    # 按顺序执行，前一步的删除会影响后续
    result, dropped = drop_by_missing_rate(result, columns, missing_threshold)
    all_dropped.extend(dropped)

    result, dropped = drop_by_dominant_ratio(result, columns, dominant_threshold)
    all_dropped.extend(dropped)

    result, dropped = drop_by_unique_count(result, columns, min_unique)
    all_dropped.extend(dropped)

    result, dropped = drop_by_variance(result, columns, variance_threshold)
    all_dropped.extend(dropped)

    logger.info(f"综合去稳定: 共删除 {len(all_dropped)} 列, 保留 {len(result.columns)} 列")
    logger.info(f"被删除列: {all_dropped}")
    return result


STABLIZE_FUNCTIONS = {
    'missing_rate': drop_by_missing_rate,
    'dominant': drop_by_dominant_ratio,
    'unique': drop_by_unique_count,
    'variance': drop_by_variance,
    'all': drop_stablize,
}
