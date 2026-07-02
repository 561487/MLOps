"""
统计量计算核心逻辑
支持基础统计量、分布统计量、缺失值统计
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_basic_stats(df, columns):
    """
    计算基础统计量：count、mean、std、min、25%、50%、75%、max

    Args:
        df: pandas DataFrame
        columns: 要计算的列名列表

    Returns:
        DataFrame，行为统计量名，列为数据列名
    """
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        logger.warning("没有有效的列")
        return pd.DataFrame()

    data = df[valid_cols]
    stats = pd.DataFrame(index=[
        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    ])

    for col in valid_cols:
        col_data = data[col].dropna()
        stats.loc['count', col] = len(col_data)
        stats.loc['mean', col] = col_data.mean()
        stats.loc['std', col] = col_data.std()
        stats.loc['min', col] = col_data.min()
        stats.loc['25%', col] = col_data.quantile(0.25)
        stats.loc['50%', col] = col_data.quantile(0.50)
        stats.loc['75%', col] = col_data.quantile(0.75)
        stats.loc['max', col] = col_data.max()

    logger.info(f"基础统计量: {len(valid_cols)} 列, 8 项指标")
    return stats


def calculate_distribution_stats(df, columns):
    """
    计算分布统计量：偏度、峰度、唯一值数

    Args:
        df: pandas DataFrame
        columns: 要计算的列名列表

    Returns:
        DataFrame，行为统计量名，列为数据列名
    """
    from scipy import stats as sp_stats

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        logger.warning("没有有效的列")
        return pd.DataFrame()

    data = df[valid_cols]
    stats = pd.DataFrame(index=['skewness', 'kurtosis', 'unique_count'])

    for col in valid_cols:
        col_data = data[col].dropna()
        if len(col_data) < 3:
            stats.loc['skewness', col] = np.nan
            stats.loc['kurtosis', col] = np.nan
        else:
            stats.loc['skewness', col] = sp_stats.skew(col_data)
            stats.loc['kurtosis', col] = sp_stats.kurtosis(col_data)
        stats.loc['unique_count', col] = col_data.nunique()

    logger.info(f"分布统计量: {len(valid_cols)} 列, 3 项指标")
    return stats


def calculate_missing_stats(df, columns):
    """
    计算缺失值统计：缺失数、缺失率

    Args:
        df: pandas DataFrame
        columns: 要计算的列名列表

    Returns:
        DataFrame，行为统计量名，列为数据列名
    """
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        logger.warning("没有有效的列")
        return pd.DataFrame()

    total = len(df)
    stats = pd.DataFrame(index=['missing_count', 'missing_rate', 'non_missing_count'])

    for col in valid_cols:
        missing = df[col].isna().sum()
        stats.loc['missing_count', col] = missing
        stats.loc['missing_rate', col] = round(missing / total, 4)
        stats.loc['non_missing_count', col] = total - missing

    logger.info(f"缺失值统计: {len(valid_cols)} 列, 3 项指标")
    return stats


def calculate_full_stats(df, columns):
    """
    计算全部统计量：基础 + 分布 + 缺失值

    Args:
        df: pandas DataFrame
        columns: 要计算的列名列表

    Returns:
        DataFrame，合并所有统计量
    """
    basic = calculate_basic_stats(df, columns)
    distribution = calculate_distribution_stats(df, columns)
    missing = calculate_missing_stats(df, columns)

    result = pd.concat([basic, distribution, missing], axis=0)
    logger.info(f"全量统计: {len(result)} 项指标, {len(result.columns)} 列")
    return result


METRIC_FUNCTIONS = {
    'basic': calculate_basic_stats,
    'distribution': calculate_distribution_stats,
    'missing': calculate_missing_stats,
    'full': calculate_full_stats,
}
