"""
采样核心逻辑
支持随机采样、分层采样、过采样(SMOTE)、欠采样
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def random_sample(df, sample_rate, random_seed=42, replace=False):
    """
    随机采样：按比例或数量随机抽取数据行

    Args:
        df: pandas DataFrame
        sample_rate: 采样比例(0~1)或目标行数
        random_seed: 随机种子
        replace: 是否放回采样

    Returns:
        采样后的DataFrame
    """
    n_total = len(df)
    if sample_rate <= 1:
        n_sample = int(n_total * sample_rate)
    else:
        n_sample = int(sample_rate)
    n_sample = max(1, min(n_sample, n_total if not replace else n_total * 10))

    sampled = df.sample(n=n_sample, random_state=random_seed, replace=replace)
    logger.info(f"随机采样: {n_total} -> {len(sampled)} 行 (比例: {len(sampled)/n_total:.2%})")
    return sampled


def stratified_sample(df, stratify_column, sample_rate, random_seed=42):
    """
    分层采样：按指定列各类别比例抽样，保证各层分布与原始一致

    Args:
        df: pandas DataFrame
        stratify_column: 分层依据列名
        sample_rate: 采样比例(0~1)或目标行数
        random_seed: 随机种子

    Returns:
        采样后的DataFrame
    """
    n_total = len(df)
    groups = df.groupby(stratify_column)
    sampled_list = []

    for _, group in groups:
        n_group = len(group)
        if sample_rate <= 1:
            n_sample = max(1, int(n_group * sample_rate))
        else:
            n_sample = max(1, min(int(sample_rate * n_group / n_total), n_group))
        sampled_list.append(group.sample(n=n_sample, random_state=random_seed))

    sampled = pd.concat(sampled_list).reset_index(drop=True)
    logger.info(f"分层采样(列:{stratify_column}): {n_total} -> {len(sampled)} 行")
    return sampled


def oversampling(df, label_column, sample_rate=1.0, random_seed=42):
    """
    过采样：使用SMOTE对少数类进行合成采样

    Args:
        df: pandas DataFrame
        label_column: 类别标签列名
        sample_rate: 目标比例（相对于多数类）
        random_seed: 随机种子

    Returns:
        过采样后的DataFrame
    """
    from imblearn.over_sampling import SMOTE

    y = df[label_column]
    X = df.drop(columns=[label_column])

    value_counts = y.value_counts()
    max_count = value_counts.max()

    # 按比例计算各类别目标数量
    strategy = {}
    for label, count in value_counts.items():
        if count < max_count:
            strategy[label] = max(count, int(max_count * sample_rate))

    smote = SMOTE(
        sampling_strategy=strategy if strategy else 'auto',
        random_state=random_seed
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)

    result = pd.DataFrame(X_resampled, columns=X.columns)
    result[label_column] = y_resampled

    logger.info(f"SMOTE过采样: {len(df)} -> {len(result)} 行")
    return result


def undersampling(df, label_column, sample_rate=0.5, random_seed=42):
    """
    欠采样：随机丢弃多数类样本

    Args:
        df: pandas DataFrame
        label_column: 类别标签列名
        sample_rate: 目标多数类保留比例（0~1）
        random_seed: 随机种子

    Returns:
        欠采样后的DataFrame
    """
    from imblearn.under_sampling import RandomUnderSampler

    y = df[label_column]
    X = df.drop(columns=[label_column])

    value_counts = y.value_counts()
    max_count = value_counts.max()
    min_count = value_counts.min()

    # 按比例计算多数类目标数量
    strategy = {}
    for label, count in value_counts.items():
        if count > min_count:
            target = max(min_count, int(count * sample_rate))
            strategy[label] = target

    undersample = RandomUnderSampler(
        sampling_strategy=strategy if strategy else 'auto',
        random_state=random_seed
    )
    X_resampled, y_resampled = undersample.fit_resample(X, y)

    result = pd.DataFrame(X_resampled, columns=X.columns)
    result[label_column] = y_resampled

    logger.info(f"欠采样: {len(df)} -> {len(result)} 行")
    return result