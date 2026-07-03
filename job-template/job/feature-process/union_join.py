"""
数据合并核心逻辑
支持 Union（行级拼接）和 Join（列级关联）操作
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def union_data(df_a, df_b):
    """
    行级拼接：将两个 DataFrame 纵向拼接（类似 SQL UNION ALL）

    按列名对齐，仅保留两个 DataFrame 共有的列进行拼接；
    若两个 DataFrame 的 dtypes 不一致，自动转换为 object 类型避免报错。

    Args:
        df_a: pandas DataFrame
        df_b: pandas DataFrame

    Returns:
        拼接后的 DataFrame
    """
    common_cols = list(set(df_a.columns) & set(df_b.columns))
    if not common_cols:
        logger.warning("两个数据集没有共同的列，将分别保留各自的列（缺失值填充 NaN）")
        result = pd.concat([df_a, df_b], ignore_index=True, sort=False)
    else:
        logger.info(f"按 {len(common_cols)} 个共有列对齐拼接")
        # 对齐列顺序，并对 dtype 不一致的列统一转 object
        a_cols = df_a[common_cols].copy()
        b_cols = df_b[common_cols].copy()
        for col in common_cols:
            if a_cols[col].dtype != b_cols[col].dtype:
                a_cols[col] = a_cols[col].astype(object)
                b_cols[col] = b_cols[col].astype(object)
        result = pd.concat([a_cols, b_cols], ignore_index=True)

    logger.info(f"Union 完成: {len(df_a)} + {len(df_b)} -> {len(result)} 行")
    return result


def join_data(df_a, df_b, on, how='inner'):
    """
    列级关联：按指定列将两个 DataFrame 进行关联（类似 SQL JOIN）

    Args:
        df_a: pandas DataFrame（左表）
        df_b: pandas DataFrame（右表）
        on: 关联键列名（单个列名字符串，或列名列表）
        how: 关联方式，支持 'inner' | 'left' | 'right' | 'outer'，默认 'inner'

    Returns:
        关联后的 DataFrame
    """
    # 处理同名列后缀，避免冲突
    result = pd.merge(df_a, df_b, on=on, how=how, suffixes=('_a', '_b'))

    logger.info(f"Join ({how}) 完成: {len(df_a)} + {len(df_b)} -> {len(result)} 行 "
                f"(on={on})")
    return result


def merge_data(df_a, df_b, merge_type, join_on=None, join_how='inner'):
    """
    统一入口：根据 merge_type 调用对应的合并函数

    Args:
        df_a: pandas DataFrame（左表/上表）
        df_b: pandas DataFrame（右表/下表）
        merge_type: 合并类型，'union' 或 'join'
        join_on: 关联键列名，仅 join 时有效
        join_how: 关联方式，仅 join 时有效

    Returns:
        合并后的 DataFrame
    """
    if merge_type == 'union':
        return union_data(df_a, df_b)
    elif merge_type == 'join':
        if not join_on:
            raise ValueError("join 操作必须指定 --join_on 参数")
        return join_data(df_a, df_b, on=join_on, how=join_how)
    else:
        raise ValueError(f"不支持的合并类型: {merge_type}，可选: union, join")
