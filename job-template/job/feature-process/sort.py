"""
排序核心逻辑
支持单列/多列排序，升序/降序
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def sort_data(df, by_columns, ascending=True):
    """
    按指定列排序

    Args:
        df: pandas DataFrame
        by_columns: 排序列名列表
        ascending: True=升序, False=降序；也支持列表逐个指定

    Returns:
        排序后的 DataFrame
    """
    if not by_columns:
        logger.warning("未指定排序列，按所有列字典序排序")
        result = df.sort_values(by=list(df.columns), ascending=ascending)
    else:
        valid_cols = [c for c in by_columns if c in df.columns]
        invalid = [c for c in by_columns if c not in df.columns]
        if invalid:
            logger.warning(f"列不存在: {invalid}")
        if not valid_cols:
            logger.error("没有有效的排序列")
            return df
        result = df.sort_values(by=valid_cols, ascending=ascending)

    result = result.reset_index(drop=True)
    logger.info(f"排序: by={by_columns if by_columns else list(df.columns)}, "
                f"ascending={ascending}, {len(result)} 行")
    return result


def sort_by_multiple(df, columns_str, orders_str):
    """
    多列分别指定升降序

    Args:
        df: pandas DataFrame
        columns_str: 逗号分隔的列名，如 "col1,col2"
        orders_str: 逗号分隔的升降序，如 "asc,desc"

    Returns:
        排序后的 DataFrame
    """
    columns = [c.strip() for c in columns_str.split(",") if c.strip()]
    orders = [o.strip().lower() for o in orders_str.split(",") if o.strip()]

    valid_cols = [c for c in columns if c in df.columns]
    invalid = [c for c in columns if c not in df.columns]
    if invalid:
        logger.warning(f"列不存在: {invalid}")
    if not valid_cols:
        logger.error("没有有效的排序列")
        return df

    ascending_list = []
    for i in range(len(valid_cols)):
        if i < len(orders):
            ascending_list.append(orders[i] in ('asc', 'true', '1'))
        else:
            ascending_list.append(True)

    result = df.sort_values(by=valid_cols, ascending=ascending_list)
    result = result.reset_index(drop=True)
    logger.info(f"多列排序: by={valid_cols}, ascending={ascending_list}, {len(result)} 行")
    return result
