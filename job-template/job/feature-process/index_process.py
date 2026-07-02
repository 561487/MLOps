"""
索引处理核心逻辑
支持增加索引列、设置索引、索引转列、按索引重命名/重排列
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def add_index_column(df, index_name='index', start=0, step=1):
    """
    增加索引列：新增一列自增序号

    Args:
        df: pandas DataFrame
        index_name: 新索引列名
        start: 起始值
        step: 步长

    Returns:
        处理后的 DataFrame
    """
    result = df.copy()
    result[index_name] = range(start, start + len(result) * step, step)
    # 将索引列移到第一列
    cols = [index_name] + [c for c in result.columns if c != index_name]
    result = result[cols]
    logger.info(f"增加索引列 '{index_name}': {len(result)} 行, "
                f"范围 [{start}, {start + (len(result)-1)*step}]")
    return result


def set_column_as_index(df, column):
    """
    设置列为索引：将指定列设为 DataFrame 的行索引

    Args:
        df: pandas DataFrame
        column: 要设为索引的列名

    Returns:
        处理后的 DataFrame
    """
    if column not in df.columns:
        logger.error(f"列 '{column}' 不存在")
        return df

    result = df.set_index(column)
    logger.info(f"设置列为索引: '{column}' -> index, shape={result.shape}")
    return result


def reset_index_to_column(df, index_name='index'):
    """
    索引转列：将行索引转为普通列，重置为默认整数索引

    Args:
        df: pandas DataFrame
        index_name: 索引转列后的列名

    Returns:
        处理后的 DataFrame
    """
    result = df.reset_index()
    # 如果原索引无名且新列名不是 'index'，重命名
    if 'index' in result.columns and index_name != 'index':
        # 检查是否是 reset_index 生成的
        if result.columns[0] == 'index':
            result.rename(columns={'index': index_name}, inplace=True)
            logger.info(f"索引转列: index -> '{index_name}', shape={result.shape}")
        else:
            logger.info(f"索引转列: shape={result.shape}")
    else:
        logger.info(f"索引转列: shape={result.shape}")
    return result


def rename_columns_by_index(df, name_map, inplace=False):
    """
    按索引位置重命名列

    Args:
        df: pandas DataFrame
        name_map: 索引->新列名的映射，如 "0:new_col_a,2:new_col_c"
        inplace: 是否就地修改（否则新增列）

    Returns:
        处理后的 DataFrame
    """
    result = df.copy()
    rename_dict = {}
    for item in name_map.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            logger.warning(f"格式错误: {item}，应为 '索引:新列名'")
            continue
        try:
            idx = int(parts[0])
            new_name = parts[1].strip()
            if idx < len(result.columns):
                rename_dict[result.columns[idx]] = new_name
            else:
                logger.warning(f"索引 {idx} 超出范围 (列数={len(result.columns)})")
        except ValueError:
            logger.warning(f"索引解析失败: {parts[0]}")

    if inplace:
        result.rename(columns=rename_dict, inplace=True)
    else:
        # 新增列（复制后重命名，保留原列）
        for old_name, new_name in rename_dict.items():
            result[new_name] = result[old_name]

    logger.info(f"按索引重命名: {rename_dict}")
    return result


def reorder_columns(df, column_order):
    """
    按指定顺序重排列

    Args:
        df: pandas DataFrame
        column_order: 逗号分隔的列名顺序，如 "col2,col1,col3"
                     未列出的列保持原顺序追加到末尾

    Returns:
        重排列后的 DataFrame
    """
    result = df.copy()
    if not column_order.strip():
        return result

    ordered = [c.strip() for c in column_order.split(",")]
    ordered = [c for c in ordered if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    result = result[ordered + remaining]
    logger.info(f"重排列: 前 {len(ordered)} 列指定顺序 + {len(remaining)} 列保持原序")
    return result


INDEX_FUNCTIONS = {
    'add_index': add_index_column,
    'set_index': set_column_as_index,
    'reset_index': reset_index_to_column,
    'rename_by_index': rename_columns_by_index,
    'reorder': reorder_columns,
}
