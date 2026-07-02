"""
执行 SQL 核心逻辑
将 CSV 数据加载为 SQLite 内存表，执行 SQL 查询
"""
import pandas as pd
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)


def execute_sql(input_file, sql_query, table_name='data'):
    """
    将 CSV 文件加载为 SQLite 内存表并执行 SQL 查询

    Args:
        input_file: 输入 CSV 文件路径
        sql_query: SQL 查询语句
        table_name: CSV 数据加载到的表名，默认 'data'

    Returns:
        查询结果 DataFrame
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    if not sql_query or not sql_query.strip():
        raise ValueError("SQL 查询语句不能为空")

    df = pd.read_csv(input_file)
    logger.info(f"加载数据: {len(df)} 行, {len(df.columns)} 列 -> 表 '{table_name}'")
    logger.info(f"列名: {list(df.columns)}")

    # 创建内存 SQLite 数据库
    conn = sqlite3.connect(':memory:')
    df.to_sql(table_name, conn, index=False, if_exists='replace')

    # 执行查询
    sql = sql_query.strip()
    logger.info(f"执行 SQL: {sql}")
    result = pd.read_sql_query(sql, conn)
    conn.close()

    logger.info(f"查询结果: {len(result)} 行, {len(result.columns)} 列")
    return result


def execute_sql_multi_table(file_map, sql_query):
    """
    多表 SQL 查询：将多个 CSV 文件加载为多张表

    Args:
        file_map: {表名: 文件路径} 字典
        sql_query: SQL 查询语句

    Returns:
        查询结果 DataFrame
    """
    conn = sqlite3.connect(':memory:')
    for table_name, file_path in file_map.items():
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        logger.info(f"加载表 '{table_name}': {len(df)} 行, {len(df.columns)} 列")

    sql = sql_query.strip()
    logger.info(f"执行 SQL: {sql}")
    result = pd.read_sql_query(sql, conn)
    conn.close()

    logger.info(f"查询结果: {len(result)} 行, {len(result.columns)} 列")
    return result
