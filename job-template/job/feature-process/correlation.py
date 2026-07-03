"""
余弦相似度计算核心逻辑
"""
import numpy as np

def calculate_vector_similarity(vec_a, vec_b):
    """
    计算两个特征向量的余弦相似度

    Args:
        vec_a: numpy array，特征向量A
        vec_b: numpy array，特征向量B（需与 vec_a 等长）

    Returns:
        float: 余弦相似度，值域 [-1, 1]
    """
    # 处理缺失值，用各自均值填充
    vec_a = np.where(np.isnan(vec_a), np.nanmean(vec_a), vec_a)
    vec_b = np.where(np.isnan(vec_b), np.nanmean(vec_b), vec_b)

    # 计算 L2 范数
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # 避免除零
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    # 余弦相似度 = 点积 / (||a|| * ||b||)
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)

    # 裁剪到 [-1, 1]，消除浮点误差
    return float(np.clip(similarity, -1.0, 1.0))