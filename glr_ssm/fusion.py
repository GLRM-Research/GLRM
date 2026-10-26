"""双分支融合：共享类别顺序的固定概率平均。"""

import numpy as np


def dual_branch_fusion(static_probabilities, dynamic_probabilities):
    ps = np.asarray(static_probabilities)
    pd = np.asarray(dynamic_probabilities)
    if ps.ndim != 2 or ps.shape != pd.shape or ps.shape[1] < 2:
        raise ValueError("两支概率须为形状相同的 (N, K) 矩阵，类别轴顺序一致。")
    for probabilities in (ps, pd):
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("概率须为 [0, 1] 内的有限数值。")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
            raise ValueError("概率每行之和须为 1。")
    return 0.5 * ps + 0.5 * pd
