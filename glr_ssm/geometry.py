"""OAS 协方差之后的 AIRM 对齐、矩阵对数与 svec。"""

import warnings

import numpy as np
from pyriemann.utils.mean import mean_riemann


def sym(matrix):
    return (matrix + np.swapaxes(matrix, -1, -2)) * 0.5


def spd_function(matrix, operation):
    """在 float64 下计算 SPD 矩阵函数，不裁剪非正特征值。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("协方差包含非有限数值。")
    eigenvalues, eigenvectors = np.linalg.eigh(sym(matrix))
    if np.any(eigenvalues <= 0):
        raise ValueError("协方差必须严格正定。")
    if operation == "log":
        values = np.log(eigenvalues)
    elif operation == "invsqrt":
        values = eigenvalues ** -0.5
    else:
        raise ValueError(f"未知矩阵函数：{operation}")
    return sym((eigenvectors * values[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2))


def svec(matrix):
    """上三角向量化；非对角元素乘 sqrt(2)，保持 Frobenius 内积。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    i, j = np.triu_indices(matrix.shape[-1])
    values = matrix[..., i, j].copy()
    values[..., i != j] *= np.sqrt(2.0)
    return values


class RiemannianFeatures:
    """逐频带拟合训练侧参考中心，转换时只使用已拟合参数。

    静态输入：(N, B, C, C)；输出：(N, B*C*(C+1)/2)。
    动态输入：(N, B, L, C, C)；输出：(N, L, B*C*(C+1)/2)。
    静态与动态分支须分别创建实例，以使用各自的参考中心。
    """

    def fit(self, covariances):
        covariances = self._validate(covariances)
        centers = []
        for band in range(covariances.shape[1]):
            matrices = covariances[:, band].reshape(-1, *covariances.shape[-2:])
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                center = mean_riemann(
                    matrices, tol=1e-8, maxiter=100, init=matrices.mean(axis=0)
                )
            inverse = spd_function(center, "invsqrt")
            logs = spd_function(sym(inverse @ matrices @ inverse), "log")
            residual = float(np.linalg.norm(logs.mean(axis=0), ord="fro"))
            if observed or not np.isfinite(residual) or residual > 1e-8:
                raise RuntimeError(f"AIRM 中心未收敛：residual={residual}")
            centers.append(center)
        self.centers_ = np.stack(centers)
        self.inverse_sqrts_ = spd_function(self.centers_, "invsqrt")
        self.input_rank_ = covariances.ndim
        self.n_channels_ = covariances.shape[-1]
        return self

    def transform(self, covariances):
        if not hasattr(self, "centers_"):
            raise RuntimeError("请先用训练数据拟合参考中心。")
        covariances = self._validate(covariances)
        if (
            covariances.ndim != self.input_rank_
            or covariances.shape[1] != len(self.centers_)
            or covariances.shape[-1] != self.n_channels_
        ):
            raise ValueError("协方差维度与拟合时不一致。")
        features = []
        for band, inverse in enumerate(self.inverse_sqrts_):
            aligned = sym(inverse @ covariances[:, band] @ inverse)
            features.append(svec(spd_function(aligned, "log")))
        return np.concatenate(features, axis=-1)

    def fit_transform(self, covariances):
        return self.fit(covariances).transform(covariances)

    @staticmethod
    def _validate(covariances):
        values = np.asarray(covariances, dtype=np.float64)
        if values.ndim not in (4, 5) or values.shape[-2] != values.shape[-1]:
            raise ValueError("输入须为静态或动态的逐频带方阵协方差。")
        if any(size == 0 for size in values.shape) or not np.isfinite(values).all():
            raise ValueError("协方差不能为空或包含非有限数值。")
        return values
