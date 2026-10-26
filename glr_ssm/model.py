"""GLRM 模型接口，组合全局静态与局部动态两支。"""

import numpy as np
import torch

from .features import CovarianceEncoder
from .fusion import dual_branch_fusion
from .geometry import RiemannianFeatures
from .gsr import GSR
from .ldrm import LDRM


class GLRM:
    """接收三频带 EEG 分析窗 (N, 3, C, 875)。

    n_channels=22、n_classes=4 对应 IV-2a；3、2 对应 IV-2b。
    static_c 必须由外部训练侧验证选定。
    两支分别训练：fit_references 只拟合当前允许训练集合的几何，
    fit_static 拟合静态分类器；动态训练调用 dynamic_logits，
    交叉熵与优化器由外部训练代码负责。重新训练阶段应创建新实例。
    """

    def __init__(self, n_channels, n_classes, static_c):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_dim = 3 * n_channels * (n_channels + 1) // 2
        self.encoder = CovarianceEncoder(n_channels)
        self.static_geometry = RiemannianFeatures()
        self.dynamic_geometry = RiemannianFeatures()
        self.gsr = GSR(n_classes=n_classes, C=static_c)
        self.ldrm = LDRM(feature_dim=self.feature_dim, n_classes=n_classes)

    def fit_references(self, training_epochs):
        """只传当前拟合阶段允许的训练 EEG；验证／测试不调用此方法。"""
        if hasattr(self.static_geometry, "centers_"):
            raise RuntimeError("参考中心已拟合；新的拟合阶段请创建新模型。")
        full, local = self.encoder.transform(training_epochs)
        self.static_geometry.fit(full)
        self.dynamic_geometry.fit(local)
        return self

    def transform(self, epochs):
        """返回静态 float64 特征和动态 float32 token，均不更新拟合参数。"""
        full, local = self.encoder.transform(epochs)
        return (
            self.static_geometry.transform(full),
            self.dynamic_geometry.transform(local).astype(np.float32),
        )

    def fit_static(self, training_epochs, labels):
        """在同一允许训练集合上拟合静态 scaler 和逻辑回归。"""
        full, _ = self.encoder.transform(training_epochs)
        self.gsr.fit(self.static_geometry.transform(full), labels)
        return self

    def dynamic_logits(self, tokens):
        """训练入口：token 可为 NumPy 数组或 Tensor，保留网络参数梯度。"""
        parameter = next(self.ldrm.parameters())
        tokens = torch.as_tensor(tokens, dtype=parameter.dtype, device=parameter.device)
        return self.ldrm(tokens)

    def predict_branches(self, epochs):
        static_features, tokens = self.transform(epochs)
        ps = self.gsr.predict_proba(static_features)
        parameter = next(self.ldrm.parameters())
        tokens = torch.as_tensor(tokens, dtype=parameter.dtype, device=parameter.device)
        pd = self.ldrm.predict_proba(tokens).cpu().numpy()
        return ps, pd

    def predict_proba(self, epochs):
        return dual_branch_fusion(*self.predict_branches(epochs))

    def predict(self, epochs):
        # NumPy argmax 在并列最大值时选预定义类别轴中的首个类别。
        return self.predict_proba(epochs).argmax(axis=1)


# 保留早期模型包的调用名称。
GLRSSM = GLRM
