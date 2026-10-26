"""局部动态分支：输入 LN、投影、Original Mamba、残差 LN 与分类头。"""

import torch
from torch import nn


class LDRM(nn.Module):
    """输入 (N, 13, feature_dim)，返回 (N, n_classes) logits。

    三频带下，22 通道的 feature_dim=759，3 通道的 feature_dim=18。
    不使用动态 StandardScaler；LayerNorm 对每个 token 的特征轴操作。
    """

    def __init__(self, feature_dim, n_classes):
        super().__init__()
        if feature_dim < 1 or n_classes < 2:
            raise ValueError("特征维度必须为正，类别数至少为 2。")
        from mamba_ssm.modules.mamba_simple import Mamba

        self.feature_dim = feature_dim
        self.n_classes = n_classes
        # 顺序与原主模型一致；保留原参数名称，便于对照原实现。
        self.bridge = nn.LayerNorm(feature_dim, eps=1e-5, elementwise_affine=True)
        self.input = nn.Linear(feature_dim, 32)
        self.temporal = Mamba(
            d_model=32, d_state=16, d_conv=4, expand=2, use_fast_path=False
        )
        self.norm = nn.LayerNorm(32, eps=1e-5)
        self.head = nn.Linear(32, n_classes)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (13, self.feature_dim):
            raise ValueError(f"局部输入须为 (N, 13, {self.feature_dim})。")
        h = self.input(self.bridge(tokens))
        h = self.norm(h + self.temporal(h))
        return self.head(h.mean(dim=1))

    @torch.no_grad()
    def predict_proba(self, tokens):
        """推断期间临时切换 eval，结束后恢复调用前的训练模式。"""
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self(tokens), dim=-1)
        finally:
            self.train(was_training)
