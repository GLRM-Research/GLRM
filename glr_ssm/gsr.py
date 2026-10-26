"""全局静态分支：训练侧 StandardScaler + L2 逻辑回归。"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class GSR:
    """C 由外部训练侧验证确定；类别轴固定为 0..n_classes-1。"""

    def __init__(self, n_classes, C):
        if n_classes < 2 or C <= 0 or not np.isfinite(C):
            raise ValueError("n_classes 至少为 2，C 须为有限正数。")
        self.n_classes = n_classes
        self.C = C

    def fit(self, features, labels):
        features = self._validate(features)
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("标签必须与特征逐行对应。")
        if not np.array_equal(np.unique(labels), np.arange(self.n_classes)):
            raise ValueError("训练标签须包含 0..n_classes-1 的全部类别。")
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C, solver="lbfgs", max_iter=2000, tol=1e-6,
                fit_intercept=True, class_weight=None, random_state=42,
            ),
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            pipeline.fit(features, labels)
        if any(isinstance(item.message, ConvergenceWarning) for item in observed):
            raise RuntimeError("逻辑回归未收敛。")
        self.pipeline_ = pipeline
        return self

    def predict_proba(self, features):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("请先拟合全局静态分类器。")
        return self.pipeline_.predict_proba(self._validate(features))

    @staticmethod
    def _validate(features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("静态特征须为有限的非空二维矩阵。")
        return features
