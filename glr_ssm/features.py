"""从三频带 EEG 分析窗提取全局与局部 OAS 协方差。"""

import numpy as np
from sklearn.covariance import oas


BANDS_HZ = ((8, 12), (12, 20), (20, 30))
N_SAMPLES = 875
WINDOW_SIZE = 125
# 250 Hz 下 0.25 秒为 62.5 点，沿用原实现的 half-up 量化。
WINDOW_STARTS = tuple(int(np.floor(0.25 * 250 * i + 0.5)) for i in range(13))


class CovarianceEncoder:
    """输入 (N, 3, C, 875)，频带顺序由 BANDS_HZ 定义。

    输入为已经在原始 run 有效连续段中滤波并裁出的 cue 后
    [0.5, 4.0) 秒 EEG；数据读取、滤波、剔除与划分由外部处理。
    """

    def __init__(self, n_channels):
        if n_channels < 1:
            raise ValueError("EEG 通道数必须为正整数。")
        self.n_channels = n_channels

    def transform(self, epochs):
        epochs = np.asarray(epochs, dtype=np.float64)
        expected = (len(BANDS_HZ), self.n_channels, N_SAMPLES)
        if epochs.ndim != 4 or epochs.shape[1:] != expected or len(epochs) == 0:
            raise ValueError(f"EEG 输入须为 (N, {expected}) 且 N > 0。")
        if not np.isfinite(epochs).all():
            raise ValueError("EEG 输入包含非有限数值。")
        n_trials, n_bands = epochs.shape[:2]
        c = self.n_channels
        full = np.empty((n_trials, n_bands, c, c), dtype=np.float64)
        local = np.empty((n_trials, n_bands, len(WINDOW_STARTS), c, c), dtype=np.float64)
        for trial in range(n_trials):
            for band in range(n_bands):
                signal = epochs[trial, band]
                full[trial, band] = oas(signal.T, assume_centered=False)[0]
                for window, start in enumerate(WINDOW_STARTS):
                    segment = signal[:, start : start + WINDOW_SIZE]
                    local[trial, band, window] = oas(segment.T, assume_centered=False)[0]
        return full, local
