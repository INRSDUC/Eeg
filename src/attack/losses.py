from __future__ import annotations

import numpy as np


def untargeted_margin(scores: np.ndarray, true_label: int) -> float:
    s_true = float(scores[true_label])
    tmp = scores.copy()
    tmp[true_label] = -np.inf
    s_other = float(np.max(tmp))
    return s_true - s_other


def tv_regularizer(delta: np.ndarray) -> float:
    if delta.shape[1] <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(delta, axis=1))))


def band_energy_penalty(delta: np.ndarray, sfreq: float, fmax_hz: float = 40.0) -> float:
    fft = np.fft.rfft(delta, axis=1)
    freqs = np.fft.rfftfreq(delta.shape[1], d=1.0 / sfreq)
    high_mask = freqs > fmax_hz
    if not np.any(high_mask):
        return 0.0
    high_energy = np.mean(np.abs(fft[:, high_mask]) ** 2)
    return float(high_energy)
