from __future__ import annotations

import numpy as np


class RaisedCosineBasis:
    def __init__(self, window_length: int, rank: int, f_min_hz: float, f_max_hz: float, sfreq: float):
        self.window_length = window_length
        self.rank = rank
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.sfreq = sfreq
        self._basis = self._build_basis()

    def _build_basis(self) -> np.ndarray:
        t = np.arange(self.window_length, dtype=np.float32) / self.sfreq
        envelope = np.hanning(self.window_length).astype(np.float32)
        freqs = np.linspace(self.f_min_hz, self.f_max_hz, self.rank, dtype=np.float32)
        phases = np.linspace(0.0, np.pi / 2.0, self.rank, dtype=np.float32)

        phi = []
        for f, p in zip(freqs, phases):
            wave = envelope * np.cos(2.0 * np.pi * f * t + p)
            phi.append(wave)
        phi = np.stack(phi, axis=0)

        norms = np.linalg.norm(phi, axis=1, keepdims=True) + 1e-8
        return phi / norms

    @property
    def matrix(self) -> np.ndarray:
        return self._basis.copy()


def synthesize_window_perturbation(coeffs: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    return coeffs @ basis_matrix
