from __future__ import annotations

import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms


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
        return _normalize_rows(phi)

    @property
    def matrix(self) -> np.ndarray:
        return self._basis.copy()


class FrequencyAtomBankBasis:
    def __init__(
        self,
        window_length: int,
        rank: int,
        f_min_hz: float,
        f_max_hz: float,
        sfreq: float,
        phase_count: int = 2,
    ):
        self.window_length = window_length
        self.rank = rank
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.sfreq = sfreq
        self.phase_count = max(int(phase_count), 1)
        self._basis = self._build_basis()

    def _build_basis(self) -> np.ndarray:
        t = np.arange(self.window_length, dtype=np.float32) / self.sfreq
        envelope = np.hanning(self.window_length).astype(np.float32)
        n_freqs = max(int(np.ceil(self.rank / self.phase_count)), 1)
        freqs = np.linspace(self.f_min_hz, self.f_max_hz, n_freqs, dtype=np.float32)
        phases = np.linspace(0.0, np.pi, self.phase_count, endpoint=False, dtype=np.float32)

        rows = []
        for freq in freqs:
            for phase in phases:
                rows.append((envelope * np.cos(2.0 * np.pi * freq * t + phase)).astype(np.float32))
        basis = np.stack(rows[: self.rank], axis=0)
        return _normalize_rows(basis)

    @property
    def matrix(self) -> np.ndarray:
        return self._basis.copy()


class SmoothResidualBasis:
    def __init__(self, window_length: int, rank: int):
        self.window_length = window_length
        self.rank = rank
        self._basis = self._build_basis()

    def _build_basis(self) -> np.ndarray:
        if self.rank <= 0:
            return np.zeros((0, self.window_length), dtype=np.float32)
        x = np.linspace(0.0, 1.0, self.window_length, dtype=np.float32)
        centers = np.linspace(0.0, 1.0, self.rank, dtype=np.float32)
        spacing = 1.0 / max(self.rank - 1, 1)
        width = max(2.0 * spacing, 1.0 / max(self.rank, 1))
        envelope = np.hanning(self.window_length).astype(np.float32)

        basis_rows = []
        for center in centers:
            hat = np.maximum(1.0 - np.abs(x - center) / width, 0.0)
            basis_rows.append((hat * envelope).astype(np.float32))
        return _normalize_rows(np.stack(basis_rows, axis=0))

    @property
    def matrix(self) -> np.ndarray:
        return self._basis.copy()


class HybridWaveformBasis:
    def __init__(self, window_length: int, rank: int, f_min_hz: float, f_max_hz: float, sfreq: float):
        self.window_length = window_length
        self.rank = rank
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.sfreq = sfreq
        self._basis = self._build_basis()

    def _build_trend_rows(self, n_rows: int) -> np.ndarray:
        if n_rows <= 0:
            return np.zeros((0, self.window_length), dtype=np.float32)
        x = np.linspace(-1.0, 1.0, self.window_length, dtype=np.float32)
        envelope = np.hanning(self.window_length).astype(np.float32)
        rows = []
        for degree in range(n_rows):
            poly = x**degree
            if degree > 0:
                poly = poly - np.mean(poly)
            rows.append((envelope * poly).astype(np.float32))
        return _normalize_rows(np.stack(rows, axis=0))

    def _build_basis(self) -> np.ndarray:
        sinusoid_rank = max(1, self.rank // 2)
        residual_rank = max(1, self.rank // 3)
        trend_rank = max(1, self.rank - sinusoid_rank - residual_rank)

        parts = [
            RaisedCosineBasis(
                window_length=self.window_length,
                rank=sinusoid_rank,
                f_min_hz=self.f_min_hz,
                f_max_hz=self.f_max_hz,
                sfreq=self.sfreq,
            ).matrix,
            SmoothResidualBasis(
                window_length=self.window_length,
                rank=residual_rank,
            ).matrix,
            self._build_trend_rows(trend_rank),
        ]
        basis = np.concatenate(parts, axis=0)
        if basis.shape[0] > self.rank:
            basis = basis[: self.rank]
        return _normalize_rows(basis.astype(np.float32, copy=False))

    @property
    def matrix(self) -> np.ndarray:
        return self._basis.copy()


def synthesize_window_perturbation(coeffs: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    return coeffs @ basis_matrix


def build_basis_matrix(
    basis_mode: str,
    window_length: int,
    rank: int,
    f_min_hz: float,
    f_max_hz: float,
    sfreq: float,
    phase_count: int = 2,
) -> np.ndarray:
    if basis_mode == "raised_cosine":
        return RaisedCosineBasis(
            window_length=window_length,
            rank=rank,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            sfreq=sfreq,
        ).matrix
    if basis_mode == "freq_atom_bank":
        return FrequencyAtomBankBasis(
            window_length=window_length,
            rank=rank,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            sfreq=sfreq,
            phase_count=phase_count,
        ).matrix
    if basis_mode == "hybrid":
        return HybridWaveformBasis(
            window_length=window_length,
            rank=rank,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            sfreq=sfreq,
        ).matrix
    raise ValueError(f"Unsupported basis_mode: {basis_mode}")
