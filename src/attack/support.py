from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowPartition:
    n_samples: int
    n_windows: int
    boundaries: list[tuple[int, int]]


def make_window_partition(n_samples: int, n_windows: int) -> WindowPartition:
    if n_windows <= 0:
        raise ValueError("n_windows must be > 0")
    edges = np.linspace(0, n_samples, n_windows + 1, dtype=int)
    bounds = []
    for i in range(n_windows):
        bounds.append((int(edges[i]), int(edges[i + 1])))
    return WindowPartition(n_samples=n_samples, n_windows=n_windows, boundaries=bounds)


def all_atoms(n_channels: int, partition: WindowPartition) -> list[tuple[int, int]]:
    atoms = []
    for c in range(n_channels):
        for w in range(partition.n_windows):
            atoms.append((c, w))
    return atoms


def build_mask(n_channels: int, n_samples: int, support: set[tuple[int, int]], partition: WindowPartition) -> np.ndarray:
    mask = np.zeros((n_channels, n_samples), dtype=np.float32)
    for c, w in support:
        s, e = partition.boundaries[w]
        mask[c, s:e] = 1.0
    return mask
