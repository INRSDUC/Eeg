from __future__ import annotations

from collections.abc import Callable

import numpy as np


def spsa_minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    steps: int,
    step_size: float,
    perturb_scale: float,
    clip_abs: float,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = x0.astype(np.float32).copy()
    best_x = x.copy()
    best_val = float(objective(x))

    for _ in range(steps):
        delta = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
        x_plus = np.clip(x + perturb_scale * delta, -clip_abs, clip_abs)
        x_minus = np.clip(x - perturb_scale * delta, -clip_abs, clip_abs)

        f_plus = float(objective(x_plus))
        f_minus = float(objective(x_minus))
        ghat = (f_plus - f_minus) / (2.0 * perturb_scale) * delta

        x = np.clip(x - step_size * ghat, -clip_abs, clip_abs)

        val = float(objective(x))
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val
