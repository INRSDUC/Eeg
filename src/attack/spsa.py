from __future__ import annotations

from collections.abc import Callable

import numpy as np


# def spsa_minimize(
#     objective: Callable[[np.ndarray], float],
#     x0: np.ndarray,
#     steps: int,
#     step_size: float,
#     perturb_scale: float,
#     clip_abs: float,
#     restarts: int = 1,
#     init_scale: float = 0.0,
#     seed: int = 0,
# ) -> tuple[np.ndarray, float]:
#     rng = np.random.default_rng(seed)
#     x0 = x0.astype(np.float32).copy()
#     best_x = np.clip(x0, -clip_abs, clip_abs)
#     best_val = float(objective(best_x))

#     n_restarts = max(int(restarts), 1)
#     for restart_idx in range(n_restarts):
#         if restart_idx == 0:
#             x = best_x.copy()
#             local_best_x = x.copy()
#             local_best_val = best_val
#         else:
#             noise = rng.normal(
#                 loc=0.0,
#                 scale=float(init_scale) * clip_abs,
#                 size=x0.shape,
#             ).astype(np.float32)
#             x = np.clip(best_x + noise, -clip_abs, clip_abs)
#             local_best_x = x.copy()
#             local_best_val = float(objective(x))

#         for _ in range(steps):
#             delta = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
#             x_plus = np.clip(x + perturb_scale * delta, -clip_abs, clip_abs)
#             x_minus = np.clip(x - perturb_scale * delta, -clip_abs, clip_abs)

#             f_plus = float(objective(x_plus))
#             f_minus = float(objective(x_minus))
#             ghat = (f_plus - f_minus) / (2.0 * perturb_scale) * delta

#             x = np.clip(x - step_size * ghat, -clip_abs, clip_abs)

#             val = float(objective(x))
#             if val < local_best_val:
#                 local_best_val = val
#                 local_best_x = x.copy()

#         if local_best_val < best_val:
#             best_val = local_best_val
#             best_x = local_best_x.copy()

#     return best_x, best_val
def spsa_minimize(
    objective,
    x0,
    steps,
    step_size,
    perturb_scale,
    clip_abs,
    restarts=1,
    init_scale=0.0,
    seed=0,
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if perturb_scale <= 0:
        raise ValueError("perturb_scale must be > 0")
    if clip_abs <= 0:
        raise ValueError("clip_abs must be > 0")

    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=np.float32)
    best_x = np.clip(x0, -clip_abs, clip_abs)
    best_val = float(objective(best_x))

    n_restarts = max(int(restarts), 1)

    for restart_idx in range(n_restarts):
        if restart_idx == 0:
            x = best_x.copy()
        else:
            noise = rng.normal(
                0.0,
                float(init_scale) * clip_abs,
                size=x0.shape,
            ).astype(np.float32)
            x = np.clip(best_x + noise, -clip_abs, clip_abs)

        local_best_x = x.copy()
        local_best_val = float(objective(x))

        for k in range(steps):
            ak = step_size / np.sqrt(k + 1.0)
            ck = perturb_scale / (k + 1.0) ** 0.101

            delta = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)

            x_plus = np.clip(x + ck * delta, -clip_abs, clip_abs)
            x_minus = np.clip(x - ck * delta, -clip_abs, clip_abs)

            f_plus = float(objective(x_plus))
            f_minus = float(objective(x_minus))

            ghat = ((f_plus - f_minus) / (2.0 * ck)) * delta
            x = np.clip(x - ak * ghat, -clip_abs, clip_abs)

            val = float(objective(x))
            if val < local_best_val:
                local_best_val = val
                local_best_x = x.copy()

        if local_best_val < best_val:
            best_val = local_best_val
            best_x = local_best_x.copy()

    return best_x, best_val