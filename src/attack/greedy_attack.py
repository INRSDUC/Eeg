from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .basis import RaisedCosineBasis, synthesize_window_perturbation
from .losses import band_energy_penalty, tv_regularizer, untargeted_margin
from .spsa import spsa_minimize
from .support import WindowPartition, all_atoms, make_window_partition


ScoreFn = Callable[[np.ndarray], np.ndarray]


class QueryBudgetExhausted(RuntimeError):
    pass


@dataclass
class AttackResult:
    x_adv: np.ndarray
    delta: np.ndarray
    support: list[tuple[int, int]]
    margin: float
    success: bool
    queries_used: int
    budget_exhausted: bool


class GreedySparseScoreAttack:
    def __init__(
        self,
        score_fn: ScoreFn,
        sfreq: float,
        n_windows: int,
        support_budget_k: int,
        basis_rank_r: int,
        basis_min_hz: float,
        basis_max_hz: float,
        max_outer_iters: int,
        max_query_budget: int | None,
        spsa_steps: int,
        spsa_step_size: float,
        spsa_perturb_scale: float,
        l2_weight: float,
        tv_weight: float,
        band_weight: float,
        max_coeff_abs: float,
        seed: int = 0,
    ):
        self.score_fn = score_fn
        self.sfreq = sfreq
        self.n_windows = n_windows
        self.support_budget_k = support_budget_k
        self.basis_rank_r = basis_rank_r
        self.basis_min_hz = basis_min_hz
        self.basis_max_hz = basis_max_hz
        self.max_outer_iters = max_outer_iters
        self.max_query_budget = max_query_budget
        self.spsa_steps = spsa_steps
        self.spsa_step_size = spsa_step_size
        self.spsa_perturb_scale = spsa_perturb_scale
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
        self.band_weight = band_weight
        self.max_coeff_abs = max_coeff_abs
        self.seed = seed
        self.queries_used = 0

    def _query_scores(self, x: np.ndarray) -> np.ndarray:
        if self.max_query_budget is not None and self.queries_used >= self.max_query_budget:
            raise QueryBudgetExhausted(
                f"Query budget exhausted at {self.queries_used} / {self.max_query_budget} queries."
            )
        self.queries_used += 1
        return self.score_fn(x)

    def _assemble_delta(
        self,
        n_channels: int,
        n_samples: int,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
    ) -> np.ndarray:
        delta = np.zeros((n_channels, n_samples), dtype=np.float32)
        for i, (c, w) in enumerate(support):
            s, e = partition.boundaries[w]
            local = synthesize_window_perturbation(coeffs[i], basis_by_window[w])
            delta[c, s:e] += local.astype(np.float32)
        return delta

    def _objective(
        self,
        x: np.ndarray,
        y: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
    ) -> float:
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
        )
        x_adv = x + delta
        scores = self._query_scores(x_adv)
        margin = untargeted_margin(scores, y)
        l2 = float(np.mean(coeffs**2))
        tv = tv_regularizer(delta)
        band = band_energy_penalty(delta, sfreq=self.sfreq)
        return margin + self.l2_weight * l2 + self.tv_weight * tv + self.band_weight * band

    def _build_result(
        self,
        x: np.ndarray,
        y: int,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        margin: float,
        budget_exhausted: bool,
    ) -> AttackResult:
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
        )
        return AttackResult(
            x_adv=x + delta,
            delta=delta,
            support=support,
            margin=margin,
            success=margin < 0.0,
            queries_used=self.queries_used,
            budget_exhausted=budget_exhausted,
        )

    def _refine_coeffs(
        self,
        x: np.ndarray,
        y: int,
        support: list[tuple[int, int]],
        init_coeffs: np.ndarray,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        if len(support) == 0:
            scores = self._query_scores(x)
            return init_coeffs, untargeted_margin(scores, y)

        flat0 = init_coeffs.reshape(-1)

        def f(flat: np.ndarray) -> float:
            coeffs = flat.reshape(len(support), self.basis_rank_r)
            return self._objective(x, y, support, coeffs, partition, basis_by_window)

        flat_best, value = spsa_minimize(
            objective=f,
            x0=flat0,
            steps=self.spsa_steps,
            step_size=self.spsa_step_size,
            perturb_scale=self.spsa_perturb_scale,
            clip_abs=self.max_coeff_abs,
            seed=self.seed,
        )
        return flat_best.reshape(len(support), self.basis_rank_r), value

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        self.queries_used = 0
        n_channels, n_samples = x.shape
        partition = make_window_partition(n_samples, self.n_windows)

        basis_by_window = {}
        for w, (s, e) in enumerate(partition.boundaries):
            basis = RaisedCosineBasis(
                window_length=e - s,
                rank=self.basis_rank_r,
                f_min_hz=self.basis_min_hz,
                f_max_hz=self.basis_max_hz,
                sfreq=self.sfreq,
            )
            basis_by_window[w] = basis.matrix

        support: list[tuple[int, int]] = []
        coeffs = np.zeros((0, self.basis_rank_r), dtype=np.float32)

        initial_scores = self._query_scores(x)
        current_margin = untargeted_margin(initial_scores, y)
        if current_margin < 0.0:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=False,
            )

        universe = all_atoms(n_channels, partition)
        selected = set()
        n_outer_iters = min(self.support_budget_k, self.max_outer_iters)

        try:
            for _ in range(n_outer_iters):
                best_candidate = None
                best_value = float("inf")
                best_candidate_coeffs = None

                for atom in universe:
                    if atom in selected:
                        continue
                    candidate_support = support + [atom]
                    candidate_coeffs = np.vstack([coeffs, np.zeros((1, self.basis_rank_r), dtype=np.float32)])

                    refined, value = self._refine_coeffs(
                        x=x,
                        y=y,
                        support=candidate_support,
                        init_coeffs=candidate_coeffs,
                        partition=partition,
                        basis_by_window=basis_by_window,
                    )
                    if value < best_value:
                        best_value = value
                        best_candidate = atom
                        best_candidate_coeffs = refined

                if best_candidate is None or best_candidate_coeffs is None:
                    break

                support.append(best_candidate)
                selected.add(best_candidate)
                coeffs = best_candidate_coeffs

                delta = self._assemble_delta(
                    n_channels=n_channels,
                    n_samples=n_samples,
                    partition=partition,
                    basis_by_window=basis_by_window,
                    support=support,
                    coeffs=coeffs,
                )
                current_margin = untargeted_margin(self._query_scores(x + delta), y)
                if current_margin < 0.0:
                    return self._build_result(
                        x=x,
                        y=y,
                        partition=partition,
                        basis_by_window=basis_by_window,
                        support=support,
                        coeffs=coeffs,
                        margin=current_margin,
                        budget_exhausted=False,
                    )
        except QueryBudgetExhausted:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=True,
            )

        return self._build_result(
            x=x,
            y=y,
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
            margin=current_margin,
            budget_exhausted=False,
        )
