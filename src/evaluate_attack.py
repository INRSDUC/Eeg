from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attack.greedy_attack import GreedySparseScoreAttack
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .defense.lightweight import (
    flag_suspicious_atoms_from_signal,
    localized_denoise,
    suppress_flagged_atoms,
)
from .model_oracle import load_eegnet_checkpoint, make_score_fn


def run_eval(n_samples: int = 20) -> dict:
    out_cfg = OutputConfig()
    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, device)

    baseline_cfg = BaselineConfig()
    bundle = load_moabb_windows(baseline_cfg)

    support_budgets = [2, 4, 6]
    query_budgets = [500, 1500]
    amplitude_caps = [0.1, 0.2]

    rows = []
    for k in support_budgets:
        for query_budget in query_budgets:
            for max_coeff_abs in amplitude_caps:
                cfg = AttackConfig(
                    support_budget_k=k,
                    max_query_budget=query_budget,
                    max_coeff_abs=max_coeff_abs,
                )
                n_clean_correct = 0
                n_success = 0
                n_success_after_denoise = 0
                n_success_after_filter = 0
                n_budget_exhausted = 0
                margins = []
                queries = []
                flagged_counts = []

                for i in range(min(n_samples, len(bundle.valid_set))):
                    x, y, _ = bundle.valid_set[i]
                    x_np = x.astype(np.float32)
                    y_int = int(y)

                    clean_scores = score_fn(x_np)
                    clean_pred = int(np.argmax(clean_scores))
                    if clean_pred != y_int:
                        continue

                    n_clean_correct += 1
                    attack = GreedySparseScoreAttack(
                        score_fn=score_fn,
                        sfreq=baseline_cfg.sfreq,
                        n_windows=cfg.n_windows,
                        support_budget_k=cfg.support_budget_k,
                        basis_rank_r=cfg.basis_rank_r,
                        basis_min_hz=cfg.basis_min_hz,
                        basis_max_hz=cfg.basis_max_hz,
                        max_outer_iters=cfg.max_outer_iters,
                        max_query_budget=cfg.max_query_budget,
                        spsa_steps=cfg.spsa_steps,
                        spsa_step_size=cfg.spsa_step_size,
                        spsa_perturb_scale=cfg.spsa_perturb_scale,
                        l2_weight=cfg.l2_weight,
                        tv_weight=cfg.tv_weight,
                        band_weight=cfg.band_weight,
                        max_coeff_abs=cfg.max_coeff_abs,
                        seed=baseline_cfg.random_seed + i,
                    )
                    result = attack.run(x_np, y_int)
                    adv_pred = int(np.argmax(score_fn(result.x_adv)))

                    denoised_pred = int(np.argmax(score_fn(localized_denoise(result.x_adv))))
                    flagged_atoms = flag_suspicious_atoms_from_signal(
                        result.x_adv,
                        n_windows=cfg.n_windows,
                    )
                    filtered_x = suppress_flagged_atoms(
                        result.x_adv,
                        flagged_atoms=flagged_atoms,
                        n_windows=cfg.n_windows,
                    )
                    filtered_pred = int(np.argmax(score_fn(filtered_x)))

                    n_success += int(adv_pred != y_int)
                    n_success_after_denoise += int(denoised_pred != y_int)
                    n_success_after_filter += int(filtered_pred != y_int)
                    n_budget_exhausted += int(result.budget_exhausted)
                    margins.append(float(result.margin))
                    queries.append(int(result.queries_used))
                    flagged_counts.append(len(flagged_atoms))

                denom = max(n_clean_correct, 1)
                rows.append(
                    {
                        "support_budget_k": k,
                        "query_budget": query_budget,
                        "max_coeff_abs": max_coeff_abs,
                        "n_clean_correct": n_clean_correct,
                        "attack_success_rate": n_success / denom,
                        "post_denoise_attack_success_rate": n_success_after_denoise / denom,
                        "post_suspicious_filter_attack_success_rate": n_success_after_filter / denom,
                        "budget_exhaustion_rate": n_budget_exhausted / denom,
                        "avg_margin": float(np.mean(margins)) if margins else 0.0,
                        "avg_queries": float(np.mean(queries)) if queries else 0.0,
                        "avg_flagged_atoms": float(np.mean(flagged_counts)) if flagged_counts else 0.0,
                    }
                )

    return {"n_eval_samples": n_samples, "results": rows}


if __name__ == "__main__":
    report = run_eval(n_samples=20)
    out_cfg = OutputConfig()
    out_path = Path(out_cfg.root) / "attack_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(report)
    print(f"Saved: {out_path}")
