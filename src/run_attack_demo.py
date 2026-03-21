from __future__ import annotations

import json
from pathlib import Path

from .attack.greedy_attack import GreedySparseScoreAttack
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn


if __name__ == "__main__":
    out_cfg = OutputConfig()
    ckpt_path = out_cfg.baseline_model_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing baseline checkpoint at {ckpt_path}. Run `python -m src.run_baseline` first."
        )

    model, device, _ = load_eegnet_checkpoint(str(ckpt_path))
    score_fn = make_score_fn(model, device)

    baseline_cfg = BaselineConfig()
    bundle = load_moabb_windows(baseline_cfg)

    x, y, _ = bundle.valid_set[0]
    x_np = x.astype("float32")
    y_int = int(y)

    atk_cfg = AttackConfig()
    attack = GreedySparseScoreAttack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=atk_cfg.n_windows,
        support_budget_k=atk_cfg.support_budget_k,
        basis_rank_r=atk_cfg.basis_rank_r,
        basis_min_hz=atk_cfg.basis_min_hz,
        basis_max_hz=atk_cfg.basis_max_hz,
        max_outer_iters=atk_cfg.max_outer_iters,
        max_query_budget=atk_cfg.max_query_budget,
        spsa_steps=atk_cfg.spsa_steps,
        spsa_step_size=atk_cfg.spsa_step_size,
        spsa_perturb_scale=atk_cfg.spsa_perturb_scale,
        l2_weight=atk_cfg.l2_weight,
        tv_weight=atk_cfg.tv_weight,
        band_weight=atk_cfg.band_weight,
        max_coeff_abs=atk_cfg.max_coeff_abs,
        seed=baseline_cfg.random_seed,
    )

    result = attack.run(x_np, y_int)

    output = {
        "success": result.success,
        "final_margin": result.margin,
        "support": result.support,
        "queries_used": result.queries_used,
        "budget_exhausted": result.budget_exhausted,
    }
    out_path = Path(out_cfg.root) / "attack_demo_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(output)
    print(f"Saved: {out_path}")
