from __future__ import annotations

import json
import os
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .attack.basis import build_basis_matrix, synthesize_window_perturbation
from .attack.greedy_attack import AttackResult, _apply_peak_ratio_constraint, build_score_attack
from .attack.losses import untargeted_margin
from .attack.support import make_window_partition
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn

_GLOBAL_SCORE_FN = None
_GLOBAL_BASELINE_CFG = None
_GLOBAL_ATTACK_CFG = None


def _power_ratio_percent(signal_l2: float, delta_l2: float) -> float:
    signal_power = max(float(signal_l2) ** 2, 1e-12)
    delta_power = float(delta_l2) ** 2
    return 100.0 * (delta_power / signal_power)


def _serialize_support(support: list[tuple[int, int] | int]) -> list[int | list[int]]:
    rows: list[int | list[int]] = []
    for atom in support:
        if isinstance(atom, tuple):
            rows.append([int(atom[0]), int(atom[1])])
        else:
            rows.append(int(atom))
    return rows


def _channel_index_from_atom(atom: tuple[int, int] | int) -> int:
    if isinstance(atom, tuple):
        return int(atom[0])
    return int(atom)


def _build_strong_report_config() -> AttackConfig:
    return AttackConfig(
        support_mode="channel_first",
        basis_mode="hybrid",
        basis_phase_count=2,
        n_windows=8,
        support_budget_k=8,
        basis_rank_r=8,
        channel_waveform_rank=32,
        max_outer_iters=8,
        max_query_budget=25000,
        spsa_steps=180,
        spsa_step_size=0.10,
        spsa_perturb_scale=0.05,
        spsa_restarts=4,
        spsa_init_scale=0.35,
        max_coeff_abs=1.00,
        max_perturbation_peak_ratio=0.05,
        candidate_probe_restarts=4,
        candidate_probe_scale=0.90,
        l2_weight=5e-5,
        tv_weight=5e-5,
        band_weight=5e-5,
    )


def _make_window_basis_cache(cfg: AttackConfig, n_samples: int, sfreq: float) -> tuple[list[tuple[int, int]], dict[int, np.ndarray]]:
    partition = make_window_partition(n_samples=n_samples, n_windows=cfg.n_windows)
    basis_by_window: dict[int, np.ndarray] = {}
    for w, (start, end) in enumerate(partition.boundaries):
        basis_by_window[w] = build_basis_matrix(
            basis_mode=cfg.basis_mode,
            window_length=end - start,
            rank=cfg.basis_rank_r,
            f_min_hz=cfg.basis_min_hz,
            f_max_hz=cfg.basis_max_hz,
            sfreq=sfreq,
            phase_count=cfg.basis_phase_count,
        )
    return partition.boundaries, basis_by_window


def _make_channel_basis_matrix(cfg: AttackConfig, n_samples: int, sfreq: float) -> np.ndarray:
    coeff_rank = cfg.channel_waveform_rank
    if coeff_rank is None:
        coeff_rank = max(int(cfg.basis_rank_r * cfg.n_windows), int(cfg.basis_rank_r))
    return build_basis_matrix(
        basis_mode=cfg.basis_mode,
        window_length=n_samples,
        rank=int(coeff_rank),
        f_min_hz=cfg.basis_min_hz,
        f_max_hz=cfg.basis_max_hz,
        sfreq=sfreq,
        phase_count=cfg.basis_phase_count,
    )


def _assemble_freqbank_prefix_delta(
    support: list[tuple[int, int] | int],
    coeffs: np.ndarray,
    k: int,
    n_channels: int,
    n_samples: int,
    boundaries: list[tuple[int, int]],
    basis_by_window: dict[int, np.ndarray],
) -> np.ndarray:
    delta = np.zeros((n_channels, n_samples), dtype=np.float32)
    for atom, atom_coeffs in zip(support[:k], coeffs[:k]):
        if not isinstance(atom, tuple):
            raise ValueError("Expected channel-window atoms for freq-bank prefix assembly")
        channel, window = int(atom[0]), int(atom[1])
        start, end = boundaries[window]
        local = atom_coeffs @ basis_by_window[window]
        delta[channel, start:end] += local.astype(np.float32)
    return delta


def _assemble_channel_prefix_delta(
    support: list[tuple[int, int] | int],
    coeffs: np.ndarray,
    k: int,
    n_channels: int,
    n_samples: int,
    basis_matrix: np.ndarray,
) -> np.ndarray:
    delta = np.zeros((n_channels, n_samples), dtype=np.float32)
    for atom, atom_coeffs in zip(support[:k], coeffs[:k]):
        if isinstance(atom, tuple):
            raise ValueError("Expected channel-only support for channel-first prefix assembly")
        waveform = synthesize_window_perturbation(atom_coeffs, basis_matrix)
        delta[int(atom), :] += waveform.astype(np.float32)
    return delta


def _compute_prefix_metrics(
    x: np.ndarray,
    y: int,
    result: AttackResult,
    score_fn,
    cfg: AttackConfig,
    baseline_cfg: BaselineConfig,
) -> list[dict]:
    if result.support and isinstance(result.support[0], tuple):
        boundaries, basis_by_window = _make_window_basis_cache(
            cfg=cfg,
            n_samples=x.shape[1],
            sfreq=baseline_cfg.sfreq,
        )
        basis_matrix = None
    else:
        boundaries, basis_by_window = [], {}
        basis_matrix = _make_channel_basis_matrix(
            cfg=cfg,
            n_samples=x.shape[1],
            sfreq=baseline_cfg.sfreq,
        )

    rows = []
    for k in range(1, len(result.support) + 1):
        if result.support and isinstance(result.support[0], tuple):
            delta_k = _assemble_freqbank_prefix_delta(
                support=result.support,
                coeffs=result.coeffs,
                k=k,
                n_channels=x.shape[0],
                n_samples=x.shape[1],
                boundaries=boundaries,
                basis_by_window=basis_by_window,
            )
        else:
            if basis_matrix is None:
                raise RuntimeError("Missing channel basis matrix for channel-first prefix evaluation")
            delta_k = _assemble_channel_prefix_delta(
                support=result.support,
                coeffs=result.coeffs,
                k=k,
                n_channels=x.shape[0],
                n_samples=x.shape[1],
                basis_matrix=basis_matrix,
            )
        delta_k = _apply_peak_ratio_constraint(
            x=x,
            delta=delta_k,
            max_perturbation_peak_ratio=cfg.max_perturbation_peak_ratio,
        )
        scores_k = score_fn(x + delta_k)
        pred_k = int(np.argmax(scores_k))
        rows.append(
            {
                "k": k,
                "pred": pred_k,
                "success": bool(pred_k != y),
                "margin": float(untargeted_margin(scores_k, y)),
                "delta_l2": float(np.linalg.norm(delta_k.reshape(-1))),
                "delta_linf": float(np.max(np.abs(delta_k))),
            }
        )
    return rows


def _binary_search_min_scale(
    x: np.ndarray,
    y: int,
    delta: np.ndarray,
    score_fn,
    n_steps: int = 12,
) -> tuple[bool, float, float, float]:
    scores_full = score_fn(x + delta)
    if int(np.argmax(scores_full)) == y:
        return False, 1.0, float(np.linalg.norm(delta.reshape(-1))), float(np.max(np.abs(delta)))

    lo, hi = 0.0, 1.0
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        scores_mid = score_fn(x + (mid * delta))
        if int(np.argmax(scores_mid)) != y:
            hi = mid
        else:
            lo = mid

    min_delta = hi * delta
    return True, float(hi), float(np.linalg.norm(min_delta.reshape(-1))), float(np.max(np.abs(min_delta)))


def _init_worker(checkpoint_path: str, baseline_cfg: BaselineConfig, attack_cfg: AttackConfig) -> None:
    global _GLOBAL_ATTACK_CFG, _GLOBAL_BASELINE_CFG, _GLOBAL_SCORE_FN
    torch.set_num_threads(1)
    model, device, _ = load_eegnet_checkpoint(checkpoint_path, device="cpu")
    _GLOBAL_SCORE_FN = make_score_fn(model, device)
    _GLOBAL_BASELINE_CFG = baseline_cfg
    _GLOBAL_ATTACK_CFG = attack_cfg


def _attack_one_sample(task: tuple[int, np.ndarray, int]) -> dict:
    score_fn = _GLOBAL_SCORE_FN
    baseline_cfg = _GLOBAL_BASELINE_CFG
    attack_cfg = _GLOBAL_ATTACK_CFG
    if score_fn is None or baseline_cfg is None or attack_cfg is None:
        raise RuntimeError("Worker globals are not initialized")

    idx, x_np, y_int = task

    attack = build_score_attack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=attack_cfg.n_windows,
        support_budget_k=attack_cfg.support_budget_k,
        basis_rank_r=attack_cfg.basis_rank_r,
        basis_min_hz=attack_cfg.basis_min_hz,
        basis_max_hz=attack_cfg.basis_max_hz,
        basis_mode=attack_cfg.basis_mode,
        basis_phase_count=attack_cfg.basis_phase_count,
        candidate_probe_restarts=attack_cfg.candidate_probe_restarts,
        candidate_probe_scale=attack_cfg.candidate_probe_scale,
        max_outer_iters=attack_cfg.max_outer_iters,
        max_query_budget=attack_cfg.max_query_budget,
        spsa_steps=attack_cfg.spsa_steps,
        spsa_step_size=attack_cfg.spsa_step_size,
        spsa_perturb_scale=attack_cfg.spsa_perturb_scale,
        spsa_restarts=attack_cfg.spsa_restarts,
        spsa_init_scale=attack_cfg.spsa_init_scale,
        l2_weight=attack_cfg.l2_weight,
        tv_weight=attack_cfg.tv_weight,
        band_weight=attack_cfg.band_weight,
        max_coeff_abs=attack_cfg.max_coeff_abs,
        max_perturbation_peak_ratio=attack_cfg.max_perturbation_peak_ratio,
        support_mode=attack_cfg.support_mode,
        channel_waveform_rank=attack_cfg.channel_waveform_rank,
        channel_shortlist_size=attack_cfg.channel_shortlist_size,
        enforce_unique_channels=attack_cfg.enforce_unique_channels,
        stop_on_success=attack_cfg.stop_on_success,
        seed=baseline_cfg.random_seed + idx,
    )
    result = attack.run(x_np, y_int)
    adv_scores = score_fn(result.x_adv)
    adv_pred = int(np.argmax(adv_scores))
    success = adv_pred != y_int
    if result.support and isinstance(result.support[0], tuple):
        boundaries, basis_by_window = _make_window_basis_cache(
            cfg=attack_cfg,
            n_samples=x_np.shape[1],
            sfreq=baseline_cfg.sfreq,
        )
        basis_matrix = None
    else:
        boundaries, basis_by_window = [], {}
        basis_matrix = _make_channel_basis_matrix(
            cfg=attack_cfg,
            n_samples=x_np.shape[1],
            sfreq=baseline_cfg.sfreq,
        )

    prefix_rows = _compute_prefix_metrics(
        x=x_np,
        y=y_int,
        result=result,
        score_fn=score_fn,
        cfg=attack_cfg,
        baseline_cfg=baseline_cfg,
    )
    for row in prefix_rows:
        if row["success"]:
            if result.support and isinstance(result.support[0], tuple):
                delta_k = _assemble_freqbank_prefix_delta(
                    support=result.support,
                    coeffs=result.coeffs,
                    k=int(row["k"]),
                    n_channels=x_np.shape[0],
                    n_samples=x_np.shape[1],
                    boundaries=boundaries,
                    basis_by_window=basis_by_window,
                )
            else:
                if basis_matrix is None:
                    raise RuntimeError("Missing channel basis matrix for channel-first prefix scaling")
                delta_k = _assemble_channel_prefix_delta(
                    support=result.support,
                    coeffs=result.coeffs,
                    k=int(row["k"]),
                    n_channels=x_np.shape[0],
                    n_samples=x_np.shape[1],
                    basis_matrix=basis_matrix,
                )
            delta_k = _apply_peak_ratio_constraint(
                x=x_np,
                delta=delta_k,
                max_perturbation_peak_ratio=attack_cfg.max_perturbation_peak_ratio,
            )
            _, min_scale, min_l2, min_linf = _binary_search_min_scale(
                x=x_np,
                y=y_int,
                delta=delta_k,
                score_fn=score_fn,
            )
            row["min_scale"] = float(min_scale)
            row["min_delta_l2"] = float(min_l2)
            row["min_delta_linf"] = float(min_linf)
        else:
            row["min_scale"] = None
            row["min_delta_l2"] = None
            row["min_delta_linf"] = None

    first_success = next((row for row in prefix_rows if row["success"]), None)
    return {
        "idx": idx,
        "true_label": y_int,
        "adv_pred": adv_pred,
        "success": bool(success),
        "final_margin": float(result.margin),
        "queries_used": int(result.queries_used),
        "budget_exhausted": bool(result.budget_exhausted),
        "support": _serialize_support(result.support),
        "signal_l2": float(np.linalg.norm(x_np.reshape(-1))),
        "delta_l2": float(np.linalg.norm(result.delta.reshape(-1))),
        "delta_linf": float(np.max(np.abs(result.delta))),
        "first_success_k": None if first_success is None else int(first_success["k"]),
        "first_success_min_l2": None if first_success is None else float(first_success["min_delta_l2"]),
        "prefix_rows": prefix_rows,
    }


def _plot_min_distortion_vs_sparsity(summary: dict, out_path: Path) -> None:
    prefix_rows = summary["prefix_summary"]
    ks = [row["k"] for row in prefix_rows]
    avg_power_ratio_pct = [
        np.nan if row["avg_min_power_ratio_pct_zero_accuracy"] is None else row["avg_min_power_ratio_pct_zero_accuracy"]
        for row in prefix_rows
    ]
    accuracy = [100.0 * row["attacked_accuracy"] for row in prefix_rows]
    zero_mask = [row["attacked_accuracy"] == 0.0 for row in prefix_rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.8))
    line_power = ax1.plot(
        ks,
        avg_power_ratio_pct,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Perturbation power (% of signal power)",
    )[0]
    zero_points = None
    if any(zero_mask):
        zero_points = ax1.scatter(
            [k for k, ok in zip(ks, zero_mask) if ok],
            [v for v, ok in zip(avg_power_ratio_pct, zero_mask) if ok],
            color="#d62728",
            s=70,
            zorder=4,
            label="0% attacked accuracy",
        )
    ax1.set_xlabel("Sparsity Budget K")
    ax1.set_ylabel("Added Perturbation Power (% of Signal Power)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Perturbation Power Needed to Fool EEGConformer on BNCI2014_001")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    line_accuracy = ax2.plot(
        ks,
        accuracy,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="#2ca02c",
        label="Attacked accuracy (%)",
    )[0]
    ax2.set_ylabel("Attacked Accuracy (%)")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(bottom=0.0)

    handles = [line_power]
    if zero_points is not None:
        handles.append(zero_points)
    handles.append(line_accuracy)
    labels = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_selected_channels(summary: dict, n_channels: int, out_path: Path) -> None:
    counts = np.asarray(summary["selected_channel_counts"], dtype=np.int64)

    plt.figure(figsize=(10, 4.8))
    xs = np.arange(n_channels)
    plt.bar(xs, counts, color="#9467bd", alpha=0.88)
    plt.xticks(xs)
    plt.xlabel("Channel Index")
    plt.ylabel("Selection Count")
    plt.title("Visualization of Selected Channels")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_full_freqbank_report() -> dict:
    global _GLOBAL_ATTACK_CFG, _GLOBAL_BASELINE_CFG, _GLOBAL_SCORE_FN

    out_cfg = OutputConfig()
    baseline_cfg = BaselineConfig()
    attack_cfg = _build_strong_report_config()
    torch.set_num_threads(1)
    requested_device = os.environ.get("EEG_ATTACK_DEVICE")

    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device=requested_device)
    score_fn = make_score_fn(model, device)
    bundle = load_moabb_windows(baseline_cfg)

    candidate_indices = []
    candidate_payloads = []
    for idx in range(len(bundle.valid_set)):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        clean_pred = int(np.argmax(score_fn(x_np)))
        if clean_pred == y_int:
            candidate_indices.append(idx)
            candidate_payloads.append((idx, x_np, y_int))

    print(f"Full test candidates: {len(candidate_indices)} / {len(bundle.valid_set)}", flush=True)

    per_sample = []
    n_success = 0
    channel_counts = np.zeros((bundle.n_chans,), dtype=np.int64)
    k_max = attack_cfg.support_budget_k
    prefix_correct_counts = np.zeros((k_max,), dtype=np.int64)
    prefix_l2_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_linf_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_power_ratio_pct_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_l2_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_linf_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_power_ratio_pct_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_success_counts = np.zeros((k_max,), dtype=np.int64)

    _GLOBAL_SCORE_FN = score_fn
    _GLOBAL_BASELINE_CFG = baseline_cfg
    _GLOBAL_ATTACK_CFG = attack_cfg

    cpu_count = os.cpu_count() or 1
    n_workers = max(1, min(8, cpu_count // 2 if cpu_count > 1 else 1))
    env_workers = os.environ.get("EEG_ATTACK_WORKERS")
    if env_workers is not None:
        n_workers = max(1, int(env_workers))

    def _consume(result_row: dict, order: int) -> None:
        nonlocal n_success
        per_sample.append(result_row)
        n_success += int(result_row["success"])
        for atom in result_row["support"]:
            channel = _channel_index_from_atom(tuple(atom) if isinstance(atom, list) else atom)
            channel_counts[channel] += 1
        prefix_rows = result_row["prefix_rows"]
        signal_l2 = float(result_row["signal_l2"])
        if not prefix_rows:
            raise RuntimeError("Expected at least one prefix row per attacked sample")
        for k_idx in range(k_max):
            effective_row = prefix_rows[min(k_idx, len(prefix_rows) - 1)]
            prefix_correct_counts[k_idx] += int(not effective_row["success"])
            prefix_l2_sums[k_idx] += float(effective_row["delta_l2"])
            prefix_linf_sums[k_idx] += float(effective_row["delta_linf"])
            prefix_power_ratio_pct_sums[k_idx] += _power_ratio_percent(signal_l2, float(effective_row["delta_l2"]))
            if effective_row["success"] and effective_row["min_delta_l2"] is not None:
                prefix_success_counts[k_idx] += 1
                prefix_min_l2_success_sums[k_idx] += float(effective_row["min_delta_l2"])
                prefix_min_linf_success_sums[k_idx] += float(effective_row["min_delta_linf"])
                prefix_min_power_ratio_pct_success_sums[k_idx] += _power_ratio_percent(
                    signal_l2,
                    float(effective_row["min_delta_l2"]),
                )
        if order % 10 == 0 or order == len(candidate_indices):
            attacked_accuracy = (order - n_success) / max(order, 1)
            print(
                f"[{order}/{len(candidate_indices)}] success_rate={n_success / order:.4f} attacked_accuracy={attacked_accuracy:.4f}",
                flush=True,
            )

    if n_workers == 1:
        for order, task in enumerate(candidate_payloads, start=1):
            _consume(_attack_one_sample(task), order)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(str(out_cfg.baseline_model_path), baseline_cfg, attack_cfg),
        ) as pool:
            for order, result_row in enumerate(pool.imap_unordered(_attack_one_sample, candidate_payloads, chunksize=1), start=1):
                _consume(result_row, order)

    n_candidates = len(candidate_indices)
    prefix_summary = []
    for k in range(1, k_max + 1):
        denom = max(n_candidates, 1)
        success_denom = int(prefix_success_counts[k - 1])
        prefix_summary.append(
            {
                "k": k,
                "success_rate": float(1.0 - (prefix_correct_counts[k - 1] / denom)),
                "attacked_accuracy": float(prefix_correct_counts[k - 1] / denom),
                "avg_delta_l2": float(prefix_l2_sums[k - 1] / denom),
                "avg_delta_linf": float(prefix_linf_sums[k - 1] / denom),
                "avg_delta_power_ratio_pct": float(prefix_power_ratio_pct_sums[k - 1] / denom),
                "avg_min_delta_l2_success_only": None
                if success_denom == 0
                else float(prefix_min_l2_success_sums[k - 1] / success_denom),
                "avg_min_delta_linf_success_only": None
                if success_denom == 0
                else float(prefix_min_linf_success_sums[k - 1] / success_denom),
                "avg_min_power_ratio_pct_success_only": None
                if success_denom == 0
                else float(prefix_min_power_ratio_pct_success_sums[k - 1] / success_denom),
                "avg_min_delta_l2_zero_accuracy": None
                if prefix_correct_counts[k - 1] != 0
                else float(prefix_min_l2_success_sums[k - 1] / denom),
                "avg_min_power_ratio_pct_zero_accuracy": None
                if prefix_correct_counts[k - 1] != 0
                else float(prefix_min_power_ratio_pct_success_sums[k - 1] / denom),
            }
        )

    per_sample.sort(key=lambda row: int(row["idx"]))

    report = {
        "dataset_name": baseline_cfg.dataset_name,
        "model_name": "EEGConformer",
        "n_valid_samples": len(bundle.valid_set),
        "n_clean_correct_attacked": n_candidates,
        "n_workers": n_workers,
        "score_device": str(device),
        "attack_config": attack_cfg.__dict__,
        "attack_success_rate": float(n_success / max(n_candidates, 1)),
        "attacked_accuracy": float((n_candidates - n_success) / max(n_candidates, 1)),
        "prefix_summary": prefix_summary,
        "selected_channel_counts": channel_counts.tolist(),
        "per_sample": per_sample,
    }

    out_dir = out_cfg.root
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "full_channel_attack_report.json"
    distortion_plot_path = out_dir / "full_channel_min_distortion_vs_sparsity.png"
    channel_plot_path = out_dir / "full_selected_channels.png"

    json_path.write_text(json.dumps(report, indent=2))
    _plot_min_distortion_vs_sparsity(report, distortion_plot_path)
    _plot_selected_channels(report, bundle.n_chans, channel_plot_path)

    report["report_path"] = str(json_path)
    report["distortion_plot_path"] = str(distortion_plot_path)
    report["channel_plot_path"] = str(channel_plot_path)
    return report


if __name__ == "__main__":
    summary = run_full_freqbank_report()
    print(
        {
            "attack_success_rate": summary["attack_success_rate"],
            "attacked_accuracy": summary["attacked_accuracy"],
            "report_path": summary["report_path"],
            "distortion_plot_path": summary["distortion_plot_path"],
            "channel_plot_path": summary["channel_plot_path"],
        }
    )
