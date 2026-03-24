from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from .attack.greedy_attack import build_score_attack
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn


EEG_CHANNEL_NAMES_22 = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _amplitude_ratio_percent_from_power_ratio(power_ratio_percent: float) -> float:
    return 100.0 * math.sqrt(max(power_ratio_percent, 0.0) / 100.0)


def _channel_l2_ratio_percent(x_ch: np.ndarray, delta_ch: np.ndarray) -> float:
    signal_l2 = max(float(np.linalg.norm(x_ch)), 1e-12)
    delta_l2 = float(np.linalg.norm(delta_ch))
    return 100.0 * (delta_l2 / signal_l2)


def _channel_power_ratio_percent(x_ch: np.ndarray, delta_ch: np.ndarray) -> float:
    signal_power = max(float(np.sum(np.square(x_ch))), 1e-12)
    delta_power = float(np.sum(np.square(delta_ch)))
    return 100.0 * (delta_power / signal_power)


def _build_attack_from_report(score_fn, baseline_cfg: BaselineConfig, report: dict, sample_idx: int):
    attack_cfg = AttackConfig(**report["attack_config"])
    return build_score_attack(
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
        seed=baseline_cfg.random_seed + sample_idx,
    )


def _rerun_report_sample(
    report: dict,
    sample_idx: int,
    bundle,
    baseline_cfg: BaselineConfig,
    score_fn,
) -> tuple[np.ndarray, int, object]:
    x, y, _ = bundle.valid_set[sample_idx]
    x_np = np.asarray(x, dtype=np.float32)
    y_int = int(y)
    attack = _build_attack_from_report(
        score_fn=score_fn,
        baseline_cfg=baseline_cfg,
        report=report,
        sample_idx=sample_idx,
    )
    result = attack.run(x_np, y_int)
    return x_np, y_int, result


def _unique_channels_from_support(support) -> list[int]:
    selected = []
    for atom in support:
        channel = int(atom[0]) if isinstance(atom, tuple) else int(atom)
        if channel not in selected:
            selected.append(channel)
    return selected


def _find_distinct_channel_example(
    report: dict,
    bundle,
    baseline_cfg: BaselineConfig,
    score_fn,
    target_channels: int,
    max_candidates: int = 200,
) -> tuple[int, np.ndarray, int, object, list[int]]:
    attack_cfg = AttackConfig(**report["attack_config"])
    constrained_cfg = AttackConfig(**{**attack_cfg.__dict__})
    constrained_cfg.support_budget_k = target_channels
    constrained_cfg.max_outer_iters = target_channels
    constrained_cfg.enforce_unique_channels = True
    constrained_cfg.stop_on_success = False
    constrained_cfg.support_mode = "channel_then_window"
    constrained_cfg.channel_shortlist_size = max(int(constrained_cfg.channel_shortlist_size or 0), target_channels + 4)
    constrained_cfg.max_query_budget = max(int(attack_cfg.max_query_budget or 0), 70000) if attack_cfg.max_query_budget is not None else 70000
    constrained_cfg.spsa_steps = max(int(attack_cfg.spsa_steps), 240)
    constrained_cfg.spsa_restarts = max(int(attack_cfg.spsa_restarts), 4)
    constrained_cfg.candidate_probe_restarts = max(int(attack_cfg.candidate_probe_restarts), 5)
    constrained_cfg.max_coeff_abs = max(float(attack_cfg.max_coeff_abs), 1.25)

    candidate_rows = sorted(
        report["per_sample"],
        key=lambda row: (
            0 if row["success"] else 1,
            row["first_success_k"] if row["first_success_k"] is not None else 999,
            row["final_margin"],
        ),
    )

    for row in candidate_rows[:max_candidates]:
        sample_idx = int(row["idx"])
        x, y, _ = bundle.valid_set[sample_idx]
        x_np = np.asarray(x, dtype=np.float32)
        y_int = int(y)

        attack = build_score_attack(
            score_fn=score_fn,
            sfreq=baseline_cfg.sfreq,
            n_windows=constrained_cfg.n_windows,
            support_budget_k=constrained_cfg.support_budget_k,
            basis_rank_r=constrained_cfg.basis_rank_r,
            basis_min_hz=constrained_cfg.basis_min_hz,
            basis_max_hz=constrained_cfg.basis_max_hz,
            basis_mode=constrained_cfg.basis_mode,
            basis_phase_count=constrained_cfg.basis_phase_count,
            candidate_probe_restarts=constrained_cfg.candidate_probe_restarts,
            candidate_probe_scale=constrained_cfg.candidate_probe_scale,
            max_outer_iters=constrained_cfg.max_outer_iters,
            max_query_budget=constrained_cfg.max_query_budget,
            spsa_steps=constrained_cfg.spsa_steps,
            spsa_step_size=constrained_cfg.spsa_step_size,
            spsa_perturb_scale=constrained_cfg.spsa_perturb_scale,
            spsa_restarts=constrained_cfg.spsa_restarts,
            spsa_init_scale=constrained_cfg.spsa_init_scale,
            l2_weight=constrained_cfg.l2_weight,
            tv_weight=constrained_cfg.tv_weight,
            band_weight=constrained_cfg.band_weight,
            max_coeff_abs=constrained_cfg.max_coeff_abs,
            max_perturbation_peak_ratio=constrained_cfg.max_perturbation_peak_ratio,
            support_mode=constrained_cfg.support_mode,
            channel_waveform_rank=constrained_cfg.channel_waveform_rank,
            channel_shortlist_size=constrained_cfg.channel_shortlist_size,
            enforce_unique_channels=constrained_cfg.enforce_unique_channels,
            stop_on_success=constrained_cfg.stop_on_success,
            seed=baseline_cfg.random_seed + sample_idx,
        )
        result = attack.run(x_np, y_int)
        selected_channels = _unique_channels_from_support(result.support)
        if result.success and len(selected_channels) >= target_channels:
            return sample_idx, x_np, y_int, result, selected_channels[:target_channels]

    raise RuntimeError(
        f"Could not find a successful constrained example with {target_channels} distinct channels "
        f"within the first {max_candidates} candidate samples."
    )


def _make_accuracy_plot(
    sparse_channel_report: dict,
    sparse_channel_time_report: dict,
    out_path: Path,
) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    series = [
        {
            "label": "Sparse Channel Attack",
            "color": "#1f77b4",
            "marker": "o",
            "report": sparse_channel_report,
        },
        {
            "label": "Sparse Channel-Time Attack",
            "color": "#2ca02c",
            "marker": "s",
            "report": sparse_channel_time_report,
        },
    ]

    for entry in series:
        ks = [row["k"] for row in entry["report"]["prefix_summary"]]
        attacked_accuracy_pct = [100.0 * row["attacked_accuracy"] for row in entry["report"]["prefix_summary"]]
        ax.plot(
            ks,
            attacked_accuracy_pct,
            marker=entry["marker"],
            markersize=7.5,
            linewidth=2.4,
            color=entry["color"],
            label=entry["label"],
        )

    ax.set_xlabel("Sparsity Budget K")
    ax.set_ylabel("Attacked Accuracy (%)")
    ax.set_title("EEGConformer Under Sparse Attacks on BNCI2014_001")
    ax.set_xticks([row["k"] for row in sparse_channel_report["prefix_summary"]])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_head_heatmap(
    x_np: np.ndarray,
    result,
    selected_channels: list[int],
    out_path: Path,
    title: str,
) -> dict:
    _configure_plot_style()
    channel_values = np.zeros((len(EEG_CHANNEL_NAMES_22),), dtype=np.float64)
    for ch_idx in selected_channels:
        channel_values[ch_idx] = _channel_power_ratio_percent(x_np[ch_idx], result.delta[ch_idx])

    info = mne.create_info(ch_names=EEG_CHANNEL_NAMES_22, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    pos = np.array([info["chs"][idx]["loc"][:2] for idx in range(len(EEG_CHANNEL_NAMES_22))], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    im, _ = mne.viz.plot_topomap(
        channel_values,
        pos,
        axes=ax,
        show=False,
        contours=6,
        cmap="YlOrRd",
        sphere=0.095,
    )

    for ch_idx in selected_channels:
        xy = pos[ch_idx]
        ax.text(
            xy[0],
            xy[1] + 0.012,
            f"{EEG_CHANNEL_NAMES_22[ch_idx]}\n{channel_values[ch_idx]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.08)
    cbar.set_label("Added Perturbation Power (% of Channel Signal Power)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "selected_channels": [
            {
                "channel_index": int(ch_idx),
                "channel_name": EEG_CHANNEL_NAMES_22[ch_idx],
                "power_ratio_pct": float(channel_values[ch_idx]),
            }
            for ch_idx in selected_channels
        ]
    }


def _make_sparse_channel_waveform_plot(
    x_np: np.ndarray,
    result,
    sample_idx: int,
    selected_channels: list[int],
    out_path: Path,
    sfreq: float,
    title: str,
) -> dict:
    _configure_plot_style()

    t = np.arange(x_np.shape[1], dtype=np.float32) / sfreq
    n_channels = len(selected_channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(11.0, 1.95 * n_channels + 0.8), sharex=True)
    if n_channels == 1:
        axes = [axes]

    channel_rows = []
    for ax, ch_idx in zip(axes, selected_channels):
        original = x_np[ch_idx]
        attacked = result.x_adv[ch_idx]
        delta = result.delta[ch_idx]
        power_ratio_pct = _channel_power_ratio_percent(original, delta)
        amplitude_ratio_pct = _channel_l2_ratio_percent(original, delta)

        ax.plot(
            t,
            attacked,
            color="#d62728",
            linewidth=1.3,
            alpha=0.95,
            zorder=1,
            label="Attacked",
        )
        ax.plot(
            t,
            original,
            color="#1f77b4",
            linewidth=1.5,
            zorder=3,
            label="Original",
        )
        ax.plot(
            t,
            delta,
            color="black",
            linewidth=1.05,
            linestyle="--",
            zorder=2,
            label="Added perturbation",
        )
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{EEG_CHANNEL_NAMES_22[ch_idx]} | Power change: {power_ratio_pct:.2f}% | "
            f"Amplitude change: {amplitude_ratio_pct:.2f}%"
        )
        ax.grid(True, alpha=0.25)
        channel_rows.append(
            {
                "channel_index": int(ch_idx),
                "channel_name": EEG_CHANNEL_NAMES_22[ch_idx],
                "power_ratio_pct": float(power_ratio_pct),
                "amplitude_ratio_pct": float(amplitude_ratio_pct),
            }
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} (sample {sample_idx})", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {"channels": channel_rows}


def plot_attack_accuracy_vs_sparsity() -> dict:
    out_cfg = OutputConfig()
    out_cfg.root.mkdir(parents=True, exist_ok=True)

    sparse_channel_report = _load_report(out_cfg.root / "unrestricted_hybrid_full_report.json")
    sparse_channel_time_report = _load_report(out_cfg.root / "channel_window_hybrid_full_report.json")

    accuracy_plot_path = out_cfg.root / "attack_accuracy_vs_sparsity.png"
    _make_accuracy_plot(
        sparse_channel_report=sparse_channel_report,
        sparse_channel_time_report=sparse_channel_time_report,
        out_path=accuracy_plot_path,
    )

    baseline_cfg = BaselineConfig()
    bundle = load_moabb_windows(baseline_cfg)
    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, device)

    sparse_channel_sample_idx = 59
    x_sparse, y_sparse, result_sparse = _rerun_report_sample(
        report=sparse_channel_report,
        sample_idx=sparse_channel_sample_idx,
        bundle=bundle,
        baseline_cfg=baseline_cfg,
        score_fn=score_fn,
    )
    sparse_channel_selected_channels = [int(ch) for ch in result_sparse.support[:4]]

    head_heatmap_path = out_cfg.root / "sparse_channel_attack_head_heatmap.png"
    head_summary = _make_head_heatmap(
        x_np=x_sparse,
        result=result_sparse,
        selected_channels=sparse_channel_selected_channels,
        out_path=head_heatmap_path,
        title="Sparse Channel Attack: Four Attacked Electrodes",
    )

    waveform_path = out_cfg.root / "sparse_channel_attack_waveforms_4ch.png"
    waveform_summary = _make_sparse_channel_waveform_plot(
        x_np=x_sparse,
        result=result_sparse,
        sample_idx=sparse_channel_sample_idx,
        selected_channels=sparse_channel_selected_channels,
        out_path=waveform_path,
        sfreq=baseline_cfg.sfreq,
        title="Sparse Channel Attack Waveforms on Four Attacked Channels",
    )

    sparse_channel_time_sample_idx, x_sparse_time, y_sparse_time, result_sparse_time, sparse_channel_time_selected_channels = _find_distinct_channel_example(
        report=sparse_channel_time_report,
        bundle=bundle,
        baseline_cfg=baseline_cfg,
        score_fn=score_fn,
        target_channels=6,
    )

    sparse_channel_time_head_heatmap_path = out_cfg.root / "sparse_channel_time_attack_head_heatmap_6ch.png"
    sparse_channel_time_head_summary = _make_head_heatmap(
        x_np=x_sparse_time,
        result=result_sparse_time,
        selected_channels=sparse_channel_time_selected_channels,
        out_path=sparse_channel_time_head_heatmap_path,
        title="Sparse Channel-Time Attack: Six Attacked Electrodes",
    )

    sparse_channel_time_waveform_path = out_cfg.root / "sparse_channel_time_attack_waveforms_6ch.png"
    sparse_channel_time_waveform_summary = _make_sparse_channel_waveform_plot(
        x_np=x_sparse_time,
        result=result_sparse_time,
        sample_idx=sparse_channel_time_sample_idx,
        selected_channels=sparse_channel_time_selected_channels,
        out_path=sparse_channel_time_waveform_path,
        sfreq=baseline_cfg.sfreq,
        title="Sparse Channel-Time Attack Waveforms on Six Attacked Channels",
    )

    top4 = sorted(
        enumerate(sparse_channel_report["selected_channel_counts"]),
        key=lambda row: row[1],
        reverse=True,
    )[:4]

    summary = {
        "figure_path": str(accuracy_plot_path),
        "head_heatmap_path": str(head_heatmap_path),
        "waveform_path": str(waveform_path),
        "sparse_channel_time_head_heatmap_path": str(sparse_channel_time_head_heatmap_path),
        "sparse_channel_time_waveform_path": str(sparse_channel_time_waveform_path),
        "dataset_name": sparse_channel_report["dataset_name"],
        "model_name": sparse_channel_report["model_name"],
        "sparse_channel_attack": {
            "display_name": "Sparse Channel Attack",
            "example_sample_idx": sparse_channel_sample_idx,
            "example_true_label": int(y_sparse),
            "attacked_accuracy_by_k": [
                {
                    "k": row["k"],
                    "attacked_accuracy": row["attacked_accuracy"],
                }
                for row in sparse_channel_report["prefix_summary"]
            ],
            "k4_attacked_accuracy": sparse_channel_report["prefix_summary"][3]["attacked_accuracy"],
            "k4_raw_delta_power_ratio_pct": sparse_channel_report["prefix_summary"][3]["avg_delta_power_ratio_pct"],
            "k4_raw_delta_amplitude_ratio_pct": _amplitude_ratio_percent_from_power_ratio(
                sparse_channel_report["prefix_summary"][3]["avg_delta_power_ratio_pct"]
            ),
            "k4_min_success_power_ratio_pct": sparse_channel_report["prefix_summary"][3]["avg_min_power_ratio_pct_zero_accuracy"],
            "k4_min_success_amplitude_ratio_pct": _amplitude_ratio_percent_from_power_ratio(
                sparse_channel_report["prefix_summary"][3]["avg_min_power_ratio_pct_zero_accuracy"]
            ),
            "top4_selected_channels_full_run": [
                {
                    "channel_index": int(channel),
                    "channel_name": EEG_CHANNEL_NAMES_22[channel],
                    "count": int(count),
                }
                for channel, count in top4
            ],
            "example_selected_channels": [
                {
                    "channel_index": int(ch_idx),
                    "channel_name": EEG_CHANNEL_NAMES_22[ch_idx],
                }
                for ch_idx in sparse_channel_selected_channels
            ],
            "head_heatmap_selected_channels": head_summary["selected_channels"],
        },
        "sparse_channel_time_attack": {
            "display_name": "Sparse Channel-Time Attack",
            "attacked_accuracy_by_k": [
                {
                    "k": row["k"],
                    "attacked_accuracy": row["attacked_accuracy"],
                }
                for row in sparse_channel_time_report["prefix_summary"]
            ],
            "example_sample_idx": sparse_channel_time_sample_idx,
            "example_true_label": int(y_sparse_time),
            "example_selected_channels": [
                {
                    "channel_index": int(ch_idx),
                    "channel_name": EEG_CHANNEL_NAMES_22[ch_idx],
                }
                for ch_idx in sparse_channel_time_selected_channels
            ],
            "head_heatmap_selected_channels": sparse_channel_time_head_summary["selected_channels"],
            "waveform_channels": sparse_channel_time_waveform_summary["channels"],
        },
        "sparse_channel_attack_waveforms": {
            "display_name": "Sparse Channel Attack Waveforms",
            "example_sample_idx": sparse_channel_sample_idx,
            "example_true_label": int(y_sparse),
            "waveform_channels": waveform_summary["channels"],
        },
    }

    summary_path = out_cfg.root / "attack_accuracy_vs_sparsity_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    result = plot_attack_accuracy_vs_sparsity()
    print(result)
