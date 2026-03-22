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
    out_path: Path,
) -> dict:
    _configure_plot_style()
    channel_values = np.zeros((len(EEG_CHANNEL_NAMES_22),), dtype=np.float64)
    for ch_idx in range(len(EEG_CHANNEL_NAMES_22)):
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

    top_channels = np.argsort(channel_values)[::-1][:4]
    for ch_idx in top_channels:
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
    ax.set_title("Sparse Channel Attack: Selected Electrodes and Perturbation Power")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "top_channels": [
            {
                "channel_index": int(ch_idx),
                "channel_name": EEG_CHANNEL_NAMES_22[ch_idx],
                "power_ratio_pct": float(channel_values[ch_idx]),
            }
            for ch_idx in top_channels
        ]
    }


def _make_sparse_channel_time_waveform_plot(
    x_np: np.ndarray,
    result,
    sample_idx: int,
    out_path: Path,
    sfreq: float,
) -> dict:
    _configure_plot_style()
    unique_channels = []
    for atom in result.support:
        channel = int(atom[0]) if isinstance(atom, tuple) else int(atom)
        if channel not in unique_channels:
            unique_channels.append(channel)
        if len(unique_channels) == 4:
            break
    if len(unique_channels) < 4:
        ranked = list(np.argsort(np.linalg.norm(result.delta, axis=1))[::-1])
        for channel in ranked:
            channel = int(channel)
            if channel not in unique_channels:
                unique_channels.append(channel)
            if len(unique_channels) == 4:
                break

    t = np.arange(x_np.shape[1], dtype=np.float32) / sfreq
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 8.4), sharex=True)

    channel_rows = []
    for ax, ch_idx in zip(axes, unique_channels[:4]):
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
    fig.suptitle(f"Sparse Channel-Time Attack Waveforms on Four Attacked Channels (sample {sample_idx})", y=0.995)
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
    sparse_channel_time_sample_idx = 37

    x_sparse, y_sparse, result_sparse = _rerun_report_sample(
        report=sparse_channel_report,
        sample_idx=sparse_channel_sample_idx,
        bundle=bundle,
        baseline_cfg=baseline_cfg,
        score_fn=score_fn,
    )
    x_sparse_time, y_sparse_time, result_sparse_time = _rerun_report_sample(
        report=sparse_channel_time_report,
        sample_idx=sparse_channel_time_sample_idx,
        bundle=bundle,
        baseline_cfg=baseline_cfg,
        score_fn=score_fn,
    )

    head_heatmap_path = out_cfg.root / "sparse_channel_attack_head_heatmap.png"
    head_summary = _make_head_heatmap(
        x_np=x_sparse,
        result=result_sparse,
        out_path=head_heatmap_path,
    )

    waveform_path = out_cfg.root / "sparse_channel_time_attack_waveforms_4ch.png"
    waveform_summary = _make_sparse_channel_time_waveform_plot(
        x_np=x_sparse_time,
        result=result_sparse_time,
        sample_idx=sparse_channel_time_sample_idx,
        out_path=waveform_path,
        sfreq=baseline_cfg.sfreq,
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
            "head_heatmap_top_channels": head_summary["top_channels"],
        },
        "sparse_channel_time_attack": {
            "display_name": "Sparse Channel-Time Attack",
            "example_sample_idx": sparse_channel_time_sample_idx,
            "example_true_label": int(y_sparse_time),
            "attacked_accuracy_by_k": [
                {
                    "k": row["k"],
                    "attacked_accuracy": row["attacked_accuracy"],
                }
                for row in sparse_channel_time_report["prefix_summary"]
            ],
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
