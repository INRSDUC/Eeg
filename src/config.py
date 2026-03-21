from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineConfig:
    model_name: str = "EEGConformer"
    dataset_name: str = "BNCI2014_001"
    subject_ids: tuple[int, ...] = (1, 2, 3)
    stability_seeds: tuple[int, ...] = (7, 11, 19)
    sfreq: float = 128.0
    filter_low_hz: float = 4.0
    filter_high_hz: float = 38.0
    use_exponential_moving_standardize: bool = True
    standardize_factor_new: float = 1e-3
    standardize_init_block_size: int = 1000
    trial_start_offset_seconds: float = -0.5
    window_size_seconds: float = 4.0
    window_stride_seconds: float = 4.0
    trialwise_decoding: bool = True
    n_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "Adam"
    criterion_name: str = "CrossEntropyLoss"
    label_smoothing: float = 0.1
    early_stopping_patience: int = 12
    early_stopping_monitor: str = "valid_accuracy"
    random_seed: int = 7
    train_fraction: float = 0.8


@dataclass
class AttackConfig:
    n_windows: int = 8
    support_budget_k: int = 5
    basis_rank_r: int = 4
    basis_min_hz: float = 2.0
    basis_max_hz: float = 30.0
    max_outer_iters: int = 5
    max_query_budget: int | None = 5000
    spsa_steps: int = 80
    spsa_step_size: float = 0.04
    spsa_perturb_scale: float = 0.02
    l2_weight: float = 1e-3
    tv_weight: float = 1e-3
    band_weight: float = 1e-3
    max_coeff_abs: float = 0.2


@dataclass
class OutputConfig:
    root: Path = Path("outputs")
    baseline_model_name: str = "eegconformer_baseline.pt"
    baseline_metrics_name: str = "baseline_metrics.json"
    baseline_scores_name: str = "baseline_scores.npz"
    baseline_multiseed_summary_name: str = "baseline_multiseed_summary.json"

    @property
    def baseline_model_path(self) -> Path:
        return self.root / self.baseline_model_name

    @property
    def baseline_metrics_path(self) -> Path:
        return self.root / self.baseline_metrics_name

    @property
    def baseline_scores_path(self) -> Path:
        return self.root / self.baseline_scores_name

    @property
    def baseline_multiseed_summary_path(self) -> Path:
        return self.root / self.baseline_multiseed_summary_name

    def baseline_model_path_for_seed(self, seed: int) -> Path:
        return self.root / f"eegconformer_baseline_seed{seed}.pt"

    def baseline_metrics_path_for_seed(self, seed: int) -> Path:
        return self.root / f"baseline_metrics_seed{seed}.json"

    def baseline_scores_path_for_seed(self, seed: int) -> Path:
        return self.root / f"baseline_scores_seed{seed}.npz"
