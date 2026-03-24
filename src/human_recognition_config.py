from __future__ import annotations

from pathlib import Path

from .config import BaselineConfig, OutputConfig


def build_bnci2014_001_human_recognition_config(model_name: str = "EEGConformer") -> BaselineConfig:
    return BaselineConfig(
        model_name=model_name,
        dataset_name="BNCI2014_001",
        subject_ids=tuple(range(1, 10)),
        target_mode="subject",
        evaluation_protocol="cross_session",
        train_session_name="0train",
        valid_session_name="1test",
        train_fraction=0.5,
    )


def build_bnci2014_001_human_recognition_output_config() -> OutputConfig:
    return OutputConfig(
        root=Path("outputs/bnci2014_001_human_recognition"),
        baseline_model_name="subject_recognition_baseline.pt",
        baseline_metrics_name="subject_recognition_metrics.json",
        baseline_scores_name="subject_recognition_scores.npz",
        baseline_multiseed_summary_name="subject_recognition_multiseed_summary.json",
    )
