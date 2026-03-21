from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from .config import BaselineConfig


@dataclass
class DatasetBundle:
    train_set: Subset
    valid_set: Subset
    n_chans: int
    n_classes: int
    input_window_samples: int


def _extract_shape_and_classes(windows_dataset) -> tuple[int, int, int]:
    x0, y0, _ = windows_dataset[0]
    n_chans = int(x0.shape[0])
    input_window_samples = int(x0.shape[1])
    metadata = windows_dataset.get_metadata()
    y_values = metadata["target"].to_numpy(dtype=np.int64, copy=False)
    n_classes = len(np.unique(y_values))
    return n_chans, n_classes, input_window_samples


def load_moabb_windows(cfg: BaselineConfig) -> DatasetBundle:
    dataset = MOABBDataset(dataset_name=cfg.dataset_name, subject_ids=list(cfg.subject_ids))

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda x: x * 1e6, apply_on_array=True),
        Preprocessor("filter", l_freq=cfg.filter_low_hz, h_freq=cfg.filter_high_hz),
        Preprocessor("resample", sfreq=cfg.sfreq),
    ]
    if cfg.use_exponential_moving_standardize:
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=cfg.standardize_factor_new,
                init_block_size=cfg.standardize_init_block_size,
            )
        )
    preprocess(dataset, preprocessors)

    trial_start_offset_samples = int(cfg.trial_start_offset_seconds * cfg.sfreq)
    window_size_samples = int(cfg.window_size_seconds * cfg.sfreq)
    if cfg.trialwise_decoding:
        window_stride_samples = window_size_samples
    else:
        window_stride_samples = int(cfg.window_stride_seconds * cfg.sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        preload=True,
    )
    metadata = windows_dataset.get_metadata()
    targets = metadata["target"].to_numpy(dtype=np.int64, copy=False)

    indices = np.arange(len(windows_dataset))
    train_idx, valid_idx = train_test_split(
        indices,
        train_size=cfg.train_fraction,
        random_state=cfg.random_seed,
        stratify=targets,
    )

    train_set = Subset(windows_dataset, train_idx.tolist())
    valid_set = Subset(windows_dataset, valid_idx.tolist())

    n_chans, n_classes, input_window_samples = _extract_shape_and_classes(windows_dataset)
    return DatasetBundle(
        train_set=train_set,
        valid_set=valid_set,
        n_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
    )
