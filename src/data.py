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
from torch.utils.data import Dataset, Subset

from .config import BaselineConfig


@dataclass
class DatasetBundle:
    train_set: Dataset
    valid_set: Dataset
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


def _inverse_symmetric_matrix_sqrt(matrix: np.ndarray, eps: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return inv_sqrt.astype(np.float32, copy=False)


def _unpack_sample(sample):
    if len(sample) == 3:
        return sample[0], sample[1], sample[2]
    if len(sample) == 2:
        return sample[0], sample[1], None
    raise ValueError(f"Unexpected dataset sample structure with length {len(sample)}")


class EuclideanAlignedSubset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices: list[int],
        alignment_mats: dict[object, np.ndarray],
        groups: np.ndarray,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.alignment_mats = alignment_mats
        self.groups = groups
        self.default_group = next(iter(alignment_mats))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        x, y, extra = _unpack_sample(self.base_dataset[base_index])
        x_np = np.asarray(x, dtype=np.float32)
        group = self.groups[base_index]
        align = self.alignment_mats.get(group, self.alignment_mats[self.default_group])
        aligned = (align @ x_np).astype(np.float32, copy=False)
        if extra is None:
            return aligned, y
        return aligned, y, extra


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, cfg: BaselineConfig) -> None:
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        max_shift = int(self.cfg.aug_time_shift_max_samples)
        if max_shift <= 0 or self.rng.random() >= self.cfg.aug_time_shift_prob:
            return x
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return x
        shifted = np.zeros_like(x)
        if shift > 0:
            shifted[:, shift:] = x[:, :-shift]
        else:
            shifted[:, :shift] = x[:, -shift:]
        return shifted

    def _amplitude_jitter(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.cfg.aug_amplitude_jitter_prob:
            return x
        scale = float(self.rng.normal(loc=1.0, scale=self.cfg.aug_amplitude_jitter_std))
        return x * scale

    def _gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.cfg.aug_gaussian_noise_prob:
            return x
        signal_scale = max(float(np.std(x)), 1e-6)
        noise = self.rng.normal(
            loc=0.0,
            scale=self.cfg.aug_gaussian_noise_std * signal_scale,
            size=x.shape,
        )
        return x + noise.astype(np.float32)

    def _channel_dropout(self, x: np.ndarray) -> np.ndarray:
        dropout_prob = float(self.cfg.aug_channel_dropout_prob)
        if dropout_prob <= 0.0:
            return x
        mask = self.rng.random(x.shape[0]) >= dropout_prob
        if mask.all():
            return x
        dropped = x.copy()
        dropped[~mask, :] = 0.0
        return dropped

    def __getitem__(self, item: int):
        sample = self.base_dataset[item]
        x, y, extra = _unpack_sample(sample)
        x_aug = np.asarray(x, dtype=np.float32).copy()
        x_aug = self._time_shift(x_aug)
        x_aug = self._amplitude_jitter(x_aug)
        x_aug = self._gaussian_noise(x_aug)
        x_aug = self._channel_dropout(x_aug)
        if extra is None:
            return x_aug, y
        return x_aug, y, extra


def _compute_alignment_mats(
    windows_dataset,
    train_indices: np.ndarray,
    groups: np.ndarray,
    eps: float,
) -> dict[object, np.ndarray]:
    cov_sums: dict[object, np.ndarray] = {}
    counts: dict[object, int] = {}

    for index in train_indices.tolist():
        x, _, _ = _unpack_sample(windows_dataset[index])
        x_np = np.asarray(x, dtype=np.float32)
        cov = x_np @ x_np.T
        trace = float(np.trace(cov))
        if trace > eps:
            cov = cov / trace
        group = groups[index]
        if group not in cov_sums:
            cov_sums[group] = cov.astype(np.float64, copy=True)
            counts[group] = 1
        else:
            cov_sums[group] += cov
            counts[group] += 1

    return {
        group: _inverse_symmetric_matrix_sqrt(cov_sums[group] / counts[group], eps=eps)
        for group in cov_sums
    }


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
    if cfg.ea_group_by_subject and "subject" in metadata.columns:
        groups = metadata["subject"].to_numpy(copy=False)
    else:
        groups = np.asarray(["global"] * len(metadata), dtype=object)

    indices = np.arange(len(windows_dataset))
    train_idx, valid_idx = train_test_split(
        indices,
        train_size=cfg.train_fraction,
        random_state=cfg.random_seed,
        stratify=targets,
    )

    train_set: Dataset = Subset(windows_dataset, train_idx.tolist())
    valid_set: Dataset = Subset(windows_dataset, valid_idx.tolist())

    if cfg.use_euclidean_alignment:
        alignment_mats = _compute_alignment_mats(
            windows_dataset=windows_dataset,
            train_indices=train_idx,
            groups=groups,
            eps=cfg.ea_eps,
        )
        train_set = EuclideanAlignedSubset(
            base_dataset=windows_dataset,
            indices=train_idx.tolist(),
            alignment_mats=alignment_mats,
            groups=groups,
        )
        valid_set = EuclideanAlignedSubset(
            base_dataset=windows_dataset,
            indices=valid_idx.tolist(),
            alignment_mats=alignment_mats,
            groups=groups,
        )

    if cfg.use_data_augmentation:
        train_set = AugmentedDataset(train_set, cfg)

    n_chans, n_classes, input_window_samples = _extract_shape_and_classes(windows_dataset)
    return DatasetBundle(
        train_set=train_set,
        valid_set=valid_set,
        n_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
    )
