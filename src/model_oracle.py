from __future__ import annotations

import numpy as np
import torch
from braindecode.models import EEGConformer
try:
    from braindecode.models import EEGNet
except ImportError:  # pragma: no cover - compatibility with older braindecode releases
    from braindecode.models import EEGNetv4 as EEGNet


def load_baseline_checkpoint(path: str, device: str | None = None):
    checkpoint = torch.load(path, map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model_name = checkpoint.get("model_name", "EEGNet")
    if model_name in {"EEGNet", "EEGNetv4"}:
        model = EEGNet(
            n_chans=int(checkpoint["n_chans"]),
            n_outputs=int(checkpoint["n_classes"]),
            n_times=int(checkpoint["input_window_samples"]),
        ).to(torch_device)
    elif model_name == "EEGConformer":
        model = EEGConformer(
            n_chans=int(checkpoint["n_chans"]),
            n_outputs=int(checkpoint["n_classes"]),
            n_times=int(checkpoint["input_window_samples"]),
        ).to(torch_device)
    else:
        raise ValueError(f"Unsupported checkpoint model_name: {model_name}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, torch_device, checkpoint


def load_eegnet_checkpoint(path: str, device: str | None = None):
    return load_baseline_checkpoint(path, device=device)


def make_score_fn(model, device):
    def score_fn(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.as_tensor(x[None, ...], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_tensor)
        return logits.squeeze(0).detach().cpu().numpy()

    return score_fn
