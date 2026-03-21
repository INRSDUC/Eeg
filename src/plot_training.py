from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def generate_training_plots(metrics_path: Path) -> dict[str, str]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    history = metrics.get("history", [])
    if not history:
        raise ValueError(f"No training history found in {metrics_path}")

    out_dir = metrics_path.parent
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    lrs = [row.get("lr", 0.0) for row in history]

    loss_path = out_dir / "train_val_loss_curve.png"
    best_loss_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2.0, color="#1f77b4")
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2.0, color="#d62728")
    plt.scatter([epochs[best_loss_idx]], [val_loss[best_loss_idx]], color="#d62728", s=50, zorder=3)
    plt.annotate(
        f"Best val loss: {val_loss[best_loss_idx]:.3f}\nEpoch {epochs[best_loss_idx]}",
        (epochs[best_loss_idx], val_loss[best_loss_idx]),
        textcoords="offset points",
        xytext=(10, -18),
        fontsize=9,
    )
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()

    acc_path = out_dir / "train_val_accuracy_curve.png"
    best_acc_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2.0, color="#2ca02c")
    plt.plot(epochs, val_acc, label="Val Accuracy", linewidth=2.0, color="#9467bd")
    plt.scatter([epochs[best_acc_idx]], [val_acc[best_acc_idx]], color="#9467bd", s=50, zorder=3)
    plt.annotate(
        f"Best val acc: {val_acc[best_acc_idx]:.3f}\nEpoch {epochs[best_acc_idx]}",
        (epochs[best_acc_idx], val_acc[best_acc_idx]),
        textcoords="offset points",
        xytext=(10, -18),
        fontsize=9,
    )
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()

    lr_path = out_dir / "learning_rate_curve.png"
    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs, lrs, linewidth=2.0, color="#ff7f0e")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(lr_path, dpi=180)
    plt.close()

    return {
        "loss_plot_path": str(loss_path),
        "accuracy_plot_path": str(acc_path),
        "lr_plot_path": str(lr_path),
    }


if __name__ == "__main__":
    metrics_path = Path("outputs/baseline_metrics.json")
    plots = generate_training_plots(metrics_path)
    print(plots)
