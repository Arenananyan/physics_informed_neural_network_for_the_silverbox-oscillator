"""Evaluate saved weights with one continuous free run across the excerpt."""

import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AccelerationNetwork
from integrator import rollout

# Evaluation configuration: save this file, then press Run in VS Code.
# None selects the most recently saved checkpoint, including repeat training runs.
# To evaluate a specific model, replace None with its exact filename, for example:
# checkpoint_name = "model_e_w128_s2000.pt"
checkpoint_name = None

# False evaluates the selected model; True evaluates all saved experiments except
# the earlier extra model_a.pt run and rebuilds the comparison CSV.
evaluate_all = False

# True opens graphs; False only saves them (useful when no display is available).
show_plots = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_checkpoints(checkpoint_dir):
    """Resolve the editable selection and give a readable error for a missing file."""
    if evaluate_all:
        paths = sorted(
            path for path in checkpoint_dir.glob("*.pt") if path.name != "model_a.pt"
        )
    elif checkpoint_name is None:
        paths = list(checkpoint_dir.glob("*.pt"))
        # Modification time identifies the latest saved run; no filename guessing.
        paths = (
            [max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))]
            if paths
            else []
        )
    else:
        paths = [checkpoint_dir / checkpoint_name]
    if not paths:
        raise FileNotFoundError("No saved models found. Run train.py first.")
    for path in paths:
        if not path.is_file():
            available = ", ".join(sorted(p.name for p in checkpoint_dir.glob("*.pt")))
            raise FileNotFoundError(
                f"Checkpoint not found: {path.name}. Available: {available}"
            )
    return paths


def load_evaluation_data(path):
    """Load the entire centered excerpt; retain the boundary for scoring only."""
    with np.load(path) as data:
        u = torch.tensor(data["u"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float32)
        fs = float(data["fs"])
        train_size = int(data["train_size"])

    return u, y, 1.0 / fs, train_size


def calculate_rmse(predicted, measured):
    """Return voltage RMSE for matching predicted and measured segments."""
    return torch.sqrt(torch.mean((predicted - measured) ** 2)).item()


def plot_predictions(
    y, y_pred, dt, train_size, checkpoint, model_name, save_path, show=True
):
    """Plot the full continuous free run and its first 128 samples."""
    measured = y.detach().cpu().numpy()
    predicted = y_pred.detach().cpu().numpy()
    t = np.arange(len(y)) * dt

    hidden = int(checkpoint["hidden_size"])
    window = int(checkpoint["window_size"])
    steps = int(checkpoint["num_steps"])
    lr = float(checkpoint["learning_rate"])

    model_label = model_name.replace("_", " ").title()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    fig.suptitle(
        f"{model_label} (3-{hidden}-{hidden}-1) | "
        f"Window: {window} | Steps: {steps} | LR: {lr}",
        fontsize=13,
        fontweight="bold",
    )

    # Full free-run
    axes[0].plot(t, measured, label="Measured", linewidth=1)
    axes[0].plot(t, predicted, label="Predicted", linewidth=1)

    axes[0].axvline(
        train_size * dt, color="red", linestyle="--", label="Train/Test Boundary"
    )

    axes[0].set_title("Full Free-Run Prediction (4096 Samples)")

    # First 128 samples
    zoom = 128

    axes[1].plot(t[:zoom], measured[:zoom], label="Measured")
    axes[1].plot(t[:zoom], predicted[:zoom], label="Predicted")

    axes[1].set_title(f"First {zoom} Samples")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    print("Figure saved to:", save_path)


def plot_test_predictions(y, y_pred, dt, train_size, model_name, save_path, show=True):
    """Plot test output from the existing full run, without restarting its state."""
    t = np.arange(train_size, len(y)) * dt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y[train_size:].numpy(), label="Measured", linewidth=1)
    ax.plot(t, y_pred[train_size:].numpy(), label="Predicted", linewidth=1)
    ax.set(
        title=f"{model_name.replace('_', ' ').title()}: test segment (1024 samples)",
        xlabel="Time from excerpt start (s)",
        ylabel="Output voltage (V)",
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def evaluate_checkpoint(checkpoint_path, data_path, figure_dir, show=True):
    """Load a checkpoint, perform one free run, and return its metrics and paths."""
    u, y, dt, train_size = load_evaluation_data(data_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden = int(checkpoint["hidden_size"])
    window = int(checkpoint["window_size"])
    steps = int(checkpoint["num_steps"])
    # Older Model A checkpoints do not include a model name.
    model_name = checkpoint.get(
        "model_name", "_".join(checkpoint_path.stem.split("_")[:2])
    )
    model = AccelerationNetwork(hidden_size=hidden)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    y0 = y[0]
    v0 = (y[1] - y[0]) / dt
    with torch.no_grad():
        y_pred, _ = rollout(model, y0, v0, u, dt)
    if not torch.isfinite(y_pred).all():
        raise RuntimeError(f"Non-finite predictions: {checkpoint_path.name}")
    train_rmse = calculate_rmse(y_pred[:train_size], y[:train_size])
    test_rmse = calculate_rmse(y_pred[train_size:], y[train_size:])
    baseline = calculate_rmse(torch.zeros_like(y[train_size:]), y[train_size:])
    print(f"\nCheckpoint: {checkpoint_path.name}")
    print(f"Training RMSE: {train_rmse:.6f} V")
    print(f"Testing RMSE:  {test_rmse:.6f} V")
    print(f"Zero baseline: {baseline:.6f} V")
    full_figure = figure_dir / f"{checkpoint_path.stem}_free_run.png"
    test_figure = figure_dir / f"{checkpoint_path.stem}_test.png"
    plot_predictions(
        y, y_pred, dt, train_size, checkpoint, model_name, full_figure, show
    )
    plot_test_predictions(y, y_pred, dt, train_size, model_name, test_figure, show)
    root = data_path.parent.parent.parent

    def relative(path):
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path.resolve())

    return {
        "experiment_id": checkpoint_path.stem,
        "model": model_name,
        "architecture": f"3-{hidden}-{hidden}-1",
        "activation": "Tanh",
        "window_size": window,
        "training_steps": steps,
        "learning_rate": float(checkpoint["learning_rate"]),
        "train_rmse_v": f"{train_rmse:.9f}",
        "test_rmse_v": f"{test_rmse:.9f}",
        "zero_baseline_rmse_v": f"{baseline:.9f}",
        "checkpoint": relative(checkpoint_path),
        "full_run_figure": relative(full_figure),
        "test_figure": relative(test_figure),
        "metric_source": "recomputed_from_checkpoint",
    }


def main():
    """Evaluate the selected saved model without changing its weights or resetting state."""
    root = PROJECT_ROOT
    checkpoint_dir = root / "results" / "checkpoints"
    paths = select_checkpoints(checkpoint_dir)
    print("Evaluating:", ", ".join(path.name for path in paths))
    rows = [
        evaluate_checkpoint(
            path,
            root / "data/processed/silverbox_excerpt.npz",
            root / "results" / "figures",
            show=show_plots,
        )
        for path in paths
    ]
    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    # A single-model evaluation must not overwrite the complete comparison table.
    table_name = (
        "experiment_results.csv" if evaluate_all else f"{paths[0].stem}_results.csv"
    )
    with (table_dir / table_name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Results saved to:", table_dir / table_name)


if __name__ == "__main__":
    main()
