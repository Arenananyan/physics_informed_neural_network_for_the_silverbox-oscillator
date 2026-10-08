"""Plot train/test RMSE from the primary experiment CSV without rerunning models."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    """Read verified metrics and save a comparison chart with the zero baseline."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=root / "results/tables/experiment_results.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/figures/experiment_comparison.png",
    )
    args = parser.parse_args(argv)
    with args.csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The experiment table is empty.")
    # Display experiments by model, window length, then training steps.
    rows.sort(
        key=lambda row: (
            row["model"],
            int(row["window_size"]),
            int(row["training_steps"]),
        )
    )
    train = [float(row["train_rmse_v"]) for row in rows]
    test = [float(row["test_rmse_v"]) for row in rows]
    baselines = [float(row["zero_baseline_rmse_v"]) for row in rows]
    if not np.allclose(baselines, baselines[0], rtol=0, atol=1e-9):
        raise ValueError("Experiments must share the same test baseline.")
    labels = [
        f"{row['model'].split('_')[-1].upper()}\nW{row['window_size']} / S{row['training_steps']}"
        for row in rows
    ]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.19, train, width=0.38, label="Train RMSE", color="#6595bd")
    ax.bar(x + 0.19, test, width=0.38, label="Test RMSE", color="#e0a15a")
    ax.axhline(
        baselines[0], color="#b13f3f", linestyle="--", label="Zero test baseline"
    )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="RMSE (V)",
        title="Silverbox: nine saved experiments (continuous free-run evaluation)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print("Comparison figure saved to:", args.output)


if __name__ == "__main__":
    main()
