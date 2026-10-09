"""Plot the lowest-test-RMSE run for each model in the experiment CSV."""

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
    # Keep all runs in the CSV, but show only the best test score per model letter.
    baselines = [float(row["zero_baseline_rmse_v"]) for row in rows]
    if not np.allclose(baselines, baselines[0], rtol=0, atol=1e-9):
        raise ValueError("Experiments must share the same test baseline.")
    best_by_model = {}
    for row in rows:
        model = row["model"].split("_")[1].upper()
        previous = best_by_model.get(model)
        if previous is None or float(row["test_rmse_v"]) < float(previous["test_rmse_v"]):
            best_by_model[model] = row
    rows = [best_by_model[model] for model in sorted(best_by_model)]
    train = [float(row["train_rmse_v"]) for row in rows]
    test = [float(row["test_rmse_v"]) for row in rows]
    labels = [
        f"{row['model'].split('_')[1].upper()}\nW{row['window_size']} / S{row['training_steps']}"
        for row in rows
    ]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.4), 5))
    ax.bar(x - 0.19, train, width=0.38, label="Train RMSE", color="#6595bd")
    ax.bar(x + 0.19, test, width=0.38, label="Test RMSE", color="#e0a15a")
    ax.axhline(
        baselines[0], color="#b13f3f", linestyle="--", label="Zero test baseline"
    )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="RMSE (V)",
        title="Silverbox: best test RMSE for each model (continuous free run)",
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
