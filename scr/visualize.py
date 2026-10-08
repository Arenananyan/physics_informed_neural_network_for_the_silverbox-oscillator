"""Plot the centered input/output and the chronological train/test split."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_processed_data(data_path):
    """Load previously processed Silverbox signals."""

    with np.load(data_path) as data:
        u = data["u"]
        y = data["y"]
        fs = float(data["fs"])
        train_size = int(data["train_size"])

    return u, y, fs, train_size


def plot_signals(u, y, fs, train_size, save_path, show=True):
    """Plot input and output signals over time."""

    # Time axis
    t = np.arange(len(u)) / fs

    # Train/test boundary
    split_time = train_size / fs

    # Create two plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Input voltage
    axes[0].plot(t, u, linewidth=0.8, label="Input voltage")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Silverbox Input Signal")
    axes[0].grid(True)
    axes[0].legend()

    # Output voltage
    axes[1].plot(t, y, linewidth=0.8, label="Output voltage")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Silverbox Output Signal")
    axes[1].grid(True)
    axes[1].legend()

    # Mark training/test boundary
    for ax in axes:

        # Vertical boundary line
        ax.axvline(split_time, color="red", linestyle="--", linewidth=1.5, zorder=3)

        # Text background style
        text_box = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=4)

        # Training region label
        ax.text(
            split_time / 2,
            0.95,
            "Training",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=text_box,
            zorder=5,
        )

        # Testing region label
        ax.text(
            (split_time + t[-1]) / 2,
            0.95,
            "Testing",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=text_box,
            zorder=5,
        )

        # Boundary label
        ax.text(
            split_time + 0.05,
            0.80,
            "Train/Test Boundary",
            transform=ax.get_xaxis_transform(),
            color="red",
            fontsize=9,
            bbox=text_box,
            zorder=5,
        )

    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300)

    print("Figure saved to:", save_path)

    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "processed" / "silverbox_excerpt.npz"

    save_path = project_root / "results" / "figures" / "silverbox_signals.png"

    u, y, fs, train_size = load_processed_data(data_path)

    plot_signals(u, y, fs, train_size, save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
