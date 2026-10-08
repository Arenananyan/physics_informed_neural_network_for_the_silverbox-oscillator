"""Prepare the exact excerpt and centering specified in Homework 1 Task 5."""

from pathlib import Path
from scipy.io import loadmat
import numpy as np

# 1. Load dataset


def load_signals(data_path):
    """
    Load input and output signals from the MATLAB file.
    """
    data = loadmat(data_path)

    print("Available variables:", data.keys())

    u = data["V1"].ravel()
    y = data["V2"].ravel()

    return u, y


# 2. Extract required samples


def extract_excerpt(u, y, start, end):
    """
    Extract a specific segment of the signals.
    """
    u_excerpt = u[start:end]
    y_excerpt = y[start:end]

    return u_excerpt, y_excerpt


# 3. Mean centering


def center_signals(u, y):
    """
    Subtract the mean from each signal.
    """
    u_mean = np.mean(u)
    y_mean = np.mean(y)

    u_centered = u - u_mean
    y_centered = y - y_mean

    return u_centered, y_centered, u_mean, y_mean


# 4. Train / Test split


def split_data(u, y, train_size):
    """
    Split signals into training and testing segments.
    """
    u_train = u[:train_size]
    y_train = y[:train_size]

    u_test = u[train_size:]
    y_test = y[train_size:]

    return u_train, y_train, u_test, y_test


# 5. Save processed dataset


def save_processed_data(processed_path, u, y, fs, train_size, u_mean, y_mean):
    """
    Save the processed signals and metadata.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        processed_path,
        u=u,
        y=y,
        fs=fs,
        train_size=train_size,
        u_mean=u_mean,
        y_mean=y_mean,
    )

    print("Processed dataset saved to:", processed_path)


# Main


def main():

    # Project paths
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "raw" / "SNLS80mV.mat"

    processed_path = project_root / "data" / "processed" / "silverbox_excerpt.npz"

    # Dataset configuration
    fs = 610.35

    start = 42650
    end = 46746

    train_size = 3072

    # Load signals
    u, y = load_signals(data_path)

    print("Input shape:", u.shape)
    print("Output shape:", y.shape)

    print("First 5 input samples:", u[:5])
    print("First 5 output samples:", y[:5])

    # Extract excerpt
    u_excerpt, y_excerpt = extract_excerpt(u, y, start, end)

    print("Excerpt input shape:", u_excerpt.shape)
    print("Excerpt output shape:", y_excerpt.shape)

    # The assignment explicitly requires means from the whole 4096-sample excerpt.
    # Keep this centering before the chronological train/test split.
    u_centered, y_centered, u_mean, y_mean = center_signals(u_excerpt, y_excerpt)

    print("Original input mean:", u_mean)
    print("Original output mean:", y_mean)

    print("Centered input mean:", np.mean(u_centered))
    print("Centered output mean:", np.mean(y_centered))

    # Train / Test split
    u_train, y_train, u_test, y_test = split_data(u_centered, y_centered, train_size)

    print("Training input:", u_train.shape)
    print("Training output:", y_train.shape)

    print("Testing input:", u_test.shape)
    print("Testing output:", y_test.shape)

    # Save processed dataset
    save_processed_data(
        processed_path, u_centered, y_centered, fs, train_size, u_mean, y_mean
    )

    # Initial conditions for the simulation
    dt = 1.0 / fs

    y0 = y_centered[0]
    v0 = (y_centered[1] - y_centered[0]) / dt

    print("Initial output voltage (y0):", y0)
    print("Initial voltage derivative (v0):", v0)


if __name__ == "__main__":
    main()
