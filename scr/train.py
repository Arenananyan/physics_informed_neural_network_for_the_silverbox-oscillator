"""Train the acceleration network on randomly selected training windows."""

import argparse
from pathlib import Path
import numpy as np
import torch

from model import AccelerationNetwork
from integrator import rollout


def load_training_data(data_path):
    """Load only the training portion as float32 tensors and return the time step."""
    with np.load(data_path) as data:
        train_size = int(data["train_size"])
        fs = float(data["fs"])

        u_train = torch.tensor(data["u"][:train_size], dtype=torch.float32)
        y_train = torch.tensor(data["y"][:train_size], dtype=torch.float32)

    return u_train, y_train, 1.0 / fs


def main(argv=None):
    """Train with the selected configuration and save weights plus metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="model_e")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional new output path; existing checkpoints are protected.",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "silverbox_excerpt.npz"

    u_train, y_train, dt = load_training_data(data_path)

    y0 = y_train[0]
    v0 = (y_train[1] - y_train[0]) / dt

    print("Training input shape:", u_train.shape)
    print("Training output shape:", y_train.shape)
    print("Time step:", dt)
    print("Initial y:", y0.item())
    print("Initial v:", v0.item())

    # Training configuration
    model_name = args.model_name
    hidden_size = args.hidden_size
    num_steps = args.steps
    window_size = args.window
    learning_rate = args.learning_rate

    if hidden_size < 1 or num_steps < 1 or learning_rate <= 0:
        parser.error("Hidden size, steps and learning rate must be positive.")
    if not 2 <= window_size <= len(u_train):
        parser.error("Window must contain 2 to len(training data) samples.")
    checkpoint_path = args.checkpoint or (
        root
        / "results"
        / "checkpoints"
        / f"{model_name}_w{window_size}_s{num_steps}.pt"
    )
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Preserving existing experiment: {checkpoint_path}. "
            "Choose a new --checkpoint path."
        )

    torch.manual_seed(42)

    model = AccelerationNetwork(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    for step in range(1, num_steps + 1):

        # Random window from training data
        start = torch.randint(0, len(u_train) - window_size + 1, (1,)).item()

        u_window = u_train[start : start + window_size]
        y_window = y_train[start : start + window_size]

        # Initial conditions for this window
        y_start = y_window[0]
        v_start = (y_window[1] - y_window[0]) / dt

        # Free-run prediction
        y_pred, _ = rollout(model, y_start, v_start, u_window, dt)

        loss = torch.mean((y_pred - y_window) ** 2)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_history.append(loss.item())

        if step % 20 == 0:
            average_loss = np.mean(loss_history[-20:])
            print(f"Step {step}/{num_steps} | " f"Average MSE: {average_loss:.6f}")

    # Save trained model
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "window_size": window_size,
            "num_steps": num_steps,
            "loss_history": loss_history,
        },
        checkpoint_path,
    )

    print(f"{model_name} saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
