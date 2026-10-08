"""Check the prescribed preprocessing and analytical RK4 trajectories."""

from pathlib import Path

import numpy as np
import torch

from data_handler import center_signals, extract_excerpt, load_signals
from integrator import rk4_step, rollout
from model import AccelerationNetwork


def main():
    """Fail clearly if the data protocol, integration or training gradients break."""
    root = Path(__file__).resolve().parent.parent
    u, y = load_signals(root / "data/raw/SNLS80mV.mat")
    excerpt_u, excerpt_y = extract_excerpt(u, y, 42650, 46746)
    expected_u, expected_y, u_mean, y_mean = center_signals(excerpt_u, excerpt_y)
    with np.load(root / "data/processed/silverbox_excerpt.npz") as data:
        np.testing.assert_array_equal(data["u"], expected_u)
        np.testing.assert_array_equal(data["y"], expected_y)
        np.testing.assert_allclose(data["u_mean"], u_mean, rtol=0, atol=0)
        np.testing.assert_allclose(data["y_mean"], y_mean, rtol=0, atol=0)
        assert len(data["u"]) == len(data["y"]) == 4096
        assert int(data["train_size"]) == 3072
        assert float(data["fs"]) == 610.35
    print("PASS: exact excerpt, full-excerpt centering and 3072/1024 split")

    def constant_acceleration(x):
        return torch.full_like(x[:, :1], 2.0)

    zero = torch.tensor(0.0)
    position, velocity = rk4_step(constant_acceleration, zero, zero, zero, zero, 1.0)
    torch.testing.assert_close(position, torch.tensor(1.0))
    torch.testing.assert_close(velocity, torch.tensor(2.0))
    position, velocity = rollout(
        constant_acceleration, zero, zero, torch.zeros(3), dt=0.5
    )
    torch.testing.assert_close(position, torch.tensor([0.0, 0.25, 1.0]))
    torch.testing.assert_close(velocity, torch.tensor([0.0, 1.0, 2.0]))
    print("PASS: RK4 step and continuous rollout match constant-acceleration solution")

    # The RK4 training path must retain gradients through the network.
    torch.manual_seed(42)
    model = AccelerationNetwork(hidden_size=4)
    predicted, _ = rollout(model, zero, zero, torch.zeros(8), dt=1 / 610.35)
    predicted.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    print("PASS: finite nonzero training gradients through RK4")


if __name__ == "__main__":
    main()
