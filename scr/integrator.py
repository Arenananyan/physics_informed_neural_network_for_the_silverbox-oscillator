"""Differentiable RK4 integration driven by the measured input voltage."""

import torch

from model import AccelerationNetwork


def dynamics(model, y, v, u):
    """
    Compute the derivatives:
        dy/dt = v
        dv/dt = alpha_theta(y, v, u)
    """

    # Neural network input: [y, v, u]
    x = torch.stack([y, v, u]).unsqueeze(0)

    # Predict acceleration
    acceleration = model(x).squeeze()

    return v, acceleration


def rk4_step(model, y, v, u_start, u_end, dt):
    """
    Perform one RK4 integration step.
    """

    # Input voltage at the middle of the time step
    u_mid = (u_start + u_end) / 2

    # First evaluation: beginning
    k1_y, k1_v = dynamics(model, y, v, u_start)

    # Second evaluation: midpoint
    k2_y, k2_v = dynamics(model, y + dt / 2 * k1_y, v + dt / 2 * k1_v, u_mid)

    # Third evaluation: midpoint
    k3_y, k3_v = dynamics(model, y + dt / 2 * k2_y, v + dt / 2 * k2_v, u_mid)

    # Fourth evaluation: end
    k4_y, k4_v = dynamics(model, y + dt * k3_y, v + dt * k3_v, u_end)

    # Combine the four evaluations
    y_next = y + dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

    v_next = v + dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return y_next, v_next


def rollout(model, y0, v0, u, dt):
    """
    Simulate the system over the entire input sequence.

    No measured output is used after initialization. Autograd remains enabled
    so the same rollout can also be used to train on short windows.
    """

    # Initial state
    y = y0
    v = v0

    # Store predicted states
    y_predictions = [y]
    v_predictions = [v]

    # Perform consecutive RK4 steps
    for i in range(len(u) - 1):

        y, v = rk4_step(model, y, v, u[i], u[i + 1], dt)

        y_predictions.append(y)
        v_predictions.append(v)

    return (torch.stack(y_predictions), torch.stack(v_predictions))


if __name__ == "__main__":

    # Create Model A
    model = AccelerationNetwork(hidden_size=4)

    # Example initial state (only for testing)
    y = torch.tensor(0.1)
    v = torch.tensor(0.0)

    # Example input voltage at two consecutive samples
    u_start = torch.tensor(0.02)
    u_end = torch.tensor(0.025)

    # Sampling time
    dt = 1.0 / 610.35

    # Perform one RK4 step
    y_next, v_next = rk4_step(model, y, v, u_start, u_end, dt)

    print(f"Initial y: {y.item():.10f}")
    print(f"Initial v: {v.item():.10f}")

    print(f"Next y: {y_next.item():.10f}")
    print(f"Next v: {v_next.item():.10f}")

    print("\n--- RK4 Mathematical Test ---")

    # A simple model with constant acceleration = 2
    def constant_acceleration_model(x):
        return torch.full_like(x[:, :1], 2.0)

    # Known initial conditions
    y0 = torch.tensor(0.0)
    v0 = torch.tensor(0.0)

    u0 = torch.tensor(0.0)
    u1 = torch.tensor(0.0)

    # One RK4 step
    y1, v1 = rk4_step(constant_acceleration_model, y0, v0, u0, u1, dt=1.0)

    print("Expected y:", 1.0)
    print("Computed y:", y1.item())

    print("Expected v:", 2.0)
    print("Computed v:", v1.item())

    # Verify numerical correctness
    assert torch.allclose(y1, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(v1, torch.tensor(2.0), atol=1e-6)

    print("RK4 mathematical test PASSED")

    print("\n--- Rollout Test ---")

    u_sequence = torch.tensor([0.0, 0.0, 0.0])

    y_pred, v_pred = rollout(
        constant_acceleration_model,
        torch.tensor(0.0),
        torch.tensor(0.0),
        u_sequence,
        dt=0.5,
    )

    print("Predicted y:", y_pred)
    print("Predicted v:", v_pred)

    assert torch.allclose(y_pred[-1], torch.tensor(1.0), atol=1e-6)

    assert torch.allclose(v_pred[-1], torch.tensor(2.0), atol=1e-6)

    print("Rollout test PASSED")
