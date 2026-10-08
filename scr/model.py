"""Neural acceleration model for the centered Silverbox state."""

import torch
import torch.nn as nn


class AccelerationNetwork(nn.Module):
    """Map scaled [output, output derivative, input] to acceleration."""

    def __init__(self, hidden_size=4):
        super().__init__()

        # Input scaling factors: [y, v, u]
        self.register_buffer("input_scale", torch.tensor([0.1, 40.0, 0.05]))

        # Output acceleration scaling factor
        self.register_buffer("acceleration_scale", torch.tensor(20000.0))

        # Neural network
        self.network = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):

        # Scale input features
        """Preserve input shape except for the final scalar acceleration channel."""
        x_scaled = x / self.input_scale

        # Predict dimensionless acceleration
        acceleration_scaled = self.network(x_scaled)

        # Convert back to physical acceleration
        acceleration = acceleration_scaled * self.acceleration_scale

        return acceleration


if __name__ == "__main__":

    # Create Model A
    model = AccelerationNetwork(hidden_size=4)

    # Example input: [y, v, u]
    x = torch.tensor([[0.1, 0.0, 0.02]], dtype=torch.float32)

    # Test forward pass
    acceleration = model(x)

    # Count trainable parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", acceleration.shape)
    print("Trainable parameters:", num_parameters)
    print("Predicted acceleration:", acceleration.item())
