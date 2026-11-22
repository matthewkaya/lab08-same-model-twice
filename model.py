"""Small feed-forward neural network for bike-rental regression."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class BikeRentalRegressor(nn.Module):
    """Predict one unrestricted rental count using configurable hidden layers."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if len(hidden_sizes) < 2:
            raise ValueError("At least two hidden layers are required.")
        if any(width < 1 for width in hidden_sizes):
            raise ValueError("Every hidden layer width must be positive.")

        layers: list[nn.Module] = []
        previous_size = input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous_size, width), nn.ReLU()))
            previous_size = width
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Return predictions shaped ``(batch_size, 1)``."""
        return self.network(features)
