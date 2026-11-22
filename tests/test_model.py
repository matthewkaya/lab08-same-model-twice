"""Tests for the regression network and optimizer updates."""

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from model import BikeRentalRegressor
from train import run_epoch


@pytest.mark.parametrize("hidden_sizes", [(8, 4), (32, 16), (16, 12, 8)])
def test_network_returns_one_scalar_per_row(hidden_sizes):
    model = BikeRentalRegressor(input_size=5, hidden_sizes=hidden_sizes)
    predictions = model(torch.randn(7, 5))

    assert predictions.shape == (7, 1)


def test_optimizer_step_updates_parameters_in_every_trainable_layer():
    torch.manual_seed(2026)
    model = BikeRentalRegressor(input_size=4, hidden_sizes=(16, 8))
    features = torch.randn(32, 4)
    targets = torch.randn(32, 1)
    loader = DataLoader(TensorDataset(features, targets), batch_size=32)
    layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    before = [
        {name: value.detach().clone() for name, value in layer.named_parameters()}
        for layer in layers
    ]

    run_epoch(model, loader, optimizer=Adam(model.parameters(), lr=0.01))

    for layer, original in zip(layers, before, strict=True):
        assert any(
            not torch.equal(original[name], value)
            for name, value in layer.named_parameters()
        )
