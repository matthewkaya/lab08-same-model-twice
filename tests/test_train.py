"""Tests for training, evaluation, and checkpoint restoration."""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import train as train_module
from model import BikeRentalRegressor
from train import run_epoch, train_model


def test_training_can_overfit_a_tiny_batch():
    torch.manual_seed(2026)
    features = torch.linspace(-2, 2, 32).reshape(-1, 1)
    targets = (2.5 * features - 0.75).to(torch.float32)
    loader = DataLoader(TensorDataset(features, targets), batch_size=len(features))
    model = BikeRentalRegressor(input_size=1, hidden_sizes=(32, 16))
    optimizer = Adam(model.parameters(), lr=0.02)
    initial_loss, _ = run_epoch(model, loader)

    for _ in range(300):
        run_epoch(model, loader, optimizer=optimizer)

    final_loss, _ = run_epoch(model, loader)
    assert final_loss < initial_loss * 0.01


def test_evaluation_does_not_change_model_parameters():
    torch.manual_seed(2026)
    model = BikeRentalRegressor(input_size=3, hidden_sizes=(8, 4))
    loader = DataLoader(
        TensorDataset(torch.randn(12, 3), torch.randn(12, 1)), batch_size=4
    )
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    run_epoch(model, loader, optimizer=None)

    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_training_records_history_and_restores_best_validation_weights(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    validation_losses = iter([3.0, 1.0, 2.0])
    training_epoch = 0

    def controlled_epoch(model, loader, criterion=None, optimizer=None, device="cpu"):
        nonlocal training_epoch
        if optimizer is not None:
            training_epoch += 1
            with torch.no_grad():
                model.weight.fill_(training_epoch)
            return 0.5, 0.25
        loss = next(validation_losses)
        return loss, loss / 2

    monkeypatch.setattr(train_module, "run_epoch", controlled_epoch)
    result = train_model(model, [], [], Adam(model.parameters()), epochs=3)

    assert result.best_epoch == 2
    assert result.best_validation_loss == 1.0
    assert len(result.history) == 3
    assert set(result.history[0]) == {
        "epoch",
        "training_loss",
        "validation_loss",
        "validation_mae",
    }
    assert torch.equal(model.weight, torch.tensor([[2.0]]))
