"""Handwritten epoch and validation-based training functions."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from bikes_data import make_rentals
from model import BikeRentalRegressor
from prepare import SEED, make_dataloaders, prepare_data, save_hour_rentals_plot

plt.switch_backend("Agg")


@dataclass
class TrainingResult:
    """Per-epoch measurements and the best validation checkpoint details."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one reproducible neural-network run."""

    name: str
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    seed: int = SEED


def experiment_configurations() -> list[ExperimentConfig]:
    """Return the main run and the two one-hyperparameter comparisons."""
    return [
        ExperimentConfig(name="main"),
        ExperimentConfig(name="low_learning_rate", learning_rate=0.0001),
        ExperimentConfig(name="wider_hidden_layers", hidden_sizes=(128, 64)),
    ]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """Train for one epoch when given an optimizer, otherwise evaluate.

    Returns sample-weighted mean loss and mean absolute error in rental counts.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Model predictions and targets must have the same shape; "
                    f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
                )
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_absolute_error += (predictions.detach() - targets).abs().sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")

    mean_loss = total_loss / total_samples
    mean_absolute_error = total_absolute_error / total_samples
    return mean_loss, mean_absolute_error


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train for multiple epochs and restore the lowest-validation-loss weights."""
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    if criterion is None:
        criterion = nn.MSELoss()

    model.to(device)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        training_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_mae = run_epoch(
            model, validation_loader, criterion, optimizer=None, device=device
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint.")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )


def evaluate_baselines(data) -> dict[str, float]:
    """Fit both baselines on training rows and report validation MAE only."""
    training_mean = float(np.mean(data.y_train))
    mean_predictions = np.full_like(data.y_validation, training_mean)
    linear_model = LinearRegression()
    linear_model.fit(data.X_train, data.y_train.ravel())
    linear_predictions = linear_model.predict(data.X_validation).reshape(-1, 1)

    return {
        "training_mean_validation_mae": float(
            mean_absolute_error(data.y_validation, mean_predictions)
        ),
        "linear_regression_validation_mae": float(
            mean_absolute_error(data.y_validation, linear_predictions)
        ),
    }


def save_learning_curve(
    history: list[dict[str, float | int]],
    best_epoch: int,
    path: str | Path,
    title: str,
) -> None:
    """Save training and validation losses with the selected epoch marked."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    training_loss = [float(row["training_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, training_loss, label="Training loss")
    ax.plot(epochs, validation_loss, label="Validation loss")
    selected_loss = validation_loss[best_epoch - 1]
    ax.scatter([best_epoch], [selected_loss], color="crimson", zorder=3)
    ax.axvline(best_epoch, color="crimson", linestyle="--", alpha=0.6)
    ax.annotate(f"Best epoch: {best_epoch}", (best_epoch, selected_loss))
    ax.set(title=title, xlabel="Epoch", ylabel="Mean squared error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiments(
    output_directory: str | Path = "reports",
    seed: int = SEED,
    data=None,
    frame=None,
) -> dict[str, Any]:
    """Run baselines and three network configurations using validation only."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = make_rentals(n=6000, seed=seed)
    save_hour_rentals_plot(frame, output_path / "rentals_by_hour.png")
    if data is None:
        data = prepare_data(frame, seed=seed)
    baselines = evaluate_baselines(data)

    configurations = experiment_configurations()
    experiment_rows: list[dict[str, Any]] = []

    for config in configurations:
        seed_everything(config.seed)
        train_loader, validation_loader, _ = make_dataloaders(
            data, batch_size=config.batch_size, seed=config.seed
        )
        model = BikeRentalRegressor(
            input_size=data.X_train.shape[1], hidden_sizes=config.hidden_sizes
        )
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        result = train_model(
            model,
            train_loader,
            validation_loader,
            optimizer,
            epochs=config.epochs,
        )
        checkpoint_row = result.history[result.best_epoch - 1]
        best_mae_row = min(result.history, key=lambda row: row["validation_mae"])
        last_row = result.history[-1]
        experiment_rows.append(
            {
                "name": config.name,
                "hyperparameters": json.dumps(
                    {
                        "hidden_sizes": config.hidden_sizes,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "epochs": config.epochs,
                        "seed": config.seed,
                    },
                    sort_keys=True,
                ),
                "best_epoch": result.best_epoch,
                "best_validation_mae": float(best_mae_row["validation_mae"]),
                "best_validation_mae_epoch": int(best_mae_row["epoch"]),
                "checkpoint_validation_mae": float(checkpoint_row["validation_mae"]),
                "best_validation_loss": result.best_validation_loss,
                "final_training_loss": float(last_row["training_loss"]),
                "final_validation_loss": float(last_row["validation_loss"]),
            }
        )

        curve_name = (
            "learning_curves.png"
            if config.name == "main"
            else f"learning_curves_{config.name}.png"
        )
        save_learning_curve(
            result.history,
            result.best_epoch,
            output_path / curve_name,
            title=f"{config.name.replace('_', ' ').title()} run",
        )
        if config.name == "main":
            with (output_path / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as history_file:
                writer = csv.DictWriter(
                    history_file,
                    fieldnames=[
                        "epoch",
                        "training_loss",
                        "validation_loss",
                        "validation_mae",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(result.history)

    with (output_path / "experiments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as experiment_file:
        writer = csv.DictWriter(
            experiment_file,
            fieldnames=experiment_rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(experiment_rows)

    selected_experiment = min(
        experiment_rows, key=lambda row: row["best_validation_mae"]
    )
    report = {
        "baselines": baselines,
        "experiments": experiment_rows,
        "selected_neural_network": selected_experiment["name"],
        "selection_metric": "best_validation_mae",
    }
    with (output_path / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(report, metrics_file, indent=2)
        metrics_file.write("\n")

    return report


def finalize_metrics(
    data,
    experiment_report: dict[str, Any],
    output_directory: str | Path = "reports",
) -> dict[str, Any]:
    """Retrain the validation-selected network, then score all models on test."""
    output_path = Path(output_directory)
    selected_name = experiment_report["selected_neural_network"]
    selected_config = next(
        config for config in experiment_configurations() if config.name == selected_name
    )

    seed_everything(selected_config.seed)
    train_loader, validation_loader, test_loader = make_dataloaders(
        data,
        batch_size=selected_config.batch_size,
        seed=selected_config.seed,
    )
    model = BikeRentalRegressor(
        input_size=data.X_train.shape[1],
        hidden_sizes=selected_config.hidden_sizes,
    )
    optimizer = Adam(model.parameters(), lr=selected_config.learning_rate)
    training_result = train_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        epochs=selected_config.epochs,
    )

    # train_model restores its best validation-loss checkpoint before scoring.
    _, selected_validation_mae = run_epoch(model, validation_loader)
    _, selected_test_mae = run_epoch(model, test_loader)

    training_mean = float(np.mean(data.y_train))
    mean_validation_mae = float(
        mean_absolute_error(
            data.y_validation,
            np.full_like(data.y_validation, training_mean),
        )
    )
    mean_test_mae = float(
        mean_absolute_error(
            data.y_test,
            np.full_like(data.y_test, training_mean),
        )
    )

    linear_model = LinearRegression()
    linear_model.fit(data.X_train, data.y_train.ravel())
    linear_validation_predictions = linear_model.predict(data.X_validation)
    linear_test_predictions = linear_model.predict(data.X_test)
    linear_validation_mae = float(
        mean_absolute_error(data.y_validation.ravel(), linear_validation_predictions)
    )
    linear_test_mae = float(
        mean_absolute_error(data.y_test.ravel(), linear_test_predictions)
    )

    metrics = {
        "dataset": {
            "rows": len(data.train_indices)
            + len(data.validation_indices)
            + len(data.test_indices),
            "split_sizes": {
                "train": len(data.train_indices),
                "validation": len(data.validation_indices),
                "test": len(data.test_indices),
            },
        },
        "baselines": {
            "training_mean": {
                "validation_mae": mean_validation_mae,
                "test_mae": mean_test_mae,
            },
            "linear_regression": {
                "validation_mae": linear_validation_mae,
                "test_mae": linear_test_mae,
            },
        },
        "selected_neural_network": {
            "configuration": {
                "name": selected_config.name,
                "hidden_sizes": list(selected_config.hidden_sizes),
                "learning_rate": selected_config.learning_rate,
                "batch_size": selected_config.batch_size,
                "epochs": selected_config.epochs,
                "seed": selected_config.seed,
            },
            "best_epoch": training_result.best_epoch,
            "validation_mae": selected_validation_mae,
            "test_mae": selected_test_mae,
        },
        "selection_metric": "validation_mae",
        "reproducibility": {
            "seed": selected_config.seed,
            "python_numpy_torch_seeded": True,
            "dataloader_seed": selected_config.seed,
            "training_loader_shuffled_only": True,
            "torch_num_threads": torch.get_num_threads(),
            "device": "cpu",
        },
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics


def main() -> None:
    """Recreate experiment artifacts and the final held-out test report."""
    torch.set_num_threads(1)
    frame = make_rentals(n=6000, seed=SEED)
    data = prepare_data(frame, seed=SEED)
    experiment_report = run_experiments(data=data, frame=frame)
    metrics = finalize_metrics(data, experiment_report)

    print("MAE (rentals):")
    for model_name, scores in metrics["baselines"].items():
        print(
            f"  {model_name}: validation={scores['validation_mae']:.3f}, "
            f"test={scores['test_mae']:.3f}"
        )
    network_metrics = metrics["selected_neural_network"]
    print(
        f"  {network_metrics['configuration']['name']}: "
        f"validation={network_metrics['validation_mae']:.3f}, "
        f"test={network_metrics['test_mae']:.3f} "
        f"(best epoch {network_metrics['best_epoch']})"
    )


if __name__ == "__main__":
    main()
