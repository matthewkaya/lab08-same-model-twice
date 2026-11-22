"""Train the Keras model using the shared Lab 07 prepared data arrays."""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from bikes_data import make_rentals  # noqa: E402
from keras_model import build_keras_model  # noqa: E402
from prepare import SEED, PreparedData, prepare_data  # noqa: E402

plt.switch_backend("Agg")


@dataclass(frozen=True)
class KerasTrainingConfig:
    """The validation-selected PyTorch experiment configuration."""

    hidden_sizes: tuple[int, ...] = (128, 64)
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    seed: int = SEED
    early_stopping_patience: int = 100


@dataclass
class KerasTrainingResult:
    """Selected validation checkpoint and its held-out test evaluation."""

    model: keras.Sequential
    history: dict[str, list[float]]
    best_epoch: int
    best_validation_loss: float
    best_validation_mae: float
    test_loss: float
    test_mae: float
    training_duration_seconds: float
    checkpoint_path: Path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and the TensorFlow backend."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def save_history_csv(history: dict[str, list[float]], path: str | Path) -> Path:
    """Write per-epoch training loss, validation loss, and validation MAE."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as history_file:
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
        for index, (training_loss, validation_loss, validation_mae) in enumerate(
            zip(history["loss"], history["val_loss"], history["val_mae"], strict=True),
            start=1,
        ):
            writer.writerow(
                {
                    "epoch": index,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_mae": validation_mae,
                }
            )
    return output_path


def save_learning_curves(
    history: dict[str, list[float]], best_epoch: int, path: str | Path
) -> Path:
    """Plot losses and validation MAE with separate, labelled y axes."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["loss"]) + 1)

    figure, loss_axis = plt.subplots(figsize=(9, 5.5))
    mae_axis = loss_axis.twinx()
    training_line = loss_axis.plot(
        epochs, history["loss"], label="Training loss (MSE)", color="tab:blue"
    )
    validation_line = loss_axis.plot(
        epochs,
        history["val_loss"],
        label="Validation loss (MSE)",
        color="tab:orange",
    )
    mae_line = mae_axis.plot(
        epochs,
        history["val_mae"],
        label="Validation MAE",
        color="tab:green",
    )
    loss_axis.axvline(best_epoch, color="crimson", linestyle="--", alpha=0.65)
    loss_axis.set(
        title="Keras bike-rental model training",
        xlabel="Epoch",
        ylabel="Mean squared error (rentals²)",
    )
    mae_axis.set_ylabel("Mean absolute error (rentals)")
    lines = training_line + validation_line + mae_line
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    loss_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def compile_model(input_size: int, config: KerasTrainingConfig) -> keras.Sequential:
    """Build and compile with the PyTorch run's Adam/MSE configuration."""
    model = build_keras_model(input_size, config.hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=False,
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_keras(
    data: PreparedData | None = None,
    config: KerasTrainingConfig = KerasTrainingConfig(),
    output_directory: str | Path = "reports",
) -> KerasTrainingResult:
    """Fit on shared train/validation arrays, then test once with best weights."""
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError(
            "Keras must use its TensorFlow backend for training; set "
            "KERAS_BACKEND=tensorflow before importing Keras."
        )

    if data is None:
        frame = make_rentals(n=6000, seed=config.seed)
        data = prepare_data(frame, seed=config.seed)

    seed_everything(config.seed)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_keras_model.keras"

    model = compile_model(data.X_train.shape[1], config)
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.0,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    started = time.perf_counter()
    fit_history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_validation, data.y_validation),
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    training_duration = time.perf_counter() - started

    history = {
        key: [float(value) for value in values]
        for key, values in fit_history.history.items()
    }
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_validation_loss = history["val_loss"][best_epoch - 1]
    best_validation_mae = history["val_mae"][best_epoch - 1]

    save_history_csv(history, output_path / "history_keras.csv")
    save_learning_curves(history, best_epoch, output_path / "learning_curves_keras.png")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best-model checkpoint was not created: {checkpoint_path}"
        )
    # Reload the exact model written by ModelCheckpoint before touching test data.
    best_model = keras.models.load_model(checkpoint_path)
    test_metrics: dict[str, Any] = best_model.evaluate(
        data.X_test,
        data.y_test,
        batch_size=config.batch_size,
        verbose=0,
        return_dict=True,
    )

    return KerasTrainingResult(
        model=best_model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_mae=best_validation_mae,
        test_loss=float(test_metrics["loss"]),
        test_mae=float(test_metrics["mae"]),
        training_duration_seconds=training_duration,
        checkpoint_path=checkpoint_path,
    )


def main() -> None:
    """Run the selected Keras experiment and print its final metrics."""
    config = KerasTrainingConfig()
    frame = make_rentals(n=6000, seed=config.seed)
    data = prepare_data(frame, seed=config.seed)
    result = train_keras(data=data, config=config)

    print(f"Backend: {keras.backend.backend()}")
    print(f"Configuration: {asdict(config)}")
    print(f"Training duration: {result.training_duration_seconds:.2f} seconds")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_validation_loss:.6f}")
    print(f"Best validation MAE: {result.best_validation_mae:.6f}")
    print(f"Final test MAE: {result.test_mae:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
