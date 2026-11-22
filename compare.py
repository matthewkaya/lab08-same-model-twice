"""Compare fixed-seed PyTorch and TensorFlow-backed Keras training runs."""

from __future__ import annotations

import csv
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import keras  # noqa: E402

import train as train_module  # noqa: E402
import train_keras as train_keras_module  # noqa: E402
from bikes_data import make_rentals  # noqa: E402
from keras_model import build_keras_model  # noqa: E402
from model import BikeRentalRegressor  # noqa: E402
from prepare import (  # noqa: E402
    SEED,
    BikeRentalDataset,
    PreparedData,
    prepare_data,
)

SEEDS = (2026, 2027, 2028)
TRAINING_CONFIG = train_keras_module.KerasTrainingConfig(
    hidden_sizes=(128, 64),
    learning_rate=0.001,
    batch_size=64,
    epochs=100,
    seed=SEED,
    early_stopping_patience=100,
)
OUTPUT_PATH = Path("reports/comparison.csv")


@dataclass
class FitResult:
    """A best-validation checkpoint and fit-only runtime for one run."""

    model: Any
    best_epoch: int
    validation_loss: float
    validation_mae: float
    seconds: float


def configure_deterministic_cpu() -> None:
    """Use deterministic CPU kernels for both frameworks."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tf.config.experimental.enable_op_determinism()
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError(
            "Comparison requires Keras 3 with the TensorFlow backend; "
            "set KERAS_BACKEND=tensorflow before importing Keras."
        )
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("A GPU is visible; comparison must run on CPU only.")


def validate_shared_data(data: PreparedData) -> None:
    """Check the shared arrays, feature dimension, and disjoint data splits."""
    if data.X_train.shape[1] != 8:
        raise ValueError(f"Expected 8 input features, got {data.X_train.shape[1]}.")
    for features, targets, split in (
        (data.X_train, data.y_train, "train"),
        (data.X_validation, data.y_validation, "validation"),
        (data.X_test, data.y_test, "test"),
    ):
        if features.dtype != np.float32 or targets.dtype != np.float32:
            raise TypeError(f"{split} arrays must be float32.")
        if features.ndim != 2 or targets.shape != (len(features), 1):
            raise ValueError(
                f"Unexpected {split} shapes: {features.shape}, {targets.shape}."
            )

    split_indices = (
        data.train_indices,
        data.validation_indices,
        data.test_indices,
    )
    for first_index, first in enumerate(split_indices):
        for second in split_indices[first_index + 1 :]:
            if np.intersect1d(first, second).size:
                raise ValueError("Train, validation, and test splits must be disjoint.")

    # Verify that PyTorch's Dataset wrapper preserves every shared array value.
    for features, targets in (
        (data.X_train, data.y_train),
        (data.X_validation, data.y_validation),
        (data.X_test, data.y_test),
    ):
        dataset = BikeRentalDataset(features, targets)
        np.testing.assert_array_equal(dataset.features.numpy(), features)
        np.testing.assert_array_equal(dataset.targets.numpy(), targets)


def verify_seeded_initialization(input_size: int) -> None:
    """Check repeatable same-framework initialization and seed separation."""

    def torch_weights(seed: int) -> list[np.ndarray]:
        train_module.seed_everything(seed)
        model = BikeRentalRegressor(input_size, TRAINING_CONFIG.hidden_sizes)
        return [
            parameter.detach().cpu().numpy().copy() for parameter in model.parameters()
        ]

    def keras_weights(seed: int) -> list[np.ndarray]:
        train_keras_module.seed_everything(seed)
        model = build_keras_model(input_size, TRAINING_CONFIG.hidden_sizes)
        return [weight.copy() for weight in model.get_weights()]

    for get_weights in (torch_weights, keras_weights):
        first = get_weights(SEEDS[0])
        repeated = get_weights(SEEDS[0])
        different_seed = get_weights(SEEDS[1])
        if not all(
            np.array_equal(left, right)
            for left, right in zip(first, repeated, strict=True)
        ):
            raise AssertionError("Same-seed initialization was not reproducible.")
        if all(
            np.array_equal(left, right)
            for left, right in zip(first, different_seed, strict=True)
        ):
            raise AssertionError("Different seeds produced identical initial weights.")


def fit_pytorch(data: PreparedData, seed: int) -> FitResult:
    """Train one PyTorch model, retaining train_model's best validation weights."""
    train_module.seed_everything(seed)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BikeRentalDataset(data.X_train, data.y_train),
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        BikeRentalDataset(data.X_validation, data.y_validation),
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
    )
    model = BikeRentalRegressor(
        input_size=data.X_train.shape[1],
        hidden_sizes=TRAINING_CONFIG.hidden_sizes,
    ).cpu()
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG.learning_rate)

    started = time.perf_counter()
    training_result = train_module.train_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        epochs=TRAINING_CONFIG.epochs,
        criterion=torch.nn.MSELoss(),
        device="cpu",
    )
    seconds = time.perf_counter() - started

    validation_loss, validation_mae = train_module.run_epoch(model, validation_loader)
    expected_best = training_result.history[training_result.best_epoch - 1]
    np.testing.assert_allclose(
        validation_loss,
        expected_best["validation_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        validation_mae,
        expected_best["validation_mae"],
        rtol=1e-6,
        atol=1e-6,
    )
    return FitResult(
        model=model,
        best_epoch=training_result.best_epoch,
        validation_loss=validation_loss,
        validation_mae=validation_mae,
        seconds=seconds,
    )


def fit_keras(data: PreparedData, seed: int, checkpoint_path: str | Path) -> FitResult:
    """Train one Keras model and reload its per-run best validation checkpoint."""
    train_keras_module.seed_everything(seed)
    run_config = replace(TRAINING_CONFIG, seed=seed)
    model = train_keras_module.compile_model(data.X_train.shape[1], run_config)
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=Path(checkpoint_path),
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
            patience=TRAINING_CONFIG.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_validation, data.y_validation),
        batch_size=TRAINING_CONFIG.batch_size,
        epochs=TRAINING_CONFIG.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    ).history
    seconds = time.perf_counter() - started

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No best checkpoint was saved for seed {seed}.")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_model = keras.models.load_model(checkpoint)
    validation_metrics = best_model.evaluate(
        data.X_validation,
        data.y_validation,
        batch_size=TRAINING_CONFIG.batch_size,
        verbose=0,
        return_dict=True,
    )
    np.testing.assert_allclose(
        validation_metrics["loss"],
        history["val_loss"][best_epoch - 1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        validation_metrics["mae"],
        history["val_mae"][best_epoch - 1],
        rtol=1e-6,
        atol=1e-6,
    )
    return FitResult(
        model=best_model,
        best_epoch=best_epoch,
        validation_loss=float(validation_metrics["loss"]),
        validation_mae=float(validation_metrics["mae"]),
        seconds=seconds,
    )


def evaluate_pytorch_test_once(model: BikeRentalRegressor, data: PreparedData) -> float:
    """Evaluate one selected PyTorch model on the test rows exactly once."""
    test_loader = DataLoader(
        BikeRentalDataset(data.X_test, data.y_test),
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
    )
    _, test_mae = train_module.run_epoch(
        model, test_loader, criterion=torch.nn.MSELoss()
    )
    return test_mae


def evaluate_keras_test_once(model: keras.Sequential, data: PreparedData) -> float:
    """Evaluate one selected Keras model on the test rows exactly once."""
    metrics = model.evaluate(
        data.X_test,
        data.y_test,
        batch_size=TRAINING_CONFIG.batch_size,
        verbose=0,
        return_dict=True,
    )
    return float(metrics["mae"])


def write_comparison_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write exactly one row per framework and seed with fixed precision."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=[
                "framework",
                "seed",
                "best_epoch",
                "validation_mae",
                "test_mae",
                "seconds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": row["framework"],
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "validation_mae": f"{row['validation_mae']:.6f}",
                    "test_mae": f"{row['test_mae']:.6f}",
                    "seconds": f"{row['seconds']:.6f}",
                }
            )
    return output_path


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print means, sample standard deviations, and relative fit time."""
    summaries: dict[str, dict[str, float]] = {}
    for framework in ("PyTorch", "Keras"):
        subset = [row for row in rows if row["framework"] == framework]
        summaries[framework] = {
            "validation_mean": mean(row["validation_mae"] for row in subset),
            "validation_sd": stdev(row["validation_mae"] for row in subset),
            "test_mean": mean(row["test_mae"] for row in subset),
            "test_sd": stdev(row["test_mae"] for row in subset),
            "seconds_mean": mean(row["seconds"] for row in subset),
        }

    print("\nThree-seed comparison (sample SD, n=3):")
    for framework, summary in summaries.items():
        print(
            f"{framework}: validation MAE="
            f"{summary['validation_mean']:.4f} ± {summary['validation_sd']:.4f}; "
            f"test MAE={summary['test_mean']:.4f} ± {summary['test_sd']:.4f}; "
            f"mean training time={summary['seconds_mean']:.2f}s"
        )
    pytorch_seconds = summaries["PyTorch"]["seconds_mean"]
    keras_seconds = summaries["Keras"]["seconds_mean"]
    relative_time = (keras_seconds - pytorch_seconds) / pytorch_seconds * 100.0
    print(
        f"Relative training-time difference (Keras vs PyTorch): {relative_time:+.2f}%"
    )
    print(
        "These six runs compare only the specified seeds; they do not establish "
        "a universal winner."
    )


def main() -> None:
    """Prepare once, run six independent fits, and score each test split once."""
    total_started = time.perf_counter()
    configure_deterministic_cpu()
    frame = make_rentals(n=6000, seed=SEED)
    data = prepare_data(frame, seed=SEED)
    validate_shared_data(data)
    verify_seeded_initialization(data.X_train.shape[1])

    print(
        "Prepared shared arrays: "
        f"train={data.X_train.shape}, validation={data.X_validation.shape}, "
        f"test={data.X_test.shape}; features={data.feature_names}"
    )
    rows: list[dict[str, Any]] = []

    # The same PreparedData object and arrays are passed to all six runs.
    for seed in SEEDS:
        fit = fit_pytorch(data, seed)
        test_mae = evaluate_pytorch_test_once(fit.model, data)
        rows.append(
            {
                "framework": "PyTorch",
                "seed": seed,
                "best_epoch": fit.best_epoch,
                "validation_mae": fit.validation_mae,
                "test_mae": test_mae,
                "seconds": fit.seconds,
            }
        )
        print(
            f"PyTorch seed={seed}: epoch={fit.best_epoch}, "
            f"validation MAE={fit.validation_mae:.4f}, test MAE={test_mae:.4f}, "
            f"training={fit.seconds:.2f}s"
        )

    with tempfile.TemporaryDirectory(prefix="lab08-comparison-") as checkpoint_dir:
        for seed in SEEDS:
            checkpoint_path = Path(checkpoint_dir) / f"keras_seed_{seed}.keras"
            fit = fit_keras(data, seed, checkpoint_path)
            test_mae = evaluate_keras_test_once(fit.model, data)
            rows.append(
                {
                    "framework": "Keras",
                    "seed": seed,
                    "best_epoch": fit.best_epoch,
                    "validation_mae": fit.validation_mae,
                    "test_mae": test_mae,
                    "seconds": fit.seconds,
                }
            )
            print(
                f"Keras seed={seed}: epoch={fit.best_epoch}, "
                f"validation MAE={fit.validation_mae:.4f}, test MAE={test_mae:.4f}, "
                f"training={fit.seconds:.2f}s"
            )

    write_comparison_csv(rows, OUTPUT_PATH)
    print_summary(rows)
    print(f"Report: {OUTPUT_PATH}")
    print(
        "Total elapsed time (including data preparation): "
        f"{time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
