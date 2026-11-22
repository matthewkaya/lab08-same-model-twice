"""Data generation, splitting, train-only preprocessing, and PyTorch loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import make_rentals

SEED = 2026
TARGET = "rentals"
NUMERIC_FEATURES = ["hour", "temperature_c", "rain_mm", "wind_kph"]
CATEGORICAL_FEATURES = ["station_type"]
BINARY_FEATURES = ["is_weekend"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


@dataclass
class PreparedData:
    """Prepared arrays and source-row indices for the three data splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


class BikeRentalDataset(Dataset):
    """Return one feature row and one target as float32 tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must contain the same rows.")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def make_preprocessor() -> ColumnTransformer:
    """Create the feature transformer; fit it only after the row split."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Compatibility with scikit-learn before 1.2.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(
    frame: pd.DataFrame | None = None,
    seed: int = SEED,
) -> PreparedData:
    """Split rows first, then fit feature transformations on training rows."""
    if frame is None:
        frame = make_rentals(n=6000, seed=seed)

    row_indices = np.arange(len(frame))
    train_indices, holdout_indices = train_test_split(
        row_indices, test_size=0.4, random_state=seed, shuffle=True
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices, test_size=0.5, random_state=seed, shuffle=True
    )

    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame.loc[:, TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X.iloc[train_indices]).astype(np.float32)
    X_validation = preprocessor.transform(X.iloc[validation_indices]).astype(np.float32)
    X_test = preprocessor.transform(X.iloc[test_indices]).astype(np.float32)

    return PreparedData(
        X_train=X_train,
        y_train=y[train_indices],
        X_validation=X_validation,
        y_validation=y[validation_indices],
        X_test=X_test,
        y_test=y[test_indices],
        train_indices=np.asarray(train_indices),
        validation_indices=np.asarray(validation_indices),
        test_indices=np.asarray(test_indices),
        feature_names=list(preprocessor.get_feature_names_out()),
        preprocessor=preprocessor,
    )


def make_dataloaders(
    data: PreparedData, batch_size: int = 64, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible loaders, shuffling training rows only."""
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BikeRentalDataset(data.X_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        BikeRentalDataset(data.X_validation, data.y_validation),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        BikeRentalDataset(data.X_test, data.y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def save_hour_rentals_plot(
    frame: pd.DataFrame, path: str | Path = "reports/rentals_by_hour.png"
) -> Path:
    """Save the required exploratory plot of rentals against hour."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frame["hour"], frame[TARGET], alpha=0.18, s=12)
    hourly_mean = frame.groupby("hour", sort=True)[TARGET].mean()
    ax.plot(hourly_mean.index, hourly_mean.values, color="darkorange", linewidth=2)
    ax.set(title="Bike rentals by hour", xlabel="Hour of day", ylabel="Rentals")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    dataset = make_rentals(n=6000, seed=SEED)
    prepared = prepare_data(dataset, seed=SEED)
    train_loader, validation_loader, test_loader = make_dataloaders(prepared)
    save_hour_rentals_plot(dataset)
    first_features, first_targets = next(iter(train_loader))
    print(f"Generated rows: {len(dataset)}")
    print(
        "Split sizes: "
        f"{len(prepared.train_indices)}, "
        f"{len(prepared.validation_indices)}, "
        f"{len(prepared.test_indices)}"
    )
    print(f"Feature count: {first_features.shape[1]}")
    print(f"First training batch: {first_features.shape}, {first_targets.shape}")
    print(f"Tensor dtype: {first_features.dtype}, {first_targets.dtype}")
    print(f"Validation/test batches: {len(validation_loader)}, {len(test_loader)}")
