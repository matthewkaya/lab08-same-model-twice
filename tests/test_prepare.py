"""Tests for splitting, train-fitted preparation, and data loading."""

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from prepare import (
    NUMERIC_FEATURES,
    BikeRentalDataset,
    make_dataloaders,
    prepare_data,
)


def test_dataset_returns_single_row_float32_tensors(prepared_data):
    dataset = BikeRentalDataset(prepared_data.X_train, prepared_data.y_train)
    features, target = dataset[0]

    assert len(dataset) == 3600
    assert features.shape == (prepared_data.X_train.shape[1],)
    assert target.shape == (1,)
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_splits_are_reproducible_correct_sized_and_disjoint(generated_frame):
    first = prepare_data(generated_frame, seed=2026)
    second = prepare_data(generated_frame, seed=2026)

    assert (
        len(first.train_indices),
        len(first.validation_indices),
        len(first.test_indices),
    ) == (
        3600,
        1200,
        1200,
    )
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert len(
        np.unique(
            np.concatenate(
                (first.train_indices, first.validation_indices, first.test_indices)
            )
        )
    ) == len(generated_frame)
    assert not np.intersect1d(first.train_indices, first.validation_indices).size
    assert not np.intersect1d(first.train_indices, first.test_indices).size
    assert not np.intersect1d(first.validation_indices, first.test_indices).size


def test_preprocessor_is_fitted_on_training_rows_only(prepared_data, generated_frame):
    fitted_scaler = prepared_data.preprocessor.named_transformers_["numeric"]
    raw_training = generated_frame.iloc[prepared_data.train_indices][NUMERIC_FEATURES]
    all_rows_mean = generated_frame[NUMERIC_FEATURES].mean().to_numpy()

    np.testing.assert_allclose(fitted_scaler.mean_, raw_training.mean().to_numpy())
    assert not np.allclose(fitted_scaler.mean_, all_rows_mean)
    np.testing.assert_allclose(
        prepared_data.X_train[:, :4].mean(axis=0), 0.0, atol=1e-6
    )
    assert not np.allclose(
        prepared_data.X_validation[:, :4].mean(axis=0), 0.0, atol=0.01
    )


def test_training_loader_shuffles_repeatably_and_other_loaders_do_not(prepared_data):
    first_train, first_validation, first_test = make_dataloaders(
        prepared_data, batch_size=128, seed=2026
    )
    second_train, _, _ = make_dataloaders(prepared_data, batch_size=128, seed=2026)
    first_order = torch.cat([features for features, _ in first_train])
    second_order = torch.cat([features for features, _ in second_train])

    assert isinstance(first_train.sampler, RandomSampler)
    assert isinstance(first_validation.sampler, SequentialSampler)
    assert isinstance(first_test.sampler, SequentialSampler)
    assert torch.equal(first_order, second_order)
    assert not torch.equal(first_order, torch.from_numpy(prepared_data.X_train))
    torch.testing.assert_close(
        torch.cat([features for features, _ in first_validation]),
        torch.from_numpy(prepared_data.X_validation),
    )
