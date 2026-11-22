"""Tests for Keras architecture, weight transfer, and training behavior."""

import keras
import numpy as np
import pytest
import torch

from keras_model import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    build_keras_model,
    copy_pytorch_weights,
    max_prediction_difference,
)
from model import BikeRentalRegressor
from prepare import BikeRentalDataset


@pytest.fixture(autouse=True)
def clear_keras_state():
    keras.backend.clear_session()
    yield
    keras.backend.clear_session()


@pytest.mark.parametrize("batch_size", [1, 7])
def test_keras_returns_one_scalar_per_sample(batch_size):
    model = build_keras_model(input_size=8, hidden_sizes=(16, 8))
    predictions = model(np.zeros((batch_size, 8), dtype=np.float32))

    assert tuple(predictions.shape) == (batch_size, 1)


@pytest.mark.parametrize(
    ("input_size", "hidden_sizes"),
    [
        (8, (64, 32)),
        (8, (128, 64)),
        (8, (16, 12, 8)),
        (3, (8, 4)),
    ],
)
def test_keras_and_pytorch_parameter_counts_match(input_size, hidden_sizes):
    torch_model = BikeRentalRegressor(input_size, hidden_sizes)
    keras_model = build_keras_model(input_size, hidden_sizes)

    torch_count = sum(parameter.numel() for parameter in torch_model.parameters())

    assert keras_model.count_params() == torch_count


def test_transferred_weights_produce_matching_predictions():
    torch.manual_seed(401)
    torch_model = BikeRentalRegressor(input_size=8, hidden_sizes=(128, 64))
    keras_model = build_keras_model(input_size=8, hidden_sizes=(128, 64))
    copy_pytorch_weights(torch_model, keras_model)
    features = np.random.default_rng(401).normal(size=(23, 8)).astype(np.float32)

    difference = max_prediction_difference(torch_model, keras_model, features)

    torch_predictions = torch_model(torch.from_numpy(features)).detach().numpy()
    keras_predictions = keras.ops.convert_to_numpy(keras_model(features))
    np.testing.assert_allclose(
        keras_predictions,
        torch_predictions,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    assert difference <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.max(
        np.abs(torch_predictions)
    )


def test_keras_uses_the_same_prepared_training_and_validation_arrays(prepared_data):
    train_dataset = BikeRentalDataset(prepared_data.X_train, prepared_data.y_train)
    validation_dataset = BikeRentalDataset(
        prepared_data.X_validation, prepared_data.y_validation
    )

    np.testing.assert_array_equal(train_dataset.features.numpy(), prepared_data.X_train)
    np.testing.assert_array_equal(train_dataset.targets.numpy(), prepared_data.y_train)
    np.testing.assert_array_equal(
        validation_dataset.features.numpy(), prepared_data.X_validation
    )
    np.testing.assert_array_equal(
        validation_dataset.targets.numpy(), prepared_data.y_validation
    )
    assert prepared_data.X_train.dtype == np.float32
    assert prepared_data.X_validation.dtype == np.float32

    keras_model = build_keras_model(
        input_size=prepared_data.X_train.shape[1], hidden_sizes=(8, 4)
    )
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
    )
    keras_model.fit(
        prepared_data.X_train,
        prepared_data.y_train,
        validation_data=(prepared_data.X_validation, prepared_data.y_validation),
        batch_size=512,
        epochs=1,
        shuffle=False,
        verbose=0,
    )


def test_keras_model_can_overfit_a_tiny_batch():
    keras.utils.set_random_seed(2026)
    features = np.linspace(-2, 2, 32, dtype=np.float32).reshape(-1, 1)
    targets = (2.5 * features - 0.75).astype(np.float32)
    model = build_keras_model(input_size=1, hidden_sizes=(16, 8))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.02),
        loss=keras.losses.MeanSquaredError(),
    )
    initial_loss = float(model.evaluate(features, targets, verbose=0))

    model.fit(
        features,
        targets,
        batch_size=len(features),
        epochs=150,
        shuffle=False,
        verbose=0,
    )
    final_loss = float(model.evaluate(features, targets, verbose=0))

    assert final_loss < initial_loss * 0.01


def test_saved_and_reloaded_keras_model_preserves_predictions(tmp_path):
    model = build_keras_model(input_size=8, hidden_sizes=(12, 6))
    features = np.random.default_rng(17).normal(size=(9, 8)).astype(np.float32)
    expected = keras.ops.convert_to_numpy(model(features))
    model_path = tmp_path / "round_trip.keras"

    model.save(model_path)
    reloaded = keras.models.load_model(model_path)
    actual = keras.ops.convert_to_numpy(reloaded(features))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_early_stopping_restores_best_weights():
    model = build_keras_model(input_size=1, hidden_sizes=(4, 2))
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    model.stop_training = False
    callback.set_model(model)
    callback.set_params({"epochs": 4, "steps": 1, "verbose": 0})
    callback.on_train_begin()
    expected_best_weights = None

    for epoch, (loss, marker) in enumerate(((1.0, 1.0), (0.5, 2.0), (0.8, 3.0))):
        model.set_weights(
            [np.full_like(weight, marker) for weight in model.get_weights()]
        )
        callback.on_epoch_end(epoch, {"val_loss": loss})
        if loss == 0.5:
            expected_best_weights = model.get_weights()
        if model.stop_training:
            break
    callback.on_train_end()

    assert expected_best_weights is not None
    for actual, expected in zip(
        model.get_weights(), expected_best_weights, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_short_keras_training_is_reproducible_with_same_seed():
    features = np.random.default_rng(31).normal(size=(24, 4)).astype(np.float32)
    targets = (features[:, :1] * 0.7 - features[:, 1:2] * 0.2).astype(np.float32)

    def train_once():
        keras.utils.set_random_seed(2027)
        model = build_keras_model(input_size=4, hidden_sizes=(8, 4))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        history = model.fit(
            features,
            targets,
            validation_data=(features, targets),
            batch_size=6,
            epochs=5,
            shuffle=True,
            verbose=0,
        )
        return model.get_weights(), history.history

    first_weights, first_history = train_once()
    second_weights, second_history = train_once()

    for first, second in zip(first_weights, second_weights, strict=True):
        np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-7)
    for metric in first_history:
        np.testing.assert_allclose(
            first_history[metric], second_history[metric], rtol=1e-7, atol=1e-7
        )
