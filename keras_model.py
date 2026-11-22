"""Keras equivalent of the PyTorch bike-rental regression network.

Set ``KERAS_BACKEND`` before importing this module to choose a backend; it
defaults to TensorFlow. Predictions copied between models are compared with
``rtol=1e-5`` and ``atol=1e-5``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

# Keras reads this setting at import time. Callers can override it in the
# environment before starting Python, for example ``KERAS_BACKEND=jax``.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import numpy as np  # noqa: E402

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5


def build_keras_model(
    input_size: int, hidden_sizes: Sequence[int] = (64, 32)
) -> keras.Sequential:
    """Build the same fully connected regression architecture as PyTorch."""
    if input_size < 1:
        raise ValueError("input_size must be positive.")
    if len(hidden_sizes) < 2:
        raise ValueError("At least two hidden layers are required.")
    if any(width < 1 for width in hidden_sizes):
        raise ValueError("Every hidden layer width must be positive.")

    model = keras.Sequential(name="bike_rental_regressor")
    model.add(keras.layers.InputLayer(shape=(input_size,)))
    for index, width in enumerate(hidden_sizes, start=1):
        model.add(keras.layers.Dense(width, activation="relu", name=f"hidden_{index}"))
    model.add(keras.layers.Dense(1, activation=None, name="rental_count"))
    return model


def copy_pytorch_weights(torch_model, keras_model: keras.Sequential) -> None:
    """Copy Linear parameters into corresponding Keras Dense layers.

    PyTorch stores each weight matrix as ``(output, input)`` while Keras
    Dense kernels use ``(input, output)``, so each matrix is transposed.
    """
    from torch import nn

    torch_dense_layers = [
        layer for layer in torch_model.network if isinstance(layer, nn.Linear)
    ]
    keras_dense_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]
    if len(torch_dense_layers) != len(keras_dense_layers):
        raise ValueError(
            "PyTorch and Keras models must contain the same number of Dense layers."
        )

    for layer_index, (torch_layer, keras_layer) in enumerate(
        zip(torch_dense_layers, keras_dense_layers, strict=True), start=1
    ):
        if tuple(keras_layer.kernel.shape) != tuple(torch_layer.weight.T.shape):
            raise ValueError(
                f"Dense layer {layer_index} has incompatible weight shapes: "
                f"PyTorch {tuple(torch_layer.weight.shape)} and "
                f"Keras {tuple(keras_layer.kernel.shape)}."
            )
        if tuple(keras_layer.bias.shape) != tuple(torch_layer.bias.shape):
            raise ValueError(f"Dense layer {layer_index} has incompatible bias shapes.")

        kernel = torch_layer.weight.detach().cpu().numpy().T
        bias = torch_layer.bias.detach().cpu().numpy()
        keras_layer.set_weights([kernel, bias])


def max_prediction_difference(
    torch_model,
    keras_model: keras.Sequential,
    features: np.ndarray,
) -> float:
    """Return maximum absolute prediction error for shared float32 inputs."""
    import torch

    torch_model.eval()
    with torch.no_grad():
        torch_predictions = (
            torch_model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
        )
    keras_output = keras_model(features, training=False)
    keras_predictions = keras.ops.convert_to_numpy(keras_output)
    if torch_predictions.shape != keras_predictions.shape:
        raise ValueError(
            "Prediction shapes differ: "
            f"PyTorch {torch_predictions.shape}, Keras {keras_predictions.shape}."
        )
    np.testing.assert_allclose(
        torch_predictions,
        keras_predictions,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    return float(np.max(np.abs(torch_predictions - keras_predictions)))
