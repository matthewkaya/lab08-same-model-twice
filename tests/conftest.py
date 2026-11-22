"""Shared synthetic data fixtures for the pipeline tests."""

import pytest

from bikes_data import make_rentals
from prepare import SEED, prepare_data


@pytest.fixture(scope="session")
def generated_frame():
    return make_rentals(n=6000, seed=SEED)


@pytest.fixture(scope="session")
def prepared_data(generated_frame):
    return prepare_data(generated_frame, seed=SEED)
