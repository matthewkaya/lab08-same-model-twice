# Lakeshore Bike Demand: The Same Model in PyTorch and Keras

## What it does

This project predicts hourly bike rentals from synthetic station-hour data. It retains the Lab 07 PyTorch implementation and adds an equivalent TensorFlow-backed Keras model, a six-run seeded comparison, and tests of prediction parity and Keras behavior.

## How to run it

From the repository root, create and activate the conda environment. It installs the pinned runtime and development dependencies and selects the TensorFlow Keras backend with CPU execution:

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
python -m pytest
python -m ruff format --check .
python -m ruff check .
python train.py
python train_keras.py
python compare.py
```

The environment uses Python 3.12, PyTorch `2.15.0+cpu`, Keras `3.15.1`, TensorFlow `2.20.0`, NumPy `2.5.4`, pandas `3.1.1`, scikit-learn `1.9.1`, matplotlib `3.11.2`, pytest `9.2.0`, and Ruff `0.17.5`. Pins are in `requirements.txt` and `requirements-dev.txt`.

## Data and split

`bikes_data.py` generates 6,000 synthetic station-hour rows with seed `2026`. The rows are split once into 3,600 training, 1,200 validation, and 1,200 test examples. Numeric features are standardized and `station_type` is one-hot encoded using transformations fitted only on training rows; `is_weekend` is passed through. The rental count is kept separate from the features. Both frameworks receive the same eight prepared features and target arrays; Keras does not define a second preprocessing pipeline. The same fitted transformations are used for validation and test data.

The exploratory plot shows distinct morning and evening peaks around 8:00 and 17:00–18:00, so rentals do not follow a straight-line relationship with hour. This helps explain why the linear regression baseline trails the neural network. The plot is generated at [`reports/rentals_by_hour.png`](reports/rentals_by_hour.png).

## Model and training

`BikeRentalRegressor` is an `nn.Module` with two ReLU hidden layers and a single unrestricted scalar output. The selected network has hidden widths `(128, 64)`. Training uses mean squared error, Adam (`learning_rate=0.001`), batches of 64, and up to 100 epochs. Python, NumPy, PyTorch, and the training DataLoader are seeded; validation and test loaders do not shuffle. Each epoch records training loss, validation loss, and validation MAE. Within a run, the best checkpoint is selected by minimum validation loss. Candidate PyTorch configurations are selected by validation MAE; the test split is used only after selection.

## Results

MAE is reported in rentals. The final neural network was selected on validation data, retrained with the same settings, and restored to its best validation-loss checkpoint before the final test evaluation.

| Model | Validation MAE | Test MAE |
|---|---:|---:|
| Training mean | 9.952 | 9.848 |
| LinearRegression | 9.032 | 8.763 |
| Neural network, hidden sizes `(128, 64)` | **2.787** | **2.685** |

The linear model performs worse because a single straight-line relationship cannot represent the plotted rush-hour peaks and other feature interactions as well as the selected network. The main run’s training and validation losses continue to fall through epoch 100, with a small gap; the selected wider network has lower validation loss and no observed validation-loss rise within the run.

![Main training and validation learning curves](reports/learning_curves.png)

The three-seed comparison uses the selected `(128, 64)` architecture for each framework. The table shows mean ± sample standard deviation (n=3); see [`COMPARISON.md`](COMPARISON.md) for observed differences and the recommendation.

| Framework | Validation MAE | Test MAE | Mean training time |
|---|---:|---:|---:|
| PyTorch | 2.6673 ± 0.1043 | 2.5825 ± 0.0900 | 7.57 s |
| Keras | 2.7284 ± 0.0653 | 2.5792 ± 0.0853 | 21.93 s |

## Experiments

Each additional run changes one hyperparameter relative to the main run. “Best epoch” is the minimum-validation-loss checkpoint; best validation MAE is the lowest value observed across epochs.

| Run | Hyperparameters | Best epoch | Best validation MAE | Final training loss | Final validation loss | Observation |
|---|---|---:|---:|---:|---:|---|
| Main | hidden `(64, 32)`, lr `0.001`, batch `64`, 100 epochs | 100 | 3.375 | 22.629 | 24.735 | Losses track closely and continue downward through the last epoch. |
| Low learning rate | hidden `(64, 32)`, lr `0.0001`, batch `64`, 100 epochs | 100 | 7.623 | 125.300 | 125.388 | Similar train and validation losses remain high, consistent with slower learning at this rate. |
| Wider hidden layers | hidden `(128, 64)`, lr `0.001`, batch `64`, 100 epochs | 100 | **2.787** | 13.047 | 16.218 | Best validation MAE; validation loss is still decreasing, with a modest train/validation gap. |

The wider network is selected because it has the lowest validation MAE. All three runs reach their best validation-loss checkpoint at epoch 100, so these curves show no validation-loss reversal within the tested training duration. Curves for every run are saved in `reports/`; per-epoch main-run history is in `reports/history.csv`, and run settings and metrics are in `reports/experiments.csv`.

## The Keras version

`keras_model.py` builds the same configurable Dense/ReLU/Dense architecture and copies PyTorch linear weights into Keras kernels with the required transpose. The prediction-equivalence check uses `rtol=1e-5` and `atol=1e-5`. `train_keras.py` seeds Python, NumPy, and Keras with `2026`; it uses Keras 3 with the TensorFlow backend, `compile()` and `fit()`, Adam, MSE, EarlyStopping with restored best weights, and a best-only ModelCheckpoint. It writes `reports/history_keras.csv` and `reports/learning_curves_keras.png`.

The binary `reports/best_keras_model.keras` checkpoint is ignored by `.gitignore`; running `train_keras.py` recreates it. `compare.py` uses the same prepared arrays for three CPU runs per framework with seeds `2026`, `2027`, and `2028`. It writes six rows to `reports/comparison.csv` and evaluates each selected model on the test split once.

## Tests

Run `python -m pytest` for all 23 cases: the 11 retained Lab 07 tests and 12 Keras test cases. The new tests cover output and parameter shapes, transferred-weight parity, shared train/validation arrays, tiny-batch overfitting, save/reload behavior, best-weight restoration, and same-seed repeatability. They do not score the held-out test set. Use `python -m ruff format --check .` and `python -m ruff check .` for formatting and lint checks.

## Limitations

These are invented station-hour observations, not records from real riders or stations. The data cannot capture actual station capacity, bike availability, holidays, events, seasonal changes, or local operating conditions. The results should not be used as real-world demand forecasts or for operational decisions without validating against representative observed data.

The comparison covers only three seeds on one small CPU workload. Initializers, batch shuffling, and framework kernels differ, so this result is not a universal framework ranking. See [`COMPARISON.md`](COMPARISON.md) before applying its recommendation to another workload.
