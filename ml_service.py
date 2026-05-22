import importlib.util
import math

import numpy as np
import pandas as pd

from feature_service import add_future_targets, build_feature_frame


TARGET_COLUMNS = ["target_return", "target_positive"]
NON_FEATURE_COLUMNS = ["close"] + TARGET_COLUMNS


def _normal_cdf(value):
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _feature_columns(dataset):
    return [
        col for col in dataset.columns
        if col not in NON_FEATURE_COLUMNS
    ]


def _prepare_matrix(dataset, columns):
    x = dataset[columns].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0)
    return x.to_numpy(dtype=float)


def _standardize_train_test(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


class NumpyRidgeRegressor:
    name = "Numpy Ridge Regression"

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.mean = None
        self.std = None

    def fit(self, x, y):
        x_scaled, _, mean, std = _standardize_train_test(x, x)
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        penalty = np.eye(x_design.shape[1]) * self.alpha
        penalty[0, 0] = 0

        self.coefficients = np.linalg.pinv(
            x_design.T @ x_design + penalty
        ) @ x_design.T @ y
        self.mean = mean
        self.std = std
        return self

    def predict(self, x):
        x_scaled = (x - self.mean) / self.std
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return x_design @ self.coefficients


class ExternalRegressor:
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return np.asarray(self.estimator.predict(x), dtype=float)


def _make_regressor():
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMRegressor

        return ExternalRegressor(
            LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbose=-1,
            ),
            "LightGBM Regressor"
        )

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBRegressor

        return ExternalRegressor(
            XGBRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            ),
            "XGBoost Regressor"
        )

    return NumpyRidgeRegressor(alpha=1.0)


def _walk_forward_validation(dataset, columns, min_train=252, test_window=21):
    if len(dataset) < min_train + test_window:
        return {
            "validation_mae": None,
            "directional_accuracy": None,
            "residual_std": float(dataset["target_return"].std()),
        }

    predictions = []
    actuals = []

    for start in range(min_train, len(dataset), test_window):
        end = min(start + test_window, len(dataset))
        train = dataset.iloc[:start]
        test = dataset.iloc[start:end]

        if test.empty:
            continue

        x_train = _prepare_matrix(train, columns)
        y_train = train["target_return"].to_numpy(dtype=float)
        x_test = _prepare_matrix(test, columns)

        model = _make_regressor()
        model.fit(x_train, y_train)
        fold_predictions = model.predict(x_test)

        predictions.extend(fold_predictions.tolist())
        actuals.extend(test["target_return"].tolist())

    if not predictions:
        return {
            "validation_mae": None,
            "directional_accuracy": None,
            "residual_std": float(dataset["target_return"].std()),
        }

    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    residuals = actuals - predictions

    return {
        "validation_mae": float(np.mean(np.abs(residuals))),
        "directional_accuracy": float(np.mean((predictions > 0) == (actuals > 0))),
        "residual_std": float(np.std(residuals, ddof=1)),
    }


def fit_predict_ml_return(ohlcv, macro_returns, horizon_days):
    features = build_feature_frame(ohlcv, macro_returns=macro_returns)

    if features.empty:
        return None

    dataset = add_future_targets(features, horizon_days)
    dataset = dataset.dropna(subset=["target_return"])
    columns = _feature_columns(dataset)
    dataset = dataset.dropna(how="all", subset=columns)

    if len(dataset) < 120 or not columns:
        return None

    validation = _walk_forward_validation(
        dataset,
        columns,
        min_train=min(252, max(80, len(dataset) // 2)),
        test_window=max(5, min(21, horizon_days))
    )

    x_train = _prepare_matrix(dataset, columns)
    y_train = dataset["target_return"].to_numpy(dtype=float)
    latest_features = features.dropna(how="all", subset=columns).tail(1)

    if latest_features.empty:
        return None

    x_latest = _prepare_matrix(latest_features, columns)
    model = _make_regressor()
    model.fit(x_train, y_train)
    expected_return = float(model.predict(x_latest)[0])

    residual_std = validation["residual_std"]

    if residual_std is None or pd.isna(residual_std) or residual_std <= 0:
        residual_std = float(dataset["target_return"].std())

    if residual_std is None or pd.isna(residual_std) or residual_std <= 0:
        probability_positive = 1.0 if expected_return > 0 else 0.0
    else:
        probability_positive = _normal_cdf(expected_return / residual_std)

    return {
        "ml_model": model.name,
        "ml_expected_return": expected_return,
        "ml_probability_positive": float(probability_positive),
        "ml_residual_volatility": float(residual_std) if residual_std else None,
        "ml_validation_mae": validation["validation_mae"],
        "ml_directional_accuracy": validation["directional_accuracy"],
        "ml_training_rows": int(len(dataset)),
        "ml_feature_count": int(len(columns)),
    }
