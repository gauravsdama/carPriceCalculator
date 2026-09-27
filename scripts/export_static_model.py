#!/usr/bin/env python3
"""Export the compact Ridge model used by the dependency-free static demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from car_price_data import DATA_PATH, MODEL_FEATURES, grouped_evaluation_split, load_data

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "demo" / "model.json"
DATASET_URL = "https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset"
FLOAT_PRECISION = 8
ARTIFACT_TYPE = "car-price-calculator.static-model"
SCHEMA_VERSION = 1


def _rounded(values) -> list[float]:
    return [round(float(value), FLOAT_PRECISION) for value in values]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            ["Year", "Mileage", "Rating"],
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["Model"],
                        ),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )


def export_payload() -> dict:
    data = load_data()
    features = data.loc[:, MODEL_FEATURES]
    target = data["Price"]
    x_train, x_test, y_train, y_test = grouped_evaluation_split(features, target)

    evaluation_model = build_pipeline().fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)

    final_model = build_pipeline().fit(features, target)
    preprocessor = final_model.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["num"]
    scaler = numeric.named_steps["scaler"]
    encoder = preprocessor.named_transformers_["cat"]
    regressor = final_model.named_steps["regressor"]
    models = [str(value) for value in encoder.categories_[0]]
    coefficients = [float(value) for value in regressor.coef_]

    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "title": "USA Mercedes Benz Prices Dataset",
            "creator": "Danish Ammar",
            "source_url": DATASET_URL,
            "source_version": 1,
            "source_last_updated": "2024-04-26",
            "license": "Apache-2.0",
            "local_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "rows": len(data),
            "note": "The checked-in CSV is a transformed copy with Name and Mileage split into fields.",
        },
        "model": {
            "method": "Ridge regression",
            "alpha": 10.0,
            "trained_rows": len(data),
            "features": ["Year", "Mileage", "Rating", "Model"],
            "numeric_features": ["Year", "Mileage", "Rating"],
            "intercept": round(float(regressor.intercept_), FLOAT_PRECISION),
            "numeric_mean": _rounded(scaler.mean_),
            "numeric_scale": _rounded(scaler.scale_),
            "numeric_coefficients": _rounded(coefficients[:3]),
            "models": models,
            "model_coefficients": _rounded(coefficients[3:]),
        },
        "evaluation": {
            "method": (
                "Single deterministic grouped 80/20 holdout; identical feature vectors stay "
                "together (random_state=42)"
            ),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "r2": round(float(r2_score(y_test, predictions)), 6),
            "rmse_usd": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
            "mae_usd": round(float(mean_absolute_error(y_test, predictions)), 2),
        },
        "bounds": {
            "year": [int(data["Year"].min()), int(data["Year"].max())],
            "mileage": [int(data["Mileage"].min()), int(data["Mileage"].max())],
            "rating": [0, 5],
            "price": [float(data["Price"].min()), float(data["Price"].max())],
        },
        "defaults": {
            "model": "GLC 300" if "GLC 300" in models else models[0],
            "year": min(2022, int(data["Year"].max())),
            "mileage": 36000,
            "rating": 4.6,
        },
        "limitations": [
            "Saved listing snapshot; not live market data.",
            "Asking prices are not verified sale prices.",
            "Dealer rating is not a vehicle-condition score.",
            "The static Ridge model differs from the Flask Random Forest.",
            "A single grouped holdout does not establish performance over time or on unseen models.",
        ],
    }


def validate_artifact_contract(payload: object) -> dict:
    """Validate semantic invariants that are shared with the browser loader."""

    if not isinstance(payload, dict):
        raise TypeError("artifact must be a JSON object")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"artifact_type must be {ARTIFACT_TYPE}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    try:
        model = payload["model"]
        models = model["models"]
        model_coefficients = model["model_coefficients"]
        numeric_features = model["numeric_features"]
        numeric_mean = model["numeric_mean"]
        numeric_scale = model["numeric_scale"]
        numeric_coefficients = model["numeric_coefficients"]
        bounds = payload["bounds"]
        defaults = payload["defaults"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"artifact is missing required field: {error}") from error

    if not isinstance(models, list) or not models or len(models) > 1_000:
        raise ValueError("model.models must contain between 1 and 1,000 entries")
    if not all(isinstance(value, str) and 0 < len(value) <= 100 for value in models):
        raise ValueError("model.models entries must be non-empty strings up to 100 characters")
    if len(set(models)) != len(models):
        raise ValueError("model.models entries must be unique")
    if not isinstance(model_coefficients, list):
        raise TypeError("model.model_coefficients must be an array")
    if len(model_coefficients) != len(models):
        raise ValueError("model coefficients must align with model names")
    if numeric_features != ["Year", "Mileage", "Rating"]:
        raise ValueError("numeric feature order is incompatible with this browser")
    if not all(
        isinstance(values, list) and len(values) == 3
        for values in (numeric_mean, numeric_scale, numeric_coefficients)
    ):
        raise ValueError("numeric model arrays must each contain three values")
    numeric_values = [
        model["intercept"],
        *model_coefficients,
        *numeric_mean,
        *numeric_scale,
        *numeric_coefficients,
    ]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in numeric_values
    ):
        raise ValueError("model coefficients and scaling values must be finite numbers")
    if not all(value > 0 for value in numeric_scale):
        raise ValueError("numeric scales must be greater than zero")

    if not isinstance(bounds, dict) or not isinstance(defaults, dict):
        raise TypeError("bounds and defaults must be objects")
    for name in ("year", "mileage", "rating", "price"):
        values = bounds.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            or values[0] > values[1]
        ):
            raise ValueError(f"bounds.{name} must be an ordered numeric pair")
    if defaults.get("model") not in models:
        raise ValueError("default model must be present in model.models")
    for name in ("year", "mileage", "rating"):
        value = defaults.get(name)
        if not isinstance(value, (int, float)) or not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError(f"defaults.{name} must be within bounds.{name}")
    return payload


def serialized_payload() -> str:
    return json.dumps(validate_artifact_contract(export_payload()), indent=2, sort_keys=True) + "\n"


def payloads_equivalent(actual, expected) -> bool:
    """Compare generated payloads while tolerating harmless BLAS float drift."""

    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(payloads_equivalent(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                payloads_equivalent(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            actual,
            expected,
            rel_tol=1e-7,
            abs_tol=1e-4,
        )
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized_payload()

    if args.check:
        if not args.output.exists():
            print(f"Static model is stale: {args.output}")
            return 1
        try:
            current_payload = json.loads(args.output.read_text())
            validate_artifact_contract(current_payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Static model is unreadable: {args.output}")
            print(error)
            return 1
        if not payloads_equivalent(current_payload, json.loads(expected)):
            print(f"Static model is stale: {args.output}")
            return 1
        print(f"Static model is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
