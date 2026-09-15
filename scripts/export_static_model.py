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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mercedesbenzRIDGE import DATA_PATH, load_data

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "demo" / "model.json"
DATASET_URL = "https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset"
FLOAT_PRECISION = 8


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
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            ["Year", "Mileage", "Rating"],
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encoder",
                                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                    ),
                                ]
                            ),
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
    features = data[["Year", "Model", "Mileage", "Rating"]]
    target = data["Price"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    evaluation_model = build_pipeline().fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)

    final_model = build_pipeline().fit(features, target)
    preprocessor = final_model.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["num"]
    scaler = numeric.named_steps["scaler"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    regressor = final_model.named_steps["regressor"]
    models = [str(value) for value in encoder.categories_[0]]
    coefficients = [float(value) for value in regressor.coef_]

    return {
        "schema_version": 1,
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
            "intercept": round(float(regressor.intercept_), FLOAT_PRECISION),
            "numeric_mean": _rounded(scaler.mean_),
            "numeric_scale": _rounded(scaler.scale_),
            "numeric_coefficients": _rounded(coefficients[:3]),
            "models": models,
            "model_coefficients": _rounded(coefficients[3:]),
        },
        "evaluation": {
            "method": "Single deterministic 80/20 random holdout (random_state=42)",
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
            "A single random holdout does not establish performance over time or on unseen models.",
        ],
    }


def serialized_payload() -> str:
    return json.dumps(export_payload(), indent=2, sort_keys=True) + "\n"


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
        except (OSError, json.JSONDecodeError):
            print(f"Static model is unreadable: {args.output}")
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
