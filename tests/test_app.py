from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from car_price_data import (
    MODEL_FEATURES,
    DatasetValidationError,
    grouped_evaluation_split,
    load_data,
)
from mercedesbenzRIDGE import app, predict_car_price, train_model
from scripts.export_static_model import payloads_equivalent, validate_artifact_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_dataset_and_model_are_deterministic():
    data = load_data()
    first = predict_car_price("GLC 300", 36000, 4.6, 2022)
    train_model.cache_clear()
    second = predict_car_price("GLC 300", 36000, 4.6, 2022)

    assert len(data) == 2429
    assert first == pytest.approx(second)


def test_dataset_loader_rejects_schema_and_value_drift(tmp_path):
    data_path = ROOT / "usa_mercedes_benz_prices.csv"
    raw = data_path.read_text()

    missing_column = tmp_path / "missing-column.csv"
    missing_column.write_text(raw.replace(",Brand,", ",", 1))
    with pytest.raises(DatasetValidationError, match="Dataset columns"):
        load_data(missing_column)

    invalid_year = tmp_path / "invalid-year.csv"
    invalid_year.write_text(raw.replace("2021,Mercedes-Benz", "unknown,Mercedes-Benz", 1))
    with pytest.raises(DatasetValidationError, match="Year contains a non-numeric value"):
        load_data(invalid_year)


def test_evaluation_split_keeps_duplicate_feature_vectors_together():
    data = load_data()
    features = data.loc[:, MODEL_FEATURES]
    x_train, x_test, _, _ = grouped_evaluation_split(features, data["Price"])
    train_keys = set(x_train.itertuples(index=False, name=None))
    test_keys = set(x_test.itertuples(index=False, name=None))

    assert train_keys.isdisjoint(test_keys)
    assert len(x_train) + len(x_test) == len(data)


def test_get_starts_without_a_prediction(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Explore a dataset-based listing estimate." in response.data
    assert b"Random Forest output from the saved dataset" not in response.data


def test_valid_post_returns_a_prediction(client):
    response = client.post(
        "/",
        data={"model": "GLC 300", "year": "2022", "mileage": "36000", "rating": "4.6"},
    )

    assert response.status_code == 200
    assert b"Estimated listing price" in response.data
    assert b"Educational estimate from saved listing data" in response.data


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "Unknown 9000", b"Choose a Mercedes-Benz model"),
        ("year", "soon", b"Use numbers for model year"),
        ("year", "2030", b"Model year must be a whole number"),
        ("mileage", "-1", b"Mileage must be between"),
        ("rating", "7", b"Dealer rating must be between"),
    ],
)
def test_invalid_posts_return_recoverable_errors(client, field, value, message):
    form = {"model": "GLC 300", "year": "2022", "mileage": "36000", "rating": "4.6"}
    form[field] = value

    response = client.post("/", data=form)

    assert response.status_code == 200
    assert message in response.data
    assert b"Estimated listing price" not in response.data


def test_static_artifact_is_current_and_well_formed():
    result = subprocess.run(
        [sys.executable, "scripts/export_static_model.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    artifact = json.loads((ROOT / "docs/demo/model.json").read_text())
    schema = json.loads((ROOT / "docs/demo/model.schema.json").read_text())

    assert result.returncode == 0, result.stdout + result.stderr
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)
    validate_artifact_contract(artifact)
    assert artifact["artifact_type"] == "car-price-calculator.static-model"
    assert artifact["schema_version"] == 1
    assert artifact["dataset"]["rows"] == 2429
    assert artifact["dataset"]["license"] == "Apache-2.0"
    assert len(artifact["model"]["models"]) == len(artifact["model"]["model_coefficients"])

    incompatible = json.loads(json.dumps(artifact))
    incompatible["model"]["model_coefficients"].pop()
    with pytest.raises(ValueError, match="align with model names"):
        validate_artifact_contract(incompatible)


def test_static_artifact_comparison_tolerates_only_small_numeric_drift():
    expected = {"model": {"coefficient": 123.4567}, "models": ["GLC 300"]}

    assert payloads_equivalent(
        {"model": {"coefficient": 123.45671}, "models": ["GLC 300"]},
        expected,
    )
    assert not payloads_equivalent(
        {"model": {"coefficient": 123.5}, "models": ["GLC 300"]},
        expected,
    )
    assert not payloads_equivalent(
        {"model": {"coefficient": 123.4567}, "models": ["E-Class"]},
        expected,
    )


def test_static_demo_has_five_saved_examples():
    html = (ROOT / "docs/demo/index.html").read_text()

    assert html.count('class="preset"') == 5
    assert "Examples" in html
