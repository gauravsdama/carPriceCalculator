"""Strict, framework-independent dataset loading and evaluation splitting."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path(__file__).with_name("usa_mercedes_benz_prices.csv")
RAW_COLUMNS = (
    "Year",
    "Brand",
    "Model",
    "Mileage_k",
    "Mileage",
    "Rating",
    "Review Count",
    "Price",
)
MODEL_FEATURES = ("Year", "Model", "Mileage", "Rating")


class DatasetValidationError(ValueError):
    """Raised when the checked-in dataset does not satisfy its input contract."""


def _numeric(series: pd.Series, field: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"{field} contains a non-numeric value") from error
    if not values.notna().all() or not values.map(math.isfinite).all():
        raise DatasetValidationError(f"{field} contains a missing or non-finite value")
    return values


def _require_whole_numbers(series: pd.Series, field: str) -> None:
    if not (series % 1 == 0).all():
        raise DatasetValidationError(f"{field} must contain whole numbers")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and normalize a dataset only after validating the raw CSV contract."""

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError(f"Could not read dataset: {path}") from error

    actual_columns = tuple(raw.columns)
    if actual_columns != RAW_COLUMNS:
        raise DatasetValidationError(
            f"Dataset columns must be {list(RAW_COLUMNS)}; received {list(actual_columns)}"
        )
    if raw.empty:
        raise DatasetValidationError("Dataset must contain at least one listing")

    blank_columns = [column for column in RAW_COLUMNS if raw[column].str.strip().eq("").any()]
    if blank_columns:
        raise DatasetValidationError(
            f"Dataset contains blank values in: {', '.join(blank_columns)}"
        )
    if not raw["Brand"].eq("Mercedes-Benz").all():
        raise DatasetValidationError("Brand must be Mercedes-Benz for every listing")

    year = _numeric(raw["Year"], "Year")
    mileage_thousands = _numeric(raw["Mileage_k"], "Mileage_k")
    mileage_remainder = _numeric(raw["Mileage"], "Mileage")
    rating = _numeric(raw["Rating"], "Rating")
    review_count = _numeric(raw["Review Count"].str.replace(",", "", regex=False), "Review Count")
    price_text = raw["Price"].str.strip()
    if not price_text.str.fullmatch(r"\$?\d[\d,]*(?:\.\d{1,2})?").all():
        raise DatasetValidationError("Price must use a numeric currency format")
    price = _numeric(
        price_text.str.removeprefix("$").str.replace(",", "", regex=False),
        "Price",
    )

    for values, field in (
        (year, "Year"),
        (mileage_thousands, "Mileage_k"),
        (mileage_remainder, "Mileage"),
        (review_count, "Review Count"),
    ):
        _require_whole_numbers(values, field)

    if not year.between(1886, 2100).all():
        raise DatasetValidationError("Year must be between 1886 and 2100")
    if not mileage_thousands.ge(0).all() or not mileage_remainder.between(0, 999).all():
        raise DatasetValidationError("Mileage components must describe non-negative miles")
    if not rating.between(0, 5).all():
        raise DatasetValidationError("Rating must be between 0 and 5")
    if not review_count.ge(0).all():
        raise DatasetValidationError("Review Count must be non-negative")
    if not price.gt(0).all():
        raise DatasetValidationError("Price must be greater than zero")

    model = raw["Model"].str.strip().str.replace(r"\s+", " ", regex=True)
    if model.eq("").any():
        raise DatasetValidationError("Model must not be blank")

    return pd.DataFrame(
        {
            "Year": year.astype(int),
            "Model": model,
            "Mileage": (mileage_thousands * 1000 + mileage_remainder).astype(int),
            "Rating": rating.astype(float),
            "Review Count": review_count.astype(int),
            "Price": price.astype(float),
        }
    )


def grouped_evaluation_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split without placing identical model features on both sides."""

    feature_keys = pd.MultiIndex.from_frame(features.loc[:, MODEL_FEATURES])
    group_ids, _ = pd.factorize(feature_keys, sort=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(features, target, groups=group_ids))
    return (
        features.iloc[train_indices].copy(),
        features.iloc[test_indices].copy(),
        target.iloc[train_indices].copy(),
        target.iloc[test_indices].copy(),
    )
