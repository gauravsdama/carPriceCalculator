from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).with_name("usa_mercedes_benz_prices.csv")

app = Flask(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mercedes-Benz Price Calculator</title>
    <style>
        :root {
            --bg: #eef1f3;
            --ink: #111827;
            --muted: #5b677a;
            --panel: #ffffff;
            --line: #d8e1ea;
            --blue: #2457a6;
            --blue-dark: #173b73;
            --silver: #e8edf2;
            --green: #0f7b5f;
            --shadow: 0 22px 64px rgba(20, 36, 62, 0.12);
        }

        * { box-sizing: border-box; }

        body {
            min-width: 320px;
            margin: 0;
            color: var(--ink);
            background: linear-gradient(145deg, #f8fafb 0%, var(--bg) 62%, #e1e6ea 100%);
            font-family: "Avenir Next", Avenir, ui-sans-serif, system-ui, -apple-system, sans-serif;
        }

        .shell {
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 64px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 44px;
        }

        .brand {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            font-weight: 900;
        }

        .brand-mark {
            display: grid;
            width: 44px;
            height: 44px;
            place-items: center;
            border-radius: 50%;
            background: var(--ink);
            color: #ffffff;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 0.95fr) minmax(340px, 0.82fr);
            gap: 26px;
            align-items: start;
        }

        h1, h2, p { margin-top: 0; }

        .snapshot {
            margin: 0;
            color: var(--muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
        }

        h1 {
            max-width: 760px;
            margin-bottom: 16px;
            font-size: clamp(2.55rem, 6vw, 5rem);
            line-height: 0.96;
        }

        h2 {
            margin-bottom: 14px;
            font-size: 1.35rem;
        }

        p {
            color: var(--muted);
            line-height: 1.68;
        }

        .panel, .metric {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.93);
            box-shadow: var(--shadow);
        }

        .panel {
            padding: 24px;
        }

        .form-grid {
            display: grid;
            gap: 15px;
        }

        label {
            display: block;
            margin-bottom: 7px;
            color: #263244;
            font-size: 0.86rem;
            font-weight: 850;
        }

        input, select {
            width: 100%;
            min-height: 46px;
            border: 1px solid #c8d3df;
            border-radius: 8px;
            padding: 10px 12px;
            background: #ffffff;
            color: var(--ink);
            font: inherit;
        }

        input:focus, select:focus, button:focus {
            outline: 3px solid rgba(36, 87, 166, 0.22);
            outline-offset: 2px;
        }

        button {
            min-height: 48px;
            border: 0;
            border-radius: 8px;
            background: var(--blue-dark);
            color: #ffffff;
            font: inherit;
            font-weight: 900;
            cursor: pointer;
        }

        .result {
            display: grid;
            gap: 8px;
            margin-bottom: 20px;
            border: 1px solid rgba(15, 123, 95, 0.24);
            border-radius: 8px;
            padding: 18px;
            background: #eefaf6;
        }

        .result span {
            color: var(--green);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
        }

        .result strong {
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1;
        }

        .result small, .disclaimer {
            color: #315348;
            line-height: 1.5;
        }

        .alert {
            border-left: 4px solid #a33b2e;
            padding: 12px 14px;
            background: #fff2ef;
            color: #76291f;
            line-height: 1.5;
        }

        .disclaimer {
            margin: 18px 0 0;
            border-top: 1px solid var(--line);
            padding-top: 16px;
            font-size: 0.86rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 24px;
        }

        .metric {
            padding: 16px;
        }

        .metric span {
            display: block;
            margin-bottom: 8px;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 850;
        }

        .metric strong {
            font-size: 1.25rem;
        }

        .sample-list {
            display: grid;
            gap: 9px;
            margin: 18px 0 0;
            padding: 0;
            list-style: none;
        }

        .sample-list li {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 11px 12px;
            background: #ffffff;
            color: #303c4e;
            font-weight: 750;
        }

        .sample-list span {
            color: var(--muted);
            font-weight: 700;
        }

        @media (max-width: 860px) {
            .topbar { align-items: flex-start; flex-direction: column; }
            .layout, .metric-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <header class="topbar">
            <div class="brand">
                <span class="brand-mark">CPC</span>
                <span>Car Price Calculator</span>
            </div>
            <p class="snapshot">{{ summary.rows }} saved listings · Random Forest</p>
        </header>

        <section class="layout">
            <div>
                <h1>Explore a dataset-based listing estimate.</h1>
                <p>This local demo trains on a saved snapshot of Mercedes-Benz asking prices. It is useful for exploring the model, not valuing a specific vehicle.</p>

                <div class="metric-grid" aria-label="Model summary">
                    <div class="metric">
                        <span>Listings</span>
                        <strong>{{ summary.rows }}</strong>
                    </div>
                    <div class="metric">
                        <span>Model R2</span>
                        <strong>{{ summary.r2 }}</strong>
                    </div>
                    <div class="metric">
                        <span>RMSE</span>
                        <strong>{{ summary.rmse }}</strong>
                    </div>
                </div>

                <ul class="sample-list" aria-label="Dataset range">
                    <li><span>Year range</span><strong>{{ summary.year_range }}</strong></li>
                    <li><span>Mileage range</span><strong>{{ summary.mileage_range }}</strong></li>
                    <li><span>Median price</span><strong>{{ summary.median_price }}</strong></li>
                </ul>
            </div>

            <form class="panel form-grid" method="POST">
                {% if prediction %}
                    <div class="result" aria-live="polite">
                        <span>Estimated listing price</span>
                        <strong>{{ prediction }}</strong>
                        <small>Random Forest output from the saved dataset. Actual prices may differ.</small>
                    </div>
                {% endif %}

                {% if error %}
                    <div class="alert" role="alert">{{ error }}</div>
                {% endif %}

                <div>
                    <label for="model">Mercedes-Benz model</label>
                    <select id="model" name="model" required>
                        {% for model in models %}
                            <option value="{{ model }}" {% if model == values.model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label for="year">Model year</label>
                    <input id="year" name="year" type="number" min="{{ summary.min_year }}" max="{{ summary.max_year }}" value="{{ values.year }}" required>
                </div>

                <div>
                    <label for="mileage">Mileage (miles)</label>
                    <input id="mileage" name="mileage" type="number" min="0" max="{{ summary.max_mileage }}" step="500" value="{{ values.mileage }}" required>
                </div>

                <div>
                    <label for="rating">Dealer rating (0–5)</label>
                    <input id="rating" name="rating" type="number" min="0" max="5" step="0.1" value="{{ values.rating }}" required>
                </div>

                <button type="submit">Estimate listing price</button>
                <p class="disclaimer">Educational estimate from saved listing data—not a live valuation or buying recommendation.</p>
            </form>
        </section>
    </main>
</body>
</html>
"""


def _money(value: float) -> str:
    return f"${value:,.0f}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Brand"], errors="ignore")

    mileage_k = pd.to_numeric(df.pop("Mileage_k"), errors="coerce").fillna(0)
    mileage_remainder = pd.to_numeric(df["Mileage"], errors="coerce").fillna(0)
    df["Mileage"] = (mileage_k * 1000) + mileage_remainder

    df["Model"] = df["Model"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["Review Count"] = (
        df["Review Count"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    return df.dropna(subset=["Year", "Model", "Mileage", "Rating", "Price"])


@lru_cache(maxsize=1)
def train_model():
    df = load_data()
    x = df[["Year", "Model", "Mileage", "Rating"]]
    y = df["Price"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
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
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Model"],
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=180,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    summary = {
        "rows": f"{len(df):,}",
        "r2": f"{r2_score(y_test, y_pred):.2f}",
        "rmse": _money(rmse),
        "mae": _money(mae),
        "min_year": int(df["Year"].min()),
        "max_year": int(df["Year"].max()),
        "year_range": f"{int(df['Year'].min())}-{int(df['Year'].max())}",
        "mileage_range": f"{int(df['Mileage'].min()):,}-{int(df['Mileage'].max()):,} mi",
        "max_mileage": int(df["Mileage"].max()),
        "median_price": _money(df["Price"].median()),
    }
    models = sorted(df["Model"].unique())
    return pipeline, models, summary


def predict_car_price(model: str, mileage: float, rating: float, year: int) -> float:
    pipeline, _, _ = train_model()
    input_data = pd.DataFrame(
        {
            "Model": [model.strip()],
            "Mileage": [float(mileage)],
            "Rating": [float(rating)],
            "Year": [int(year)],
        }
    )
    return max(0.0, float(pipeline.predict(input_data)[0]))


def validate_values(form, models: list[str], summary: dict) -> tuple[dict, str | None]:
    values = {
        "model": (form.get("model") or "").strip(),
        "year": (form.get("year") or "").strip(),
        "mileage": (form.get("mileage") or "").strip(),
        "rating": (form.get("rating") or "").strip(),
    }

    try:
        year_number = float(values["year"])
        mileage_number = float(values["mileage"])
        rating_number = float(values["rating"])
    except ValueError:
        return values, "Use numbers for model year, mileage, and dealer rating."

    if values["model"] not in models:
        return values, "Choose a Mercedes-Benz model from the dataset."
    if (
        not year_number.is_integer()
        or not summary["min_year"] <= year_number <= summary["max_year"]
    ):
        return (
            values,
            f"Model year must be a whole number from {summary['min_year']} to {summary['max_year']}.",
        )
    if not 0 <= mileage_number <= summary["max_mileage"]:
        return values, f"Mileage must be between 0 and {summary['max_mileage']:,}."
    if not 0 <= rating_number <= 5:
        return values, "Dealer rating must be between 0 and 5."

    values.update(
        {
            "year": int(year_number),
            "mileage": int(mileage_number),
            "rating": rating_number,
        }
    )
    return values, None


@app.route("/", methods=["GET", "POST"])
def index():
    _, models, summary = train_model()
    default_model = "GLC 300" if "GLC 300" in models else models[0]
    values = {
        "model": default_model,
        "year": min(2022, summary["max_year"]),
        "mileage": 36000,
        "rating": 4.6,
    }

    prediction = None
    error = None
    if request.method == "POST":
        values, error = validate_values(request.form, models, summary)
        if error is None:
            prediction = _money(
                predict_car_price(
                    values["model"],
                    values["mileage"],
                    values["rating"],
                    values["year"],
                )
            )

    return render_template_string(
        PAGE_TEMPLATE,
        models=models,
        prediction=prediction,
        summary=summary,
        values=values,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
