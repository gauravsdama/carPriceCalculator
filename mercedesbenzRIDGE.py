from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
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
            --bg: #f4f7fa;
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
            background:
                linear-gradient(120deg, rgba(36, 87, 166, 0.11), transparent 34%),
                linear-gradient(90deg, rgba(17, 24, 39, 0.045) 1px, transparent 1px),
                var(--bg);
            background-size: auto, 92px 92px, auto;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
            border-radius: 8px;
            background: var(--ink);
            color: #ffffff;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 0.95fr) minmax(340px, 0.82fr);
            gap: 26px;
            align-items: start;
        }

        h1, h2, p { margin-top: 0; }

        .eyebrow {
            margin: 0 0 12px;
            color: var(--blue);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
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
                <span class="brand-mark">MB</span>
                <span>Car Price Calculator</span>
            </div>
            <p class="eyebrow">RandomForest estimator</p>
        </header>

        <section class="layout">
            <div>
                <p class="eyebrow">Mercedes-Benz used market</p>
                <h1>Estimate a fair listing price from model, year, mileage, and rating.</h1>
                <p>This calculator cleans the local Mercedes-Benz listing dataset, trains a reusable scikit-learn pipeline, and returns an estimated price without command-line prompts.</p>

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
                        <span>Estimated price</span>
                        <strong>{{ prediction }}</strong>
                    </div>
                {% endif %}

                <div>
                    <label for="model">Model</label>
                    <input id="model" name="model" list="models" value="{{ values.model }}" required>
                    <datalist id="models">
                        {% for model in models %}
                            <option value="{{ model }}"></option>
                        {% endfor %}
                    </datalist>
                </div>

                <div>
                    <label for="year">Year</label>
                    <input id="year" name="year" type="number" min="{{ summary.min_year }}" max="{{ summary.max_year + 1 }}" value="{{ values.year }}" required>
                </div>

                <div>
                    <label for="mileage">Mileage</label>
                    <input id="mileage" name="mileage" type="number" min="0" step="500" value="{{ values.mileage }}" required>
                </div>

                <div>
                    <label for="rating">Dealer rating</label>
                    <input id="rating" name="rating" type="number" min="0" max="5" step="0.1" value="{{ values.rating }}" required>
                </div>

                <button type="submit">Estimate Price</button>
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

    df["Model"] = (
        df["Model"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df["Review Count"] = (
        df["Review Count"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
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
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    summary = {
        "rows": f"{len(df):,}",
        "r2": f"{r2_score(y_test, y_pred):.2f}",
        "rmse": _money(rmse),
        "min_year": int(df["Year"].min()),
        "max_year": int(df["Year"].max()),
        "year_range": f"{int(df['Year'].min())}-{int(df['Year'].max())}",
        "mileage_range": f"{int(df['Mileage'].min()):,}-{int(df['Mileage'].max()):,} mi",
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

    if request.method == "POST":
        values.update(
            {
                "model": request.form.get("model", values["model"]),
                "year": int(float(request.form.get("year", values["year"]))),
                "mileage": int(float(request.form.get("mileage", values["mileage"]))),
                "rating": float(request.form.get("rating", values["rating"])),
            }
        )

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
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
