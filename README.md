# Car Price Calculator

An educational Mercedes-Benz listing-price estimator built from a saved dataset. The local Flask app trains a scikit-learn Random Forest; the dependency-free static demo uses a smaller exported Ridge model so it can run on GitHub Pages.

The project is aimed at reviewers and learners who want to inspect a complete data-to-browser workflow. It is not a live valuation tool or purchasing advice.

## Try the static demo

[Open the live GitHub Pages demo](https://gauravsdama.github.io/carPriceCalculator/).

The deployed source is in `docs/demo` and uses only relative files. To run it locally:

```bash
python -m http.server 8000 --directory docs/demo
```

Open `http://127.0.0.1:8000`. The browser loads `model.json` from the same static site and sends no vehicle inputs to an application server.

## Run the Flask app

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python mercedesbenzRIDGE.py
```

Or with standard Python packaging tools:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mercedesbenzRIDGE.py
```

Open `http://127.0.0.1:5000`.

## Rebuild and verify

```bash
uv run python scripts/export_static_model.py
uv run python scripts/export_static_model.py --check
uv run ruff check .
uv run pytest
uv run pip-audit
```

The exported JSON is deterministic for the checked-in data and code. CI repeats lint, tests, artifact freshness, dependency audit, and a static-server smoke check.

## Data and model limits

The repository includes 2,429 saved asking-price listings. `usa_mercedes_benz_prices.csv` is a transformed copy of Danish Ammar's *USA Mercedes Benz Prices Dataset*, version 1. Kaggle metadata marks the dataset Apache-2.0 and was last updated April 26, 2024; it does not state when the listings were collected.

The models use year, normalized model name, mileage, and dealer rating. They do not know vehicle condition, detailed trim, options, location, accident history, verified sale price, or market changes after the snapshot. Dealer rating describes the dealer rather than the vehicle.

Reported metrics come from one deterministic random 80/20 holdout. Duplicate or related listings may cross that split, so the results do not establish future-market accuracy. The static Ridge estimator and Flask Random Forest intentionally report their methods separately.

See `THIRD_PARTY_NOTICES.md` for dataset provenance and `docs/RELEASE_READINESS.md` for the current publication decision.

## Licensing

The project is licensed under Apache-2.0. Dataset attribution and its Apache-2.0 license text are included in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.
