# Release readiness

Checkpoint: 2026-09-15

Status: **published and verified**.

## Product boundary

The intended audience is a hiring reviewer or learner evaluating a small end-to-end machine-learning project. The honest story is: this repository cleans a saved Mercedes-Benz listing dataset, trains a local Random Forest for an educational Flask estimator, and exports a smaller Ridge snapshot for a dependency-free browser demo.

Neither surface is a vehicle valuation service. The data contains asking prices rather than verified sales, has no collection timestamp or vehicle location, and omits condition, trim detail, options, accident history, and later market changes. Dealer rating describes the dealer, not the car.

## Provenance and ownership

- Canonical repository: `gauravsdama/carPriceCalculator`; the GitHub API reports it as public, non-fork, and unarchived.
- First-party commits are attributed to Gaurav Dama / `gauravsdama`.
- Dataset provenance is recorded in `THIRD_PARTY_NOTICES.md`; Kaggle metadata declares Apache-2.0.
- First-party source code and documentation are licensed under Apache-2.0.

## Model evidence and limits

- Flask model: 180-tree `RandomForestRegressor`, `min_samples_leaf=2`, deterministic 80/20 random split, numeric imputation/scaling, and one-hot model encoding.
- Static demo: Ridge regression with `alpha=10`, evaluated on the same deterministic split and refit on all saved rows for export.
- Both evaluations are single random holdouts from one snapshot. Duplicate or closely related listings may cross the split, and neither model has temporal, grouped-by-model, or external validation.
- Unknown models are rejected in both UIs. Inputs are bounded to saved-data ranges.

## Release package

- Source dataset and deterministic model exporter are retained.
- `uv.lock` is the rebuild lock; `requirements.txt` remains the minimal pip entry point.
- CI runs lint, tests, artifact freshness, dependency audit, and a static-server smoke check.
- The static demo uses only relative assets and no network request except the user-selected dataset link.
- UI copy is inventoried in `docs/product/UI_COPY.md`.

## Known follow-ups

1. The transformed dataset derivation should be reviewed against the original transformation notebook or script if one exists; none is present in repository history.
2. The current holdout is educational evidence only. A stronger claim would require grouped and time-aware evaluation on licensed data with a collection date.

## Verification evidence

Run successfully in this checkout on 2026-09-15:

- `uv sync --locked`
- `uv run ruff format --check .` — 9 Python files already formatted
- `uv run ruff check .` — all checks passed
- `uv run pytest` — 13 tests passed
- `uv run python scripts/export_static_model.py --check` — generated artifact is current
- `uv run pip-audit` — no known vulnerabilities
- Static HTTP smoke check — the complete demo loaded from `docs/demo` with no console warnings or errors
- Live Flask browser check — native invalid-input feedback appeared, and a valid GLC 300 request (2021, 50,000 miles, dealer rating 4.7) returned `$33,380` with no console warnings or errors
- Responsive browser check at 390 × 844 — no horizontal overflow and the primary action remained 48 pixels tall
- Public GitHub Pages check — `https://gauravsdama.github.io/carPriceCalculator/` loaded the saved model, recalculated an estimate after input changes, and produced no console warnings or errors

The Flask Random Forest holdout reports R² `0.70`, RMSE `$16,669`, and MAE `$8,730`. The static Ridge holdout reports R² `0.7229`, RMSE `$16,143.77`, and MAE `$9,016.43`. Current desktop and mobile screenshots were captured during browser verification; they are test evidence, not committed marketing assets.
