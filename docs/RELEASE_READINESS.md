# Release readiness

Checkpoint: 2026-09-15

Status: **local release candidate; not yet published**.

## Product boundary

The intended audience is a hiring reviewer or learner evaluating a small end-to-end machine-learning project. The honest story is: this repository cleans a saved Mercedes-Benz listing dataset, trains a local Random Forest for an educational Flask estimator, and exports a smaller Ridge snapshot for a dependency-free browser demo.

Neither surface is a vehicle valuation service. The data contains asking prices rather than verified sales, has no collection timestamp or vehicle location, and omits condition, trim detail, options, accident history, and later market changes. Dealer rating describes the dealer, not the car.

## Provenance and ownership

- Canonical repository: `gauravsdama/carPriceCalculator`; the GitHub API reports it as public, non-fork, and unarchived.
- First-party commits are attributed to Gaurav Dama / `gauravsdama`.
- Dataset provenance is recorded in `THIRD_PARTY_NOTICES.md`; Kaggle metadata declares Apache-2.0.
- First-party source code and documentation are licensed under Apache-2.0.

## Model evidence and limits

- Flask model: 180-tree `RandomForestRegressor`, `min_samples_leaf=2`, deterministic grouped 80/20 split, numeric passthrough, and one-hot model encoding.
- Static demo: Ridge regression with `alpha=10`, evaluated on the same deterministic split and refit on all saved rows for export.
- Both evaluations keep identical feature vectors on the same side, preventing duplicate-feature leakage. Closely related but non-identical listings may still cross the split, and neither model has temporal, grouped-by-model-name, or external validation.
- Unknown models are rejected in both UIs. Inputs are bounded to saved-data ranges.

## Release package

- Source dataset and deterministic model exporter are retained.
- `pyproject.toml` is the sole dependency manifest and `uv.lock` is the reproducible lock.
- CI runs formatting, lint, unit tests, and artifact freshness across Python 3.11–3.14, plus a dependency audit and real Chrome calculation tests.
- The static artifact has a checked-in JSON Schema plus matching semantic validation in Python and the browser.
- The static demo uses only relative assets and no network request except the user-selected dataset link.
- UI copy is inventoried in `docs/product/UI_COPY.md`.

## Known follow-ups

1. The transformed dataset derivation should be reviewed against the original transformation notebook or script if one exists; none is present in repository history.
2. The current holdout is educational evidence only. A stronger claim would require grouped and time-aware evaluation on licensed data with a collection date.

## Verification evidence

Run successfully in this checkout on 2026-09-15:

- `uv sync --locked`
- `uv run ruff format --check .` — 11 Python files already formatted
- `uv run ruff check .` — all checks passed
- Python 3.11–3.14 — 15 non-browser tests passed on each declared version
- Chrome end-to-end suite — 2 tests passed, covering calculations and incompatible-artifact rejection
- Python 3.11–3.14 — generated artifact is current on each declared version
- `uv run pip-audit` — no known vulnerabilities

The Flask Random Forest grouped holdout reports R² `0.84`, RMSE `$13,172`, and MAE `$7,562`. The static Ridge grouped holdout reports R² `0.799948`, RMSE `$14,570.38`, and MAE `$8,589.36`. These changed metrics reflect a different deterministic grouped sample and should not be compared as evidence of model improvement.

The public GitHub Pages site still represents published commit `847b84d`; this local release candidate has not been pushed or deployed.
