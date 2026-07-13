# Mercedes-Benz Price Calculator

Flask web app that estimates used Mercedes-Benz listing prices from model, year, mileage, and dealer rating.

The app cleans the included `usa_mercedes_benz_prices.csv` dataset, trains a scikit-learn `RandomForestRegressor`, shows model metrics, and serves a browser form for quick estimates.

This project is educational and should not be treated as financial or purchasing advice.

## Features

- Browser-based price calculator
- Reusable scikit-learn preprocessing and prediction pipeline
- One-hot encoding for model names with unknown-model handling
- Median imputation and scaling for numerical inputs
- Model summary cards for listing count, R2, RMSE, and data ranges

## Run Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python mercedesbenzRIDGE.py
```

Open `http://127.0.0.1:5000`.

## Project Files

```text
mercedesbenzRIDGE.py             Flask app and ML pipeline
usa_mercedes_benz_prices.csv     Local training dataset
requirements.txt                 Python dependencies
```
