# carPriceCalculator
# Mercedes-Benz Price Prediction & Analysis

This project builds a machine learning pipeline to predict the price of used Mercedes-Benz vehicles based on various features such as model, year, mileage, and rating. It also includes code to visualize trends in pricing relative to year and mileage.

> **Note:** The project is designed for educational and research purposes only. Ensure you have the right to use and modify the data provided.

## Overview

The script performs the following tasks:
- **Data Preprocessing:**  
  - Loads a CSV file (`usa_mercedes_benz_prices.csv`) containing Mercedes-Benz pricing data.
  - Cleans the dataset by removing unnecessary columns and merging mileage values.
  - Converts fields like 'Review Count' and 'Price' to numerical values.
  
- **Machine Learning Pipeline:**  
  - Splits the data into training and testing sets.
  - Uses a combination of imputation, one-hot encoding (for the car model), and scaling to preprocess the features.
  - Trains a `RandomForestRegressor` to predict car prices.
  
- **Price Prediction:**  
  - Provides a function (`predict_car_price`) to estimate the price of a Mercedes-Benz based on user input.
  - Accepts user inputs for car model, mileage, rating, and year.
  
- **Visualization:**  
  - Contains functions (using Matplotlib) to plot trends such as price vs. year and price vs. mileage for selected models.

## Features

- **Data Cleaning & Transformation:**  
  Merges multiple mileage columns and converts monetary and count fields to proper numeric types.

- **Preprocessing Pipelines:**  
  Utilizes Scikit-Learn's `ColumnTransformer` and `Pipeline` to handle both categorical (car model) and numerical features.

- **Model Training:**  
  Implements a RandomForestRegressor to model the relationship between the features and the car price.

- **Interactive Prediction:**  
  The script prompts the user to input car details and outputs an estimated price.

- **Data Visualization:**  
  Generates line plots for price trends over year and mileage, helping to analyze market behavior for specific car models.

## Prerequisites

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- A CSV file named `usa_mercedes_benz_prices.csv` located in the same directory as the script

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/mercedes-price-prediction.git
   cd mercedes-price-prediction
