import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('usa_mercedes_benz_prices.csv')

# Drop 'Brand' column
df.drop('Brand', axis=1, inplace=True)

# Merge 'Mileage_k' and 'Mileage' into one column
df['Mileage'] = df['Mileage_k'] * 1000 + df['Mileage']
df.drop('Mileage_k', axis=1, inplace=True)

# Convert 'Review Count' and 'Price' to numeric
df['Review Count'] = df['Review Count'].str.replace(',', '').astype(float)
df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '').astype(float)

X = df[['Year', 'Model', 'Mileage', 'Rating']]
y = df['Price']

cat_columns = ['Model']
# Assuming num_columns is a list of the numerical columns in your data
num_columns = ['Year', 'Mileage', 'Rating']

# Assuming X_train, X_test, y_train, y_test are your training and test data and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline for preprocessing categorical columns
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Add handle_unknown parameter here
])

# Combine preprocessing steps for numerical and categorical columns
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), num_columns),
    ('cat', cat_pipeline, cat_columns)
])


"""
# Initialize RandomForestRegressor
regressor = RandomForestRegressor()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
"""

def predict_car_price(model, mileage, rating, year):
    # Construct input data with all necessary columns
    input_data = pd.DataFrame(
        {'Model': [model], 'Mileage': [mileage], 'Rating': [rating], 'Year': [year], 'Review Count': [500]})

    # Predict the price
    predicted_price = pipeline.predict(input_data)[0]
    return predicted_price


# Train your model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())  # You can use your preferred regressor here
])

pipeline.fit(X_train, y_train)

print("enter mercedes benz model only. For example Mercedes gle 43 should be entered properly capitalized in the format of 'GLE 43'.\nModels that dont follow mercedes nomenclature should be capitalized normally such as mercedes metris being 'Metris'")


model = input("Enter car model: ")
mileage = float(input("Enter mileage: "))
rating = float(input("Enter rating: "))
year = int(input("Enter year: "))


# Now, you can use the predict_car_price function to get predictions
estimated_price = predict_car_price(model, mileage, rating, year)
print("Estimated price:", estimated_price)



gf = pd.read_csv('usa_mercedes_benz_prices.csv')

"""
def graphModels(cars):
    plt.figure(figsize=(15, 5*len(cars)))

    for i, car in enumerate(cars, start=1):
        plt.subplot(len(cars), 2, i*2-1)
        plotYear(car)
        plt.subplot(len(cars), 2, i*2)
        plotMileage(car)

    plt.tight_layout()
    plt.show()

def plotYear(car):
    car_data = gf[gf['Model'] == car]
    if car_data.empty:
        print(f"No data found for {car}")
        return
    car_data_sorted = car_data.sort_values(by='Price')
    price = car_data_sorted['Price']
    year = car_data_sorted['Year']
    plt.scatter(year, price, label=car, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(f'Price vs Year - {car}')
    plt.legend()
    plt.xlim(min(year) - 1, max(year) + 1)
    plt.ylim(1000, 100000)  # Adjust the y-axis range

def plotMileage(car):
    car_data = gf[gf['Model'] == car]
    if car_data.empty:
        print(f"No data found for {car}")
        return
    car_data_sorted = car_data.sort_values(by='Price')
    price = car_data_sorted['Price']
    mileage = car_data_sorted['Mileage']
    plt.scatter(mileage, price, label=car, marker='o')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title(f'Price vs Mileage - {car}')
    plt.legend()
    plt.xlim(0, max(mileage) + 10000)  # Adjust the x-axis range
    plt.ylim(10000, 100000)  # Adjust the y-axis range

# Example usage:
cars = ["GLC 300"]
graphModels(cars)

# Example usage:
"""

""""""
def plotYear(car):
    car_data = gf[gf['Model'] == car]
    if car_data.empty:
        print(f"No data found for {car}")
        return
    car_data_sorted = car_data.sort_values(by='Year')
    price = car_data_sorted['Price']
    year = car_data_sorted['Year']
    plt.plot(year, price, label=car)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(f'Price vs Year - {car}')
    plt.legend()
    plt.grid(True)

def plotMileage(car):
    car_data = gf[gf['Model'] == car]
    if car_data.empty:
        print(f"No data found for {car}")
        return
    car_data_sorted = car_data.sort_values(by='Mileage')
    price = car_data_sorted['Price']
    mileage = car_data_sorted['Mileage']
    plt.plot(mileage, price, label=car)
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title(f'Price vs Mileage - {car}')
    plt.legend()
    plt.grid(True)

# Plotting for GLC 300
plotYear("GLB 250")
plt.show()

plotMileage("GLB 250")
plt.show()