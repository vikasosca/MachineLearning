from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import time

# 1. Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
housing_df = california.frame
print("Shape of the dataset:", housing_df.shape)
print("\nFirst 5 rows of the dataset:")
print(housing_df.head())
print("\nDescription of the dataset:")
print(california.DESCR)

# 2. Separate features (X) and target (y)
X = housing_df.drop('MedHouseVal', axis=1)
y = housing_df['MedHouseVal']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nShape of training features:", X_train.shape)
print("Shape of testing features:", X_test.shape)
print("Shape of training target:", y_train.shape)
print("Shape of testing target:", y_test.shape)

# 4. Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error on the test set: {mse:.2f}")

# 7. (Optional) Make a prediction for a new data point
# Let's take the first instance from the test set as a new example
new_house = X_test.iloc[[0]]
predicted_price = model.predict(new_house)[0]
actual_price = y_test.iloc[0]
print("\nPrediction for a new house:")
print("Features of the new house:")
print(new_house)
print(f"Predicted price: ${predicted_price * 1000:.2f}") # Prices are in units of $100,000
print(f"Actual price: ${actual_price * 1000:.2f}")
time.sleep(5)
print("-----------------------Polinial regression ---------------")

# 1. Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
housing_df = california.frame
X = housing_df.drop('MedHouseVal', axis=1)
y = housing_df['MedHouseVal']

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create polynomial features
poly = PolynomialFeatures(degree=2)  # You can experiment with different degrees
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("Original training features shape:", X_train.shape)
print("Polynomial training features shape (degree 2):", X_train_poly.shape)

# 4. Train a linear regression model on the polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 5. Make predictions on the test set using the polynomial features
y_pred_poly = poly_model.predict(X_test_poly)

# 6. Evaluate the polynomial regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"\nMean Squared Error with Polynomial Regression (degree 2): {mse_poly:.2f}")

# 7. (Optional) Make a prediction for the same new data point
new_house = X_test.iloc[[0]]
new_house_poly = poly.transform(new_house)
predicted_price_poly = poly_model.predict(new_house_poly)[0]
actual_price = y_test.iloc[0]
print("\nPrediction for the same new house using Polynomial Regression:")
print("Original features of the new house:")
print(new_house)
print(f"Predicted price (Polynomial): ${predicted_price_poly * 1000:.2f}")
print(f"Actual price: ${actual_price * 1000:.2f}")