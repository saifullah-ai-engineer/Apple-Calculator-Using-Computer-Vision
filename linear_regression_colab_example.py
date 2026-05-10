"""
Linear Regression complete example for Google Colab.

What this code does:
1. Creates easy-to-understand mock house-price data.
2. Splits the data into training and testing sets.
3. Trains a Linear Regression model.
4. Tests the model with accuracy/error metrics.
5. Draws graphs for the data, predictions, and errors.

How to use in Google Colab:
- Open a new Colab notebook.
- Copy and paste this whole file into one code cell.
- Run the cell.
"""

# =========================
# 1. Import libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Make the random numbers repeatable so everyone gets the same result.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================
# 2. Create mock data
# =========================
# Example problem:
# Predict house price using house size.
# X = house size in square feet
# y = house price in dollars

number_of_houses = 100

# Generate house sizes between 500 and 3500 square feet.
house_size = np.random.randint(500, 3501, number_of_houses)

# Realistic simple formula:
# price = 50,000 base price + 150 dollars per square foot + random noise
noise = np.random.normal(loc=0, scale=30000, size=number_of_houses)
house_price = 50000 + (150 * house_size) + noise

# Put the data into a DataFrame so it is easy to read.
data = pd.DataFrame(
    {
        "House_Size_sqft": house_size,
        "House_Price_USD": house_price,
    }
)

print("First 5 rows of data:")
print(data.head())
print("\nData shape:", data.shape)


# =========================
# 3. Split input and output
# =========================
# X must be 2D for scikit-learn, so we use double brackets.
X = data[["House_Size_sqft"]]
y = data["House_Price_USD"]

# Train/test split:
# 80% data for training, 20% data for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================
# 4. Create and train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training complete!")
print("Intercept:", round(model.intercept_, 2))
print("Slope / Coefficient:", round(model.coef_[0], 2))


# =========================
# 5. Test the model
# =========================
y_pred = model.predict(X_test)

# Accuracy / error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Testing Results:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score / Accuracy:", round(r2, 4))
print("Accuracy Percentage:", str(round(r2 * 100, 2)) + "%")


# =========================
# 6. Compare actual vs predicted values
# =========================
results = pd.DataFrame(
    {
        "Actual_Price": y_test.values,
        "Predicted_Price": y_pred,
        "Difference": y_test.values - y_pred,
    }
)

print("\nActual vs Predicted sample:")
print(results.head(10))


# =========================
# 7. Predict a new example
# =========================
new_house_size = pd.DataFrame({"House_Size_sqft": [2000]})
predicted_price = model.predict(new_house_size)

print("\nPrediction for a 2000 sqft house:")
print("Estimated price: $" + str(round(predicted_price[0], 2)))


# =========================
# 8. Graph 1: Data and regression line
# =========================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.7)
plt.scatter(X_test, y_test, color="green", label="Testing Data", alpha=0.7)

# Regression line over the full size range.
line_x = np.linspace(data["House_Size_sqft"].min(), data["House_Size_sqft"].max(), 100)
line_x_df = pd.DataFrame({"House_Size_sqft": line_x})
line_y = model.predict(line_x_df)

plt.plot(line_x, line_y, color="red", linewidth=3, label="Regression Line")
plt.title("Linear Regression: House Size vs House Price")
plt.xlabel("House Size (sqft)")
plt.ylabel("House Price (USD)")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 9. Graph 2: Actual vs predicted prices
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="purple", alpha=0.8)

# Perfect prediction reference line.
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())
plt.plot([min_price, max_price], [min_price, max_price], color="red", linestyle="--")

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.grid(True)
plt.show()


# =========================
# 10. Graph 3: Prediction errors
# =========================
errors = y_test.values - y_pred

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=12, color="orange", edgecolor="black")
plt.title("Prediction Error Distribution")
plt.xlabel("Error = Actual Price - Predicted Price")
plt.ylabel("Number of Houses")
plt.grid(True)
plt.show()
