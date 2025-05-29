import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
data = pd.read_csv(r"C:\Users\hp\Downloads\Housing.csv")

# Step 2: Encode categorical variables for multiple regression
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 3: Define features and target
# Simple Regression
X_simple = data_encoded[['area']]  
# Multiple Regression (selecting a few encoded and numeric columns)
X_multiple = data_encoded.drop('price', axis=1)

y = data_encoded['price']

# Step 4: Split the data
X_train_s, X_test_s, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_m, X_test_m, _, _ = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Step 5: Train models
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train)

model_multiple = LinearRegression()
model_multiple.fit(X_train_m, y_train)

# Step 6: Predictions
y_pred_s = model_simple.predict(X_test_s)
y_pred_m = model_multiple.predict(X_test_m)

# Step 7: Evaluation
print("Simple Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_s))
print("MSE:", mean_squared_error(y_test, y_pred_s))
print("R²:", r2_score(y_test, y_pred_s))

print("\n Multiple Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_m))
print("MSE:", mean_squared_error(y_test, y_pred_m))
print("R²:", r2_score(y_test, y_pred_m))

# Step 8: Plot simple regression
plt.scatter(X_test_s, y_test, color='blue', label='Actual Price')
plt.plot(X_test_s, y_pred_s, color='red', label='Predicted Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Coefficients
print("\n Model Coefficients:")
print("Simple Model Coefficient:", model_simple.coef_[0])
print("Simple Model Intercept:", model_simple.intercept_)
print("\nMultiple Model Coefficients:")
for feature, coef in zip(X_multiple.columns, model_multiple.coef_):
    print(f"{feature}: {coef}")
