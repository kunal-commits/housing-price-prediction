# Linear Regression - Housing Price Prediction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv('Housing.csv')  # Ensure this file is in your working directory

# 2. Preprocess data
# Convert 'yes'/'no' to 1/0
df.replace({'yes': 1, 'no': 0}, inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Features and target
X = df.drop('price', axis=1)
y = df['price']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# 7. Coefficients
print("\nModel Coefficients:")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# 8. Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
