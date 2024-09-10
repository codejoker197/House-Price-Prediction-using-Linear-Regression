import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = {
    'SquareFootage': [1500, 2500, 1800, 2200, 1600, 3000, 4000, 3500, 2800, 2700, 2300, 2400],
    'Bedrooms': [3, 4, 3, 4, 3, 5, 6, 4, 3, 4, 4, 3],
    'Bathrooms': [2, 3, 2, 2, 2, 4, 5, 3, 2, 3, 3, 2],
    'Price': [300000, 500000, 350000, 450000, 320000, 600000, 800000, 550000, 480000, 470000, 460000, 450000]
}


df = pd.DataFrame(data)


X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]

y = df['Price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2) Score: {r2}")


print("\nModel Coefficients:")
print(f"SquareFootage Coefficient: {model.coef_[0]}")
print(f"Bedrooms Coefficient: {model.coef_[1]}")
print(f"Bathrooms Coefficient: {model.coef_[2]}")
print(f"Intercept: {model.intercept_}")


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()