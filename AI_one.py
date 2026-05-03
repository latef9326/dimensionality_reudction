import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ======================================
# ORIGINAL DATA
# ======================================

# Apartment sizes (in m²)
X = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)

# Apartment prices (in thousands of PLN)
y = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650])

# ======================================
# CREATE AND TRAIN MODEL
# ======================================

model = LinearRegression()
model.fit(X, y)

print("=== ORIGINAL MODEL ===")
print(f"Slope (a): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# ======================================
# PREDICTIONS
# ======================================

sizes_to_predict = np.array([[75], [95], [150]])
predictions = model.predict(sizes_to_predict)

print("\n=== PREDICTIONS ===")
for size, price in zip(sizes_to_predict, predictions):
    print(f"Predicted price of {size[0]} m² apartment: {price:.2f} thousand PLN")

# ======================================
# VISUALIZATION – ORIGINAL MODEL
# ======================================

plt.scatter(X, y, color='green', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Apartment size (m²)')
plt.ylabel('Apartment price (thousand PLN)')
plt.title('Linear Regression – Original Data')
plt.legend()
plt.show()


# ======================================
# EXERCISE 3 – Increase all prices by 100
# ======================================

y_modified = y + 100

model_modified = LinearRegression()
model_modified.fit(X, y_modified)

print("\n=== MODEL AFTER INCREASING PRICES BY 100 ===")
print(f"Slope (a): {model_modified.coef_[0]}")
print(f"Intercept (b): {model_modified.intercept_}")

plt.scatter(X, y_modified, color='blue', label='Modified data')
plt.plot(X, model_modified.predict(X), color='red', label='Regression line')
plt.xlabel('Apartment size (m²)')
plt.ylabel('Apartment price (thousand PLN)')
plt.title('Linear Regression – Prices Increased by 100')
plt.legend()
plt.show()


# ======================================
# EXERCISE 4 – Add an Outlier
# ======================================

# Add a very small but very expensive apartment
X_outlier = np.append(X, [[10]], axis=0)
y_outlier = np.append(y, 500)

model_outlier = LinearRegression()
model_outlier.fit(X_outlier, y_outlier)

print("\n=== MODEL AFTER ADDING OUTLIER ===")
print(f"Slope (a): {model_outlier.coef_[0]}")
print(f"Intercept (b): {model_outlier.intercept_}")

plt.scatter(X_outlier, y_outlier, color='purple', label='Data with outlier')
plt.plot(X_outlier, model_outlier.predict(X_outlier), color='red', label='Regression line')
plt.xlabel('Apartment size (m²)')
plt.ylabel('Apartment price (thousand PLN)')
plt.title('Linear Regression – With Outlier')
plt.legend()
plt.show()