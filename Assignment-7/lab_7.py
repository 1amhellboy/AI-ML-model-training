import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# 1 : Loading dataset

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

df = pd.read_csv(url)

df = df.rename(columns={"Hours": "Hours", "Scores": "Scores"})

print("Sample data:")
print(df.head())

X = df[["Hours"]]
y = df["Scores"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (Y):")
print(y.head())



# 2 : Model Implementation

# --- Linear Regression Model ---
lin_reg = LinearRegression()

# --- Polynomial Regression Model (degree 2) ---
poly = PolynomialFeatures(degree=2)


X_poly = poly.fit_transform(X)   # create new features [1, x, x^2]
poly_reg = LinearRegression()



# 3 : Training and Prediction

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Linear Regression ---
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# --- Train Polynomial Regression ---
X_train_poly = poly.fit_transform(X_train)   # transform train features
X_test_poly = poly.transform(X_test)         # transform test features
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

print("Linear Regression Predictions:", y_pred_lin[:5])
print("Polynomial Regression Predictions:", y_pred_poly[:5])



# 4: Evaluate Model Performance

# Linear Regression metrics
r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)

# Polynomial Regression metrics
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

print("Linear Regression -> R²:", r2_lin, " | MAE:", mae_lin)
print("Polynomial Regression -> R²:", r2_poly, " | MAE:", mae_poly)



# 5 : Plotting the Regression Fits

# --- Linear Regression Plot ---
plt.scatter(X, y, label="Data points")
plt.plot(X_test, y_pred_lin, 'r', label="Linear Regression")
plt.title("Linear Regression Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()

# --- Polynomial Regression Plot ---

# grid use karege for curve
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_grid_poly = poly.transform(X_grid)
y_grid_poly = poly_reg.predict(X_grid_poly)


plt.scatter(X, y, label="Data points")
plt.plot(X_grid, y_grid_poly, 'g', label="Polynomial Regression (deg=2)")
plt.title("Polynomial Regression Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()


# 6 : Discussion and Conclusion



