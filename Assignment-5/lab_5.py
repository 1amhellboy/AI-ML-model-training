# Lab 5 - Multiple Linear Regression with Data Preprocessing + Normalization
# Course: Artificial Intelligence and Machine Learning
# Task: Predict G3 (final grade) from student features

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# =======================
# 1. Load the dataset
# =======================
df = pd.read_csv("multiple_linear_data(in).csv")  # Adjust path if needed
print("First 5 rows of dataset:")
print(df.head())

# =======================
# 2. Check for missing values
# =======================
print("\nMissing values in each column:")
print(df.isnull().sum())

# =======================
# 3. Separate categorical and numeric columns
# =======================
categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(exclude=['object']).columns

print("\nCategorical columns:", categorical_cols.tolist())
print("Numeric columns:", numeric_cols.tolist())

# =======================
# 4. One-Hot Encode categorical columns
# drop_first=True → avoids dummy variable trap
# =======================
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nEncoded dataset (first 5 rows):")
print(df_encoded.head())

# =======================
# 5. Define X (features) and y (target)
# =======================
X = df_encoded.drop("G3", axis=1)
y = df_encoded["G3"]

# =======================
# 6. Normalize Features
# =======================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Now all features are between 0 and 1

# =======================
# 7. Train the Linear Regression Model
# (Training & testing on same dataset as per lab sheet)
# =======================
model = LinearRegression()
model.fit(X_scaled, y)

# =======================
# 8. Evaluate Model
# =======================
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
print(f"\nMean Squared Error (MSE) on same data: {mse:.2f}")

# =======================
# 9. Predict on new sample data
# =======================
# Example new data: fill values in same order as X.columns
new_data = np.array([
    17,     # age
    2,      # studytime
    0,      # failures
    3,      # freetime
    4,      # goout
    4,      # health
    5,      # absences
    80,     # G1
    75,     # G2
    # --- Encoded categorical columns ---
    1,  # address_U
    0,  # famsize_GT3
    0, 0,  # reason_course, reason_home
    1,  # schoolsup_yes
    0,  # famsup_yes
    1,  # paid_yes
    1,  # activities_yes
    1,  # higher_yes
    1,  # internet_yes
    0   # romantic_yes
]).reshape(1, -1)

# Ensure new_data matches feature count
if new_data.shape[1] != X.shape[1]:
    raise ValueError(f"new_data has {new_data.shape[1]} features, expected {X.shape[1]}")

# Apply same normalization as training data
new_data_scaled = scaler.transform(new_data)

predicted_grade = model.predict(new_data_scaled)
print(f"Predicted grade for new student: {predicted_grade[0]:.2f}")
