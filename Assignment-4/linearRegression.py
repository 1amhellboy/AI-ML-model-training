import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the California Housing Dataset


# fetch_california_housing() downloads the dataset and returns it as a Bunch object (like a dictionary)
california = fetch_california_housing()


# Convert Dataset to DataFrame


# Create a pandas DataFrame from the data and use feature names for columns
df = pd.DataFrame(california.data, columns=california.feature_names)

# Add the target (median house value) as a new column
df['MedHouseVal'] = california.target


# Explore the Dataset (EDA)

# Show the first 5 rows to understand the structure
print("First 5 rows of dataset:")
print(df.head())

# Basic statistics (mean, std, min, max) for numerical columns
print("\nDataset statistics:")
print(df.describe())

# Plot histogram for median house value
plt.figure(figsize=(6,4))
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title('Distribution of Median House Value')
plt.show()

# Correlation heatmap to see relationships
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Feature Selection

# Select relevant numerical features
features = ['AveRooms', 'MedInc', 'HouseAge', 'AveOccup']

X = df[features]     # (input features)
y = df['MedHouseVal'] # (output)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Normalize Features

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and Train Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)


# Predictions on Test Data


# Get predicted values for test set
y_pred = model.predict(X_test)


# Evaluate the Model


# Calculate R² score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")



# 10. Scatter Plot (Predictions vs Actual)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs. Predicted Values')
# Draw ideal line where prediction = actual
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.show()


#  Interpret Results


print("\nModel Interpretation:")
print("The R² score tells us how much variance in house price is explained by our model.")
print("If R² is close to 1, predictions are good. Here, it shows moderate performance.")
print("MSE shows the average squared prediction error — lower is better.")
print("Limitations: This model assumes linear relationships and may not handle outliers or complex patterns well.")
