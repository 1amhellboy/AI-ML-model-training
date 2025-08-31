import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Task 1: Load Data and Print First 5 Rows

# Load Titanic dataset (downloaded CSV from Kaggle)
data = pd.read_csv("titanic.csv")

# Show first 5 rows
print(data.head())


# Task 2: Explore and Visulize based on

# 1. Total no. of people survived vs not survived

sns.countplot(x="Survived",data=data)
plt.title("Total Number of People Survived vs Not Survived")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("count")
plt.show()


# 2. Survival rated based on gender

sns.countplot(x="Sex", hue="Survived", data=data)
plt.title("Survival Rate Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# 3. Survival rate based on age categories

# age bins(categories)
bins = [0, 12, 18, 35, 50, 80] 
labels = ['Child', 'Teen', 'Young Adult', 'Middle Age', 'Senior']
# converting a continuous variable (Age) into categorical groups (AgeGroup). It’s called binning or discretization.
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, include_lowest=True)

sns.countplot(x="AgeGroup", hue="Survived", data=data, order=labels)
plt.title("Survival Rate Based on Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

# Task 3: Data Preprocessing


# 1. Drop irrelevent columns
# PassengerId, Name, Ticket, Cabin don’t help survival prediction
data = data.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

# 2. Handle missing values
# Fill Age with median, Embarked with mode (most frequent value)
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# 3. Encode categorical columns
# 'Sex' and 'Embarked' are categorical -> convet convert them into numbers
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"]) # male=1, female=0
data["Embarked"] = le.fit_transform(data["Embarked"]) # S, C, Q -> 0,1,2

# After fixing Age, recreate AgeGroup to remove NaNs
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, include_lowest=True)

# Check cleaned dataset
print(data.head())
print(data.isnull().sum()) # Verify no missing value


# Task 4: Define X (features) and y (target)

y = data["Survived"] # Define target variable
x = data.drop("Survived",axis=1) # Define feature matrix

# Convert categorical column 'AgeGroup' into dummy variables (one-hot encoding):It converts each category into a new binary column
x = pd.get_dummies(x,columns=["AgeGroup"],drop_first=True)

print("Features (X):")
print(x.head(), "\n")

print("Features (Y):")
print(y.head(),"\n")

# Task 5: Split into training and testing subsets

# Split dataset: 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)


# Task 6: Train Logistic Regression Model

# Create model (use solver='liblinear' for small datasets like Titanic)
log_reg = LogisticRegression(solver='liblinear', random_state=42)

# Train the model on training data
log_reg.fit(x_train, y_train)

# Print model coefficients (importance of each feature)
print("Intercept (bias):", log_reg.intercept_)
print("Coefficient (weights for each feature):")
for col, coef in zip(x_train.columns, log_reg.coef_[0]):
    print(f"{col}: {coef:.4f}")



# Task 7: Evaluate the Model

# Predictions on training and testing data
y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)

# --- Training set metrics ---
print("Training Set Metrics:")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall   :", recall_score(y_train, y_train_pred))
print("F1 Score :", f1_score(y_train, y_train_pred))
print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, y_train_pred))
print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred))

# --- Testing set metrics ---
print("\nTesting Set Metrics:")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

# --- Confusion Matrix for Training Set ---
cm_train = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.title("Confusion Matrix - Training Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Confusion Matrix for Testing Set ---
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.title("Confusion Matrix - Testing Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()