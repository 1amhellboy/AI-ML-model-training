import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the Dataset
# Step1. Data Collection — you bring raw data into your pipeline.
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
print("First 5 rows of original dataset:\n", df.head())


# Drop Unneeded Columns
# Step2. Feature Selection — you keep only useful columns for your analysis
df.drop(['Pclass', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
print("\nRemaining columns after dropping:\n", df.columns.tolist())


# Check Missing Values & Duplicates
# Step3. Data Quality Check — knowing where your data is incomplete or repeated.
print("\nMissing values in each column:")
print(df.isnull().sum())

print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()


# Handle Missing Values
# Step4. Data Cleaning — filling gaps so models can use all rows.
df['Age'].fillna(df['Age'].mean(), inplace=True)


# Encode Categorical Variables
# Step5. Encoding — turning words into numbers for ML models.
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# Scale Numeric Values
# Step6. Normalization — putting all numeric values on the same scale so no feature dominates.
scaler = StandardScaler()
df[['Age']] = scaler.fit_transform(df[['Age']])

# Create New Feature
# Step7. Feature Engineering — creating a new column (AgeGroup) from existing data to help find patterns.
df['AgeGroup'] = pd.cut(df['Age'], bins=[-np.inf, -1, 0.0, 0.8, 1.6, 3],
labels=['Error', 'Child', 'Teen', 'Adult', 'Senior'])

print("\nFinal dataset preview:\n", df.head())

# Visulization
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

