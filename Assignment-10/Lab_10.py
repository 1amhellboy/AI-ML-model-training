# lab10_random_forest.py
# Step-by-step code for Lab 10 (Random Forest on Wine Quality)
# Requirements: pandas, numpy, scikit-learn, matplotlib
# Run: python lab10_random_forest.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns  # optional for nicer confusion matrix plotting

# -------------------------
# 1. Read the dataset
# -------------------------
# Using the UCI red wine CSV (semicolon-separated)
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url_red, sep=';')

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------
# 2. Extract independent & dependent variables
# -------------------------
X = df.drop(columns=['quality'])
y = df['quality']  # numeric quality (0-10 in general; in practice values like 3-8)

# -------------------------
# 3. Convert quality to 3 categories: 'poor', 'average', 'best'
#    (one reasonable mapping: <=4 poor, 5-6 average, >=7 best)
# -------------------------
def quality_to_label(q):
    if q <= 4:
        return 'poor'
    elif q <= 6:
        return 'average'
    else:
        return 'best'

y_labels = y.apply(quality_to_label)
print("\nClass distribution (labels):")
print(y_labels.value_counts())

# Encode labels to integers for sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y_labels)  # 0,1,2 with le.classes_ telling mapping

print("Label encodings:", dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------
# 4. Split dataset into train/test (75-25)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------
# 5. Perform normalization on numerical features
#    (StandardScaler used here; fit on train, transform train+test)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Note: Tree models don't require scaling, but lab asks to normalize.

# -------------------------
# 6. Build Random Forest (sklearn) with default parameters (plus random_state for reproducibility)
# -------------------------
rf = RandomForestClassifier(random_state=42)  # default n_estimators usually 100
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# -------------------------
# 7. Confusion matrix
# -------------------------
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# 8. Accuracy, Precision, Recall, F1-score
# -------------------------
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf_macro = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
rec_rf_macro  = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
f1_rf_macro   = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)

print(f"Random Forest — Accuracy: {acc_rf:.4f}, Precision(macro): {prec_rf_macro:.4f}, Recall(macro): {rec_rf_macro:.4f}, F1(macro): {f1_rf_macro:.4f}")

# -------------------------
# 9. Compare with Decision Tree (default)
# -------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6,4))
sns.heatmap(cm_dt, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt_macro = precision_score(y_test, y_pred_dt, average='macro', zero_division=0)
rec_dt_macro  = recall_score(y_test, y_pred_dt, average='macro', zero_division=0)
f1_dt_macro   = f1_score(y_test, y_pred_dt, average='macro', zero_division=0)

print(f"Decision Tree — Accuracy: {acc_dt:.4f}, Precision(macro): {prec_dt_macro:.4f}, Recall(macro): {rec_dt_macro:.4f}, F1(macro): {f1_dt_macro:.4f}")

# -------------------------
# Extra: Feature importances from Random Forest
# -------------------------
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop feature importances (Random Forest):")
print(feat_imp.head(10))

# -------------------------
# Save models or results if needed (optional)
# -------------------------
# from joblib import dump
# dump(rf, "rf_wine_model.joblib")
