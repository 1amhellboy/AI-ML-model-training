# Step 0: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd

# Step 1: Load Dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Step 2: Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Step 3: Setup KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store metrics for averaging
accuracies, precisions, recalls, f1s = [], [], [], []

# Step 4: Cross-Validation Loop
fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    print(f"\nFold {fold} Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Step 6: Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.show()

    # Step 7: Class Distribution Visualization
    unique, counts = np.unique(y_test, return_counts=True)
    sns.barplot(x=[class_names[u] for u in unique], y=counts)
    plt.title(f"Class Distribution in Fold {fold} Test Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    fold += 1

# Step 5: Print Average Performance Across Folds
print("\nAverage Performance over 5 Folds:")
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f}")

# Step 8 (Optional): Learning Curve Visualization
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Decision Tree")
plt.legend()
plt.show()