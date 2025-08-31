# Lab 4 – Linear Regression with scikit-learn  
**Course**: Artificial Intelligence and Machine Learning (CSET301)  
**Objective**: Build, train, and evaluate a simple **Linear Regression** model on the **California Housing** dataset using Python, Pandas, Matplotlib/Seaborn, and scikit-learn.

---

## 📌 Problem Statement
Predict **Median House Value** in California based on selected numerical features such as **Average Rooms**, **Median Income**, **House Age**, and **Average Occupancy**.  
We will:
- Load the dataset directly from scikit-learn.
- Perform basic EDA (Exploratory Data Analysis).
- Select relevant features.
- Train and evaluate a Linear Regression model.
- Visualize predictions vs actual data.

---

## 🛠 Tools and Libraries Used
- **Python 3.x**
- **pandas** – Data manipulation
- **numpy** – Numerical computations
- **matplotlib / seaborn** – Data visualization
- **scikit-learn** – Dataset, ML model, metrics

---

## 📂 Code Flow

### 1️⃣ Import Dependencies
Load all required libraries for:
- Data processing (`pandas`, `numpy`)
- Visualization (`matplotlib`, `seaborn`)
- ML tasks (`sklearn.datasets`, `train_test_split`, `LinearRegression`, `metrics`)

---

### 2️⃣ Load Dataset
- Use `fetch_california_housing()` from scikit-learn to get the **California Housing dataset**.
- Data is returned as a **Bunch** (dictionary-like object).
  
---

### 3️⃣ Convert to DataFrame
- Create a Pandas DataFrame using `pd.DataFrame()`.
- Assign column names from `california.feature_names`.
- Add target variable `MedHouseVal` to the DataFrame.

---

### 4️⃣ Exploratory Data Analysis (EDA)
- Show the first 5 rows with `.head()` to check structure.
- Use `.describe()` for summary statistics.
- **Histograms** to observe distribution of target values.
- **Correlation heatmap** (seaborn `heatmap()`) to see relationships between features and target.

---

### 5️⃣ Feature Selection
- From the dataset, select relevant numeric features expected to influence house prices:
- Set:
- `X` = Features
- `y` = Target (`MedHouseVal`)

---

### 6️⃣ Split Dataset
- Use `train_test_split()` with:
- **80%** → Training set
- **20%** → Testing set
- `random_state=42` for reproducibility

---

### 7️⃣ Initialize and Train the Model
- Create a `LinearRegression()` object.
- Fit the model on training data: `model.fit(X_train, y_train)`.

---

### 8️⃣ Make Predictions
- Use `model.predict(X_test)` to get predicted house values for the test set.

---

### 9️⃣ Model Evaluation
- **R² Score**: Measure how well the model explains variance in the target.
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values.

---

### 🔟 Visualization: Predictions vs Actual Values
- Scatter plot with actual values on X-axis and predicted values on Y-axis.
- Add diagonal reference line (`y = x`) for perfect predictions.

---

### 1️⃣1️⃣ Interpretation
- Summarize:
- Goodness of fit using R².
- Average prediction error using MSE.
- Limitations: linear assumptions, sensitivity to outliers, possible feature correlations.

---

## 📊 Output Example
**Metrics Example Output:**
**Visuals:**
- Distribution graph of house values.
- Correlation heatmap.
- Scatter plot (Predicted vs Actual).
