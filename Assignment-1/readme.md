# Data Preprocessing 

Preprocessing means getting your raw data ready so that a machine learning model (or any data analysis) can actually understand and use it.

Raw data often has problems — it can be messy, incomplete, or in the wrong format.Preprocessing is like cleaning, organizing, and transforming your data before feeding it into your model.


# Common Steps in Preprocessing 

Cleaning – Removing or fixing bad data
Example: Handling missing values, removing duplicates, correcting spelling errors.

Normalization/Scaling – Bringing all numbers to a similar range
Example: Converting ages like 5, 25, 70 into 0.07, 0.35, 1.0 so large numbers don’t dominate the model.

Encoding Categorical Data – Converting words into numbers
Example: Turning ["Red", "Green", "Blue"] into [0, 1, 2] or one-hot encoding.

Splitting Data – Dividing into training and testing sets
Example: 80% of data to train the model, 20% to test it.

# Analogy

Think of preprocessing like washing and chopping vegetables before cooking:
Raw vegetables (data) are dirty, uneven, and full of inedible parts.
You wash, peel, cut, and organize them so the chef (your ML model) can cook a perfect dish.
If you skip this step, your final meal (model performance) will suffer.

# Assignment 1

The goal is to practice data preprocessing using the Titanic dataset with pandas, numpy, and sklearn.
You’ll need to:

-Load the dataset from the given URL.
-Handle missing values (fill or drop them appropriately).
-Deal with duplicate rows.
-Convert categorical columns (like Sex and Embarked) into numerical form using label encoding.
-Normalize numerical features (like Age and Fare).
-Sort and filter data (e.g., by Fare or Age).
-Engineer a new feature (like an AgeGroup column).
-(Optional) Create visualizations (e.g., survival count plot, fare distribution histogram).


## 📌 Quick Flow of Lab 1
1. **Load Data** → Get Titanic CSV into pandas DataFrame.
2. **Inspect Data** → Check missing values & duplicates.
3. **Clean Data** → Fill missing values, remove duplicates.
4. **Select Features** → Drop unused columns.
5. **Encode Categories** → Convert text to numbers with `LabelEncoder`.
6. **Scale Numbers** → Use `StandardScaler` to normalize.
7. **Engineer Features** → Create `AgeGroup` with `pd.cut()`.
8. **Visualize** → Plot survival count with seaborn.



## 🛠 Tools & Memory Tricks

| Tool | One-Liner Definition | Memory Trick |
|------|----------------------|--------------|
| **pandas (`pd`)** | Library to handle **tables** of data (DataFrames). | Think: **Excel in Python** |
| **numpy (`np`)** | Handles fast **math & arrays**. | Think: **Math engine behind pandas** |
| **LabelEncoder** | Turns **categories → numbers**. | Think: *Name tags become ID numbers* |
| **StandardScaler** | Rescales numbers → mean 0, std 1. | Think: **Put all players at same height** before the match |
| **matplotlib (`plt`)** | Creates **basic plots & graphs**. | Think: *Draw anything from scratch* |
| **seaborn (`sns`)** | Creates **beautiful, easy graphs** built on matplotlib. | Think: *Matplotlib but with makeup* |
| **`np.inf`** | Infinity value for ranges. | Think: **Never-ending number** |
| **`pd.cut()`** | Turns continuous numbers into **categories**. | Think: **Put ages into buckets** (“child”, “adult”) |
| **`.fillna()`** | Fills missing values with a chosen number. | Think: **Patch the holes** in data |
| **`.drop_duplicates()`** | Removes repeated rows. | Think: **Delete photocopies** |
