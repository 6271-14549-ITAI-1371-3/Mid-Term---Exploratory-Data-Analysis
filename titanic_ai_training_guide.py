"""Titanic AI/ML Training Guide

This single script combines the main lessons from the example labs:
1. Python/data-science environment setup
2. Loading and inspecting data
3. Exploratory data analysis (EDA)
4. Cleaning missing values
5. Encoding categorical features
6. Splitting data correctly
7. Training and comparing classification models
8. Evaluating and interpreting model results

The target is `Survived`:
    0 = passenger did not survive
    1 = passenger survived

You can run this file as a normal Python script or paste its sections into
Jupyter/Google Colab cells. Install dependencies first if needed:
    pip install pandas numpy matplotlib seaborn scikit-learn
"""

# -----------------------------------------------------------------------------
# 1. IMPORT LIBRARIES AND CONFIGURE THE ENVIRONMENT
# -----------------------------------------------------------------------------
# Pandas: tables and data cleaning
# NumPy: numerical operations
# Matplotlib/Seaborn: charts
# scikit-learn: splitting data, preprocessing, models, and evaluation
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# 2. LOAD THE TITANIC DATASET
# -----------------------------------------------------------------------------
# Reading from a raw GitHub URL makes this project reproducible: anyone with
# internet access can run the same code without manually downloading a file.
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print("\nFirst five rows:")
print(df.head())

# -----------------------------------------------------------------------------
# 3. UNDERSTAND THE DATA BEFORE MODELING
# -----------------------------------------------------------------------------
# `info()` shows data types and non-null counts. Missing values matter because
# most machine-learning algorithms cannot train on NaN values.
print("\nData types and non-null counts:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").transpose())

print("\nMissing values by column:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nUnique values in categorical columns:")
for column in ["Sex", "Embarked"]:
    print(f"{column}: {df[column].dropna().unique().tolist()}")

# A target distribution check tells us whether the classes are balanced.
# If one class is much larger, accuracy alone can be misleading.
print("\nSurvival counts:")
print(df["Survived"].value_counts().rename(index={0: "Did not survive", 1: "Survived"}))
print(f"Overall survival rate: {df['Survived'].mean():.1%}")

# -----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
# EDA is the process of looking for distributions, patterns, relationships,
# unusual values, and data-quality problems before selecting a model.
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot(data=df, x="Survived", ax=axes[0, 0])
axes[0, 0].set_title("Target distribution")
axes[0, 0].set_xticklabels(["Did not survive", "Survived"])

sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 1])
axes[0, 1].set_title("Survival by passenger class")

sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0, 2])
axes[0, 2].set_title("Survival by sex")

sns.histplot(data=df, x="Age", hue="Survived", bins=25, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Age distributions by survival")

sns.boxplot(data=df, x="Survived", y="Fare", ax=axes[1, 1])
axes[1, 1].set_title("Fare distribution by survival")

survival_by_class = df.groupby("Pclass", observed=True)["Survived"].mean()
axes[1, 2].bar(survival_by_class.index.astype(str), survival_by_class.values)
axes[1, 2].set_title("Survival rate by class")
axes[1, 2].set_xlabel("Passenger class")
axes[1, 2].set_ylabel("Survival rate")
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Grouped survival rates are often easier to interpret than raw counts.
print("\nSurvival rate by sex:")
print(df.groupby("Sex", observed=True)["Survived"].mean().sort_values(ascending=False))
print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass", observed=True)["Survived"].mean().sort_values(ascending=False))

# Correlation applies to numeric columns only. It can suggest relationships,
# but correlation is not proof of causation.
plt.figure(figsize=(10, 7))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Numeric feature correlation matrix")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5. CHOOSE FEATURES AND TARGET
# -----------------------------------------------------------------------------
# We intentionally leave out columns that are identifiers or overly specific:
# - PassengerId is just an ID, not a meaningful passenger characteristic.
# - Name and Ticket require extra text/feature engineering.
# - Cabin is mostly missing in this version of the dataset.
# We keep a compact set of interpretable features for a learning project.
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target_column = "Survived"

X = df[feature_columns]
y = df[target_column]

print("\nSelected features:")
print(feature_columns)
print(f"X shape: {X.shape} | y shape: {y.shape}")

# -----------------------------------------------------------------------------
# 6. SPLIT INTO TRAINING AND TESTING DATA
# -----------------------------------------------------------------------------
# The test set acts like new, unseen data. `stratify=y` preserves the same
# proportion of survivors/non-survivors in both sets. The split happens before
# learning imputation or scaling values to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nData split:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print(f"Training survival rate: {y_train.mean():.1%}")
print(f"Testing survival rate:  {y_test.mean():.1%}")

# -----------------------------------------------------------------------------
# 7. BUILD REUSABLE PREPROCESSING PIPELINES
# -----------------------------------------------------------------------------
# Numerical columns: fill missing values with the median, then standardize.
# Median is robust when values are skewed or contain outliers.
numeric_features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical columns: fill missing values with the most frequent category,
# then one-hot encode them. `handle_unknown='ignore'` prevents errors if a
# category appears in future data but was absent during training.
categorical_features = ["Sex", "Embarked"]
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# -----------------------------------------------------------------------------
# 8. TRAIN TWO CLASSIFICATION MODELS
# -----------------------------------------------------------------------------
# A pipeline guarantees that exactly the same cleaning and transformations are
# applied during training and prediction. This is safer than manually cleaning
# training and test data in separate, possibly inconsistent steps.
models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    ),
    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
        ]
    ),
}

results = []
trained_models = {}

for model_name, model_pipeline in models.items():
    print(f"\nTraining {model_name}...")
    model_pipeline.fit(X_train, y_train)
    trained_models[model_name] = model_pipeline

    y_pred = model_pipeline.predict(X_test)
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
        }
    )

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False, formatters={
    "Accuracy": "{:.3f}".format,
    "Precision": "{:.3f}".format,
    "Recall": "{:.3f}".format,
    "F1": "{:.3f}".format,
}))

# -----------------------------------------------------------------------------
# 9. EVALUATE THE BEST MODEL IN MORE DETAIL
# -----------------------------------------------------------------------------
# F1 balances precision and recall. For a binary target:
# - Precision: of people predicted to survive, how many actually survived?
# - Recall: of people who survived, how many did the model find?
# - F1: a combined score that balances precision and recall.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

print(f"\nDetailed evaluation: {best_model_name}")
print(classification_report(y_test, best_predictions, target_names=["Did not survive", "Survived"]))

cm = confusion_matrix(y_test, best_predictions)
print("Confusion matrix [rows = actual, columns = predicted]:")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    display_labels=["Did not survive", "Survived"],
    cmap="Blues",
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 10. INTERPRET THE LOGISTIC REGRESSION MODEL
# -----------------------------------------------------------------------------
# Coefficients show how each transformed feature contributes to the prediction.
# Positive coefficients push predictions toward survival; negative coefficients
# push them toward non-survival. This interpretation applies to logistic
# regression, not automatically to every model.
logistic = trained_models["Logistic Regression"]
feature_names = logistic.named_steps["preprocessor"].get_feature_names_out()
coefficients = logistic.named_steps["model"].coef_[0]
coefficient_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
}).sort_values("Coefficient")

print("\nLogistic regression coefficients:")
print(coefficient_table.to_string(index=False))

plt.figure(figsize=(10, 6))
colors = ["crimson" if value < 0 else "seagreen" for value in coefficient_table["Coefficient"]]
plt.barh(coefficient_table["Feature"], coefficient_table["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient (direction and strength of association)")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 11. MAKE A PREDICTION FOR A NEW PASSENGER
# -----------------------------------------------------------------------------
# A new passenger must be supplied with the same raw feature columns used by
# X. The pipeline then performs imputation, encoding, scaling, and prediction.
new_passenger = pd.DataFrame([
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C",
    }
])

prediction = best_model.predict(new_passenger)[0]
probability = best_model.predict_proba(new_passenger)[0, 1]
print("\nExample prediction:")
print(new_passenger.to_string(index=False))
print(f"Predicted outcome: {'Survived' if prediction == 1 else 'Did not survive'}")
print(f"Estimated probability of survival: {probability:.1%}")

# -----------------------------------------------------------------------------
# 12. KEY LESSONS AND LIMITATIONS
# -----------------------------------------------------------------------------
print("""
Key lessons:
- Explore the data before modeling; charts can reveal patterns that averages hide.
- Keep the target (`Survived`) separate from the input features.
- Split before fitting preprocessing steps to prevent data leakage.
- Use imputation for missing values and one-hot encoding for categories.
- Use a Pipeline so training and future predictions follow the same process.
- Compare more than one metric; accuracy alone may hide class-specific errors.
- A model learns associations in historical data. It does not prove that a
  feature caused survival, and it should not be treated as a real-world causal
  explanation or a guarantee for an individual person.
""")
