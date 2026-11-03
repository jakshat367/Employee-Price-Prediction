import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# Load Dataset
data = pd.read_csv("salary_data.csv")


# Input Features
X = data[
    [
        "experience",
        "education",
        "job_role",
        "location",
        "company_size"
    ]
]


# Target
y = data["salary"]


# Categorical Columns
categorical_columns = [
    "education",
    "job_role",
    "location",
    "company_size"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Machine Learning Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train Model
model.fit(X_train, y_train)


# Prediction
predictions = model.predict(X_test)


# Model Evaluation
print("Model Training Completed!")

print("R2 Score:", r2_score(y_test, predictions))

print(
    "Mean Absolute Error:",
    mean_absolute_error(y_test, predictions)
)


# Save Model
os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/salary_model.pkl"
)

print("Model Saved Successfully!")