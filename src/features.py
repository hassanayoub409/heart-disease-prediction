"""Data loading and preprocessing shared by the training script and the app."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_PATH = ROOT / "models" / "results.json"

TARGET = "HeartDisease"
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + CATEGORICAL

# Allowed values for the categorical inputs (used to build the app's dropdowns)
CATEGORY_LABELS = {
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {
        "ASY": "Asymptomatic",
        "ATA": "Atypical angina",
        "NAP": "Non-anginal pain",
        "TA": "Typical angina",
    },
    "RestingECG": {
        "Normal": "Normal",
        "ST": "ST-T wave abnormality",
        "LVH": "Left ventricular hypertrophy",
    },
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn physiologically impossible zeros into missing values.

    In this dataset, Cholesterol == 0 (172 rows) and RestingBP == 0 (1 row) are
    placeholders for "not measured", not real readings. They are marked as NaN
    here and imputed later *inside* the model pipeline, so the imputation values
    are learned from training data only.
    """
    df = df.copy()
    df[["Cholesterol", "RestingBP"]] = df[["Cholesterol", "RestingBP"]].replace(0, np.nan)
    return df


def load_data() -> pd.DataFrame:
    return clean(pd.read_csv(DATA_PATH))


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
