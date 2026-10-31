"""Streamlit app: Heart Disease Risk Explorer.

Run with:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src import features as F

st.set_page_config(page_title="Heart Disease Risk Explorer", page_icon="🫀", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(F.MODEL_PATH)


@st.cache_data
def load_results():
    return json.loads(F.RESULTS_PATH.read_text())


if not (F.MODEL_PATH.exists() and F.RESULTS_PATH.exists()):
    st.error("No trained model found. Run `python train.py` first, then restart the app.")
    st.stop()

model, results = load_model(), load_results()

st.title("Heart Disease Risk Explorer")
st.warning(
    "Educational demo only. This is **not** a medical device and must not be used "
    "for diagnosis or treatment decisions. Please consult a healthcare professional."
)

# ---------- Sidebar: patient inputs ----------
st.sidebar.header("Patient details")


def choice(label, key):
    options = F.CATEGORY_LABELS[key]
    return st.sidebar.selectbox(label, list(options), format_func=options.get)


age = st.sidebar.slider("Age (years)", 18, 100, 54)
sex = choice("Sex", "Sex")
chest_pain = choice("Chest pain type", "ChestPainType")
resting_bp = st.sidebar.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
cholesterol = st.sidebar.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
fasting_bs = st.sidebar.checkbox("Fasting blood sugar > 120 mg/dl")
resting_ecg = choice("Resting ECG", "RestingECG")
max_hr = st.sidebar.slider("Maximum heart rate achieved", 60, 202, 140)
exercise_angina = choice("Exercise-induced angina", "ExerciseAngina")
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", -2.5, 6.5, 1.0, step=0.1)
st_slope = choice("ST slope", "ST_Slope")

patient = pd.DataFrame(
    [{
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": int(fasting_bs), "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }]
)[F.FEATURES]

tab_pred, tab_models, tab_about = st.tabs(["Prediction", "Model comparison", "About"])

# ---------- Tab 1: prediction ----------
with tab_pred:
    prob = float(model.predict_proba(patient)[0, 1])
    col1, col2 = st.columns([1, 2])
    col1.metric("Estimated probability of heart disease", f"{prob:.0%}")
    col1.progress(prob)
    if prob >= 0.5:
        col2.error("The model classifies this profile as **higher risk** (probability ≥ 50%).")
    else:
        col2.success("The model classifies this profile as **lower risk** (probability < 50%).")
    col2.caption(
        f"Prediction made by the **{results['winner']}** model, which had the best "
        "cross-validation accuracy. Adjust the inputs in the sidebar to see how the estimate changes."
    )
    with st.expander("Show the input row sent to the model"):
        st.dataframe(patient, hide_index=True)

# ---------- Tab 2: model comparison ----------
with tab_models:
    st.subheader("Decision Tree vs Random Forest vs XGBoost")
    rows = [
        {
            "Model": name,
            "CV accuracy": m["cv_accuracy"],
            "Train accuracy": m["train_accuracy"],
            "Test accuracy": m["test_accuracy"],
            "Test recall": m["test_recall"],
            "Test F1": m["test_f1"],
            "Test ROC-AUC": m["test_roc_auc"],
        }
        for name, m in results["models"].items()
    ]
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table)
    st.caption(
        f"Hyperparameters tuned with 5-fold cross-validation on {results['n_train']} training rows; "
        f"metrics above are on {results['n_test']} held-out test rows. With a test set this small, "
        "differences of a few percentage points between models are within noise."
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Test accuracy**")
    c1.bar_chart(table["Test accuracy"])
    c2.markdown(f"**Top features ({results['winner']})**")
    top = pd.DataFrame(results["top_features"]).set_index("feature")
    c2.bar_chart(top["importance"])
    with st.expander("Best hyperparameters"):
        st.json({n: m["best_params"] for n, m in results["models"].items()})

# ---------- Tab 3: about ----------
with tab_about:
    st.markdown(
        """
**Data.** 918 patient records with 11 clinical features, from the *Heart Failure Prediction Dataset*
by fedesoriano (Kaggle, 2021), itself a merge of five public cardiology datasets from the UCI repository.

**Data cleaning.** 172 rows have a cholesterol value of 0 and one row has a resting blood pressure of 0.
These are impossible values that stand in for "not measured", so they are treated as missing and
filled with the training-set median inside the model pipeline.

**Method.** One-hot encoding for categorical features, then three tree-based models tuned with
grid search and 5-fold cross-validation. The best model is picked by cross-validation score,
so the test set stays untouched until the final evaluation.

**Limitations.** Small dataset, patients from a handful of clinics, and no external validation.

*Built with concepts from the Machine Learning Specialization (DeepLearning.AI / Coursera, Stanford Online),
re-implemented independently. Not affiliated with or endorsed by the course providers.*
"""
    )
