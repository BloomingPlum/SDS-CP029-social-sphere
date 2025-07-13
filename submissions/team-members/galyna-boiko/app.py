import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

###############################################################################
# Configuration
###############################################################################

DATA_PATH = "data.csv"          # <- your uploaded dataset (same folder as app)
MODEL_PATH = "addiction_xgb.joblib"  # <- trained binary XGBoost model
TARGET_COL = "Addicted_Score"       # numeric 0‑9, we binarise to Low/High

# Threshold to split Low vs High addiction (change if you used something else)
THRESHOLD = 6  # 0‑6 == Low, 7‑9 == High

###############################################################################
# Load data and derive feature lists
###############################################################################

df = pd.read_csv(DATA_PATH)

numeric_cols = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Conflicts_Over_Social_Media",
]

categoric_cols = [
    "Gender",
    "Academic_Level",
    "Country",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Relationship_Status",
]

input_cols = numeric_cols + categoric_cols

###############################################################################
# Helper functions
###############################################################################

def preprocess(user_input: pd.DataFrame) -> pd.DataFrame:
    """Convert categories to category dtype so they match the model pipeline."""
    for col in categoric_cols:
        user_input[col] = user_input[col].astype("category")
    return user_input[input_cols]


def predict_addiction(user_input: pd.DataFrame):
    """Return the string label (Low/High) predicted by the model."""
    model = joblib.load(MODEL_PATH)
    proba_high = model.predict_proba(user_input)[0, 1]
    label = "High" if proba_high >= 0.5 else "Low"
    return label, proba_high

###############################################################################
# Streamlit UI
###############################################################################

st.title("Social‑Media Addiction Classifier")
st.markdown(
    "This app predicts whether a student is **Low** or **High** on the "
    "social‑media *Addicted Score* scale based on survey answers."
)

# Show basic dataset information
with st.expander("Dataset summary"):
    st.write("Total records:", len(df))
    st.write("Addicted_Score distribution:")
    st.bar_chart(df[TARGET_COL].value_counts().sort_index())

    st.write("Average numeric features by addiction level:")
    temp = df.assign(label=lambda x: np.where(x[TARGET_COL] >= THRESHOLD, "High", "Low"))
    st.dataframe(temp.groupby("label")[numeric_cols].mean().round(2))

st.header("Enter survey responses")

with st.form("user_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        age = st.slider("Age", 15, 40, 20)
        usage = st.slider("Avg daily social‑media use (hours)", 0.0, 12.0, 4.0, 0.1)
        sleep = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
        mental = st.slider("Mental‑health self‑score (1‑10)", 1, 10, 6)
        conflicts = st.slider("Conflicts over social media (count)", 0, 10, 2)

    with col_right:
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        acad_level = st.selectbox("Academic level", sorted(df["Academic_Level"].dropna().unique()))
        country = st.selectbox("Country", sorted(df["Country"].dropna().unique()))
        platform = st.selectbox("Most‑used platform", sorted(df["Most_Used_Platform"].dropna().unique()))
        affects = st.selectbox("Affects academic performance?", sorted(df["Affects_Academic_Performance"].dropna().unique()))
        relation = st.selectbox("Relationship status", sorted(df["Relationship_Status"].dropna().unique()))

    submitted = st.form_submit_button("Predict addiction level")

if submitted:
    user_df = pd.DataFrame(
        {
            "Age": [age],
            "Avg_Daily_Usage_Hours": [usage],
            "Sleep_Hours_Per_Night": [sleep],
            "Mental_Health_Score": [mental],
            "Conflicts_Over_Social_Media": [conflicts],
            "Gender": [gender],
            "Academic_Level": [acad_level],
            "Country": [country],
            "Most_Used_Platform": [platform],
            "Affects_Academic_Performance": [affects],
            "Relationship_Status": [relation],
        }
    )

    processed = preprocess(user_df)
    label, proba_high = predict_addiction(processed)

    st.markdown(
        f"### Predicted addiction level: **{label}**  \n"
        f"Probability of *High* addiction: **{proba_high:.2%}**"
    )
