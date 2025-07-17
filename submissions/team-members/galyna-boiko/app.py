import streamlit as st
import pandas as pd
import joblib
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "addiction_xgb.joblib")
TARGET_COL = 'Conflicts_Over_Social_Media_Binary'

# Load data
df = pd.read_csv(DATA_PATH)

# Create derived target column
df[TARGET_COL] = df["Conflicts_Over_Social_Media"].apply(lambda x: 0 if x <= 2 else 1).astype(int)

# Define columns to exclude from input
exclude_cols = ['Student_ID', 'Conflicts_Over_Social_Media', TARGET_COL, 'Addicted_Score']

# Identify numeric and categoric columns
numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols).tolist()
categoric_cols = df.select_dtypes(include='object').columns.difference(exclude_cols).tolist()

# Extract categories from real data
sample_categories = {
    col: sorted(df[col].dropna().unique().tolist()) for col in categoric_cols
}

# Load model
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.title("📱 Predict Conflicts Over Social Media")
st.write("Fill in the following details to see if social media use may be causing interpersonal conflicts.")

user_data = {}

numeric_input_ranges = {
    'Age': {'min': 18, 'max': 24, 'step': 1.0},
    'Avg_Daily_Usage_Hours': {'min': 1.5, 'max': 8.5, 'step': 0.5},
    'Sleep_Hours_Per_Night': {'min': 3.5, 'max': 10, 'step': 0.5},
    'Mental_Health_Score': {'min': 4, 'max': 9, 'step': 1.0}
}


# Numeric input fields
for col in numeric_cols:
    params = numeric_input_ranges.get(col, {'min': 0.0, 'max': 100.0, 'step': 1.0})

    min_val = float(params['min'])
    max_val = float(params['max'])
    step = float(params['step'])

    # Default value: midpoint
    default_val = round((min_val + max_val) / 2, 2)

    user_data[col] = st.slider(
        f"{col}",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step
    )

# Categorical input fields
for col in categoric_cols:
    user_data[col] = st.selectbox(f"{col}", sample_categories[col])



# Prediction button
if st.button("Predict Conflict"):
    # ✅ Step 1: Create input dataframe
    input_df = pd.DataFrame([user_data])

    # ✅ Step 2: Convert categoricals to category dtype
    input_df[categoric_cols] = input_df[categoric_cols].astype("category")

    # ✅ Optional: Align with training category levels
    for col in categoric_cols:
        if col in df.columns:
            input_df[col] = input_df[col].astype(
                pd.CategoricalDtype(categories=df[col].dropna().unique())
            )

    # ✅ Step 3: Match column order to model
    model_features = model.get_booster().feature_names
    input_df = input_df[model_features]

    # ✅ Step 4: Debug – Show input to model
    st.subheader("🧾 Input to Model")
    st.write(input_df)

    # ✅ Step 5: Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ✅ Step 6: Show output
    label = "🚨 Conflicts are likely" if prediction == 1 else "✌️ Conflicts are unlikely"
    st.success(f"Prediction: {label}")
    st.info(f"Model confidence (probability of conflict): {prob:.2f}")
