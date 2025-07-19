import os, joblib, pandas as pd, streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.manifold import TSNE

st.set_page_config(layout="wide")

# ------------------------------------------------------------------#
# 1. Load artefacts                                                  #
# ------------------------------------------------------------------#
BASE_DIR = os.path.dirname(__file__)

# Load models and levels
MODEL_CONFLICTS   = joblib.load(os.path.join(BASE_DIR, "binary_conflicts_xgb_v2.joblib"))
MODEL_ADDICTION = joblib.load(os.path.join(BASE_DIR, "addiction_score_lin_reg_v2.joblib"))
LEVELS_CONFLICTS  = joblib.load(os.path.join(BASE_DIR, "category_levels_v2.joblib"))
FEATS_CONFLICTS   = MODEL_CONFLICTS.get_booster().feature_names

# ------------------------------------------------------------------#
# 2. Sidebar Navigation                                              #
# ------------------------------------------------------------------#
st.sidebar.image("image.png", use_container_width=True)

st.sidebar.markdown("""
## Social Media Usage Analysis

This application explores conflicts and addictive behavior related to social media usage.  
Select a model to see the prediction or cluster analysis results.
""")

page = st.sidebar.radio("Select Analysis Type", [
    "Conflict Prediction", 
    "Addiction Score Prediction", 
    "Clustering Analysis"   # <-- NEW
])

# ------------------------------------------------------------------#
# 3. Conflict Prediction Page                                       #
# ------------------------------------------------------------------#
if page == "Conflict Prediction":
    st.title("Conflict Prediction with XGBoost")

    st.markdown("""
    This tool estimates the likelihood of social media–related conflicts based on lifestyle and digital habits.

    It is designed for **young adults (18–24 years old)**, especially students. You can provide the details below to see the risk.
    """)

    # Define input feature groups
    CATEGORIC_COLS = list(LEVELS_CONFLICTS.keys())
    NUMERIC_COLS   = [c for c in FEATS_CONFLICTS if c not in CATEGORIC_COLS]

    # --- User input ------------------------------------------------
    user = {}
    user["Age"]                   = st.slider("Age", 18, 24, 20)
    user["Avg_Daily_Usage_Hours"] = st.slider("Average daily usage (hours)", 1.5, 8.5, 4.0, 0.5)
    user["Sleep_Hours_Per_Night"] = st.slider("Sleep hours per night", 3.5, 10.0, 7.0, 0.5)
    user["Mental_Health_Score"]   = st.slider("Mental health score", 4, 9, 5)

    for col in CATEGORIC_COLS:
        user[col] = st.selectbox(col.replace("_", " "), LEVELS_CONFLICTS[col])

    # --- Prediction ------------------------------------------------
    if st.button("Predict conflict likelihood"):
        X = pd.DataFrame([user])

        for col in CATEGORIC_COLS:
            X[col] = pd.Categorical(X[col], categories=LEVELS_CONFLICTS[col])

        X = X[FEATS_CONFLICTS]

        proba = float(MODEL_CONFLICTS.predict_proba(X)[0][1])
        pred  = int(proba >= 0.5)

        label = "🚨 Conflicts are likely" if pred else "✌️ Conflicts are unlikely"
        st.success(label)
        st.info(f"Model probability of conflict: **{proba:.2%}**")

# ------------------------------------------------------------------#
# 4. Placeholder for Other Pages                                     #
# ------------------------------------------------------------------#
elif page == "Addiction Score Prediction":
    st.title("Addiction Score Prediction with Linear Regression")

    st.markdown("""
    Estimate how addicted someone might be to social media based on their digital and lifestyle habits.
    """)

    # Reuse same input fields
    user = {}
    user["Age"]                   = st.slider("Age", 18, 24, 20)
    user["Avg_Daily_Usage_Hours"] = st.slider("Average daily usage (hours)", 1.5, 8.5, 4.0, 0.5)
    user["Sleep_Hours_Per_Night"] = st.slider("Sleep hours per night", 3.5, 10.0, 7.0, 0.5)
    user["Mental_Health_Score"]   = st.slider("Mental‑health score", 4, 9, 5)

    # These should match training set levels
    LEVELS_ADDICTION = LEVELS_CONFLICTS  # if shared, or load a separate one if different

    for col in LEVELS_ADDICTION:
        user[col] = st.selectbox(col.replace("_", " "), LEVELS_ADDICTION[col], key=f"addiction_{col}")

    

    if st.button("Predict addiction score"):
        X = pd.DataFrame([user])

        # Apply same categorical casting
        for col in LEVELS_ADDICTION:
            X[col] = pd.Categorical(X[col], categories=LEVELS_ADDICTION[col])

        # Make prediction
        predicted_score = MODEL_ADDICTION.predict(X)[0]
        rounded_score = round(predicted_score)

        st.success(f"Predicted addiction score: **{rounded_score}** (from 2 to 9)")
        st.info(f"Raw model output: {predicted_score:.2f}")

elif page == "Clustering Analysis":
    st.title("Clustering Analysis with KMeans + PCA")

    st.markdown("""
        Explore clusters based on digital and lifestyle habits.  
        This analysis groups users by similar behaviors.
    """)

    # Data loading and preprocessing (if not yet loaded)
    df = pd.read_csv(os.path.join(BASE_DIR, "data.csv"))
    df = df.drop(['Student_ID'], axis=1)

    # Numeric and categorical separation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encoding categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    # Scaling numeric columns
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])
    df_scaled_numeric = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=df.index)

    # Combine preprocessed data
    X_preprocessed = pd.concat([df_scaled_numeric, df_encoded], axis=1)

    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_preprocessed)

    # Add cluster labels to original data
    df_with_labels = df.copy()
    df_with_labels["Cluster"] = labels

    # Extract numeric columns + cluster label
    numeric_df = df_with_labels.select_dtypes(include='number')

    # Calculate cluster summary statistics (mode-based)
    cluster_summary = numeric_df.groupby("Cluster").agg(lambda x: x.mode().iloc[0]).round(1)
    cluster_summary = cluster_summary.applymap(lambda x: '{:g}'.format(x))

    # Display summary table
    st.subheader("📋 Cluster summary (most common values)")
    
    def highlight_clusters(row):
        colors = ['background-color: #FFDDC1',   # soft orange
                'background-color: #D4E2D4',   # soft green
                'background-color: #C6DEF1']   # soft blue
        cluster_number = row.name  # gets the cluster number (0, 1, or 2)
        return [colors[cluster_number % len(colors)]] * len(row)

    st.dataframe(cluster_summary.style.apply(highlight_clusters, axis=1))

    st.markdown("<br>", unsafe_allow_html=True)
    # Additional visualization from previous messages
    st.write("### 📊 Interactive Cluster Visualization")

    df_pca = pd.DataFrame(
        PCA(n_components=3).fit_transform(X_preprocessed),
        columns=['PC1', 'PC2', 'PC3']
    )
    df_pca['Cluster'] = [f'{label}' for label in labels]

    explained = PCA(n_components=3).fit(X_preprocessed).explained_variance_ratio_ * 100

    # Plot
    fig = px.scatter_3d(
        df_pca,
        x="PC1", y="PC2", z="PC3",
        color="Cluster",
        opacity=0.75,
        title="Distribution of Clusters in PCA Space",
        labels={
            "PC1": f"PC1 ({explained[0]:.1f}%)",
            "PC2": f"PC2 ({explained[1]:.1f}%)",
            "PC3": f"PC3 ({explained[2]:.1f}%)"
        },
        color_discrete_map={
            "0": "orange",
            "1": "green",
            "2": "blue"
        }
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.4, color='DarkSlateGrey')))
    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]:.1f}%)",
            yaxis_title=f"PC2 ({explained[1]:.1f}%)",
            zaxis_title=f"PC3 ({explained[2]:.1f}%)"
        ),
        legend_title_text="Clusters",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("🧭 t-SNE Visualization")
    st.write("This 2D visualization helps explore clusters with non-linear dimensionality reduction (t-SNE).")

    # Compute t-SNE (or cache it if needed)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_preprocessed)

    # Prepare DataFrame for plotting
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE_1", "TSNE_2"])
    df_tsne["Cluster"] = labels.astype(str)
    df_tsne["Avg_Daily_Usage_Hours"] = df["Avg_Daily_Usage_Hours"].values

    # Use 'Avg_Daily_Usage_Hours' to scale bubble sizes
    min_val = df_tsne["Avg_Daily_Usage_Hours"].min()
    df_tsne["Size"] = df_tsne["Avg_Daily_Usage_Hours"] - min_val + 0.1

    color_map = {
        "0": "#ff7f0e",  # orange
        "1": "#2ca02c",  # green
        "2": "#1f77b4"   # blue
    }

    # Plot
    fig_tsne = px.scatter(
        data_frame=df_tsne,
        x="TSNE_1",
        y="TSNE_2",
        color="Cluster",
        size="Size",
        title="t-SNE Cluster Map (Bubble Size: Daily Usage Hours)",
        size_max=18,
        width=800,
        height=600,
        color_discrete_map=color_map
    )
    fig_tsne.update_traces(marker=dict(line=dict(width=1, color='white')))

    fig_tsne.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_tsne, use_container_width=True)