import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "kmeans_pipeline.joblib"
DB_PATH = BASE_DIR / "customers.db"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

st.set_page_config(page_title="CustomerLens | Segmentation", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.7rem 2rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#4338ca 70%,#7c3aed); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin-bottom:.3rem;}
.hero p {color:#e0e7ff; margin-bottom:0;}
.small-muted {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            annual_income REAL NOT NULL,
            spending_score REAL NOT NULL,
            cluster_id INTEGER NOT NULL,
            segment_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None, "Dataset not found. Download Mall_Customers.csv from the Kaggle link in the instructions and place it in the data/ folder."
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        return None, f"Could not read CSV: {exc}"
    aliases = {
        "Annual Income": "Annual Income (k$)",
        "Annual_Income": "Annual Income (k$)",
        "Spending Score": "Spending Score (1-100)",
        "Spending_Score": "Spending Score (1-100)",
        "Genre": "Gender",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).copy()
    if len(df) < 3:
        return None, "Dataset must contain at least three valid rows."
    return df, None

@st.cache_resource
def train_model(data_signature, df):
    # data_signature is passed to invalidate the cached model when source data changes.
    X = df[FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    candidate_scores = {}
    for k in range(2, min(8, len(df) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) > 1:
            candidate_scores[k] = float(silhouette_score(X_scaled, labels))
    best_k = max(candidate_scores, key=candidate_scores.get) if candidate_scores else min(5, len(df)-1)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "features": FEATURES}, MODEL_PATH)
    clustered = df.copy()
    clustered["cluster_id"] = labels
    return {"scaler": scaler, "model": model, "scores": candidate_scores,
            "silhouette": candidate_scores.get(best_k), "k": best_k}, clustered

def make_segment_names(clustered):
    profiles = clustered.groupby("cluster_id")[FEATURES].mean()
    income_mid = profiles[FEATURES[0]].median()
    spend_mid = profiles[FEATURES[1]].median()
    names = {}
    for cid, row in profiles.iterrows():
        high_income = row[FEATURES[0]] >= income_mid
        high_spend = row[FEATURES[1]] >= spend_mid
        if high_income and high_spend:
            label = "High Income · High Spending"
        elif high_income:
            label = "High Income · Careful Spending"
        elif high_spend:
            label = "Lower Income · High Spending"
        else:
            label = "Lower Income · Careful Spending"
        names[int(cid)] = f"{label} (Cluster {int(cid)})"
    return names, profiles

def save_prediction(name, age, income, score, cluster_id, segment):
    with get_connection() as conn:
        conn.execute("""
        INSERT INTO predictions
        (customer_name, age, annual_income, spending_score, cluster_id, segment_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, age, income, score, cluster_id, segment, datetime.now().isoformat(timespec="seconds")))

def load_history():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)

init_db()
st.markdown("""
<div class="hero">
  <h1>CustomerLens</h1>
  <p>Customer segmentation and spending behavior analysis using K-Means clustering.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Dashboard", "Customer Explorer", "Predict Segment", "Prediction History"])
    st.divider()
    st.caption("B.Tech Data Science · Machine Learning Mini Project")
    st.markdown("[Dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")

df, data_error = load_data()
if data_error:
    st.warning(data_error)
    st.markdown("""
    **Dataset setup**
    1. Download `Mall_Customers.csv` from the Kaggle dataset linked in the sidebar.
    2. Place the file in the project's `data/` folder.
    3. Commit and push the CSV to GitHub.
    4. Refresh this Streamlit app.
    """)
    st.stop()

data_signature = f"{len(df)}-{df[FEATURES].sum().sum():.4f}-{df[FEATURES].mean().sum():.4f}"
bundle, clustered = train_model(data_signature, df)
names, profiles = make_segment_names(clustered)
clustered["segment_name"] = clustered["cluster_id"].map(names)

if page == "Dashboard":
    st.subheader("Business overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(clustered):,}")
    c2.metric("Average income", f"${df[FEATURES[0]].mean():.1f}k")
    c3.metric("Average spending score", f"{df[FEATURES[1]].mean():.1f}/100")
    c4.metric("Clusters identified", bundle["k"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Customer segment distribution")
        counts = clustered["segment_name"].value_counts().sort_index()
        st.bar_chart(counts)
    with right:
        st.markdown("#### Income vs spending")
        fig, ax = plt.subplots(figsize=(7, 4))
        for cid, group in clustered.groupby("cluster_id"):
            ax.scatter(group[FEATURES[0]], group[FEATURES[1]], label=names[int(cid)], alpha=.8)
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1–100)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=.2)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Model evaluation")
    m1, m2 = st.columns(2)
    m1.metric("Selected K", bundle["k"])
    m2.metric("Silhouette score", f"{bundle['silhouette']:.3f}" if bundle["silhouette"] is not None else "N/A")
    st.caption("Silhouette score measures cluster separation; it is not classification accuracy.")
    st.markdown("#### Cluster profiles")
    profile_view = profiles.rename(columns={
        FEATURES[0]: "Mean annual income (k$)",
        FEATURES[1]: "Mean spending score"
    })
    profile_view["Customer count"] = clustered.groupby("cluster_id").size()
    profile_view["Segment"] = [names[int(i)] for i in profile_view.index]
    st.dataframe(profile_view, use_container_width=True)

elif page == "Customer Explorer":
    st.subheader("Customer Explorer")
    st.write("Browse the original dataset with its learned cluster assignments.")
    filters = st.columns(2)
    gender_col = "Gender" if "Gender" in clustered.columns else None
    if gender_col:
        choices = ["All"] + sorted(clustered[gender_col].dropna().astype(str).unique().tolist())
        selected_gender = filters[0].selectbox("Gender", choices)
        if selected_gender != "All":
            clustered = clustered[clustered[gender_col].astype(str) == selected_gender]
    segment_choices = ["All"] + sorted(clustered["segment_name"].unique().tolist())
    selected_segment = filters[1].selectbox("Segment", segment_choices)
    if selected_segment != "All":
        clustered = clustered[clustered["segment_name"] == selected_segment]
    st.dataframe(clustered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered customers CSV",
                       clustered.to_csv(index=False).encode("utf-8"),
                       file_name="segmented_customers.csv", mime="text/csv")

elif page == "Predict Segment":
    st.subheader("Predict a new customer's segment")
    st.write("The model uses annual income and spending score to assign the nearest learned cluster.")
    with st.form("prediction_form"):
        name = st.text_input("Customer name")
        a, b, c = st.columns(3)
        age = a.number_input("Age", min_value=1, max_value=120, value=30)
        income = b.number_input("Annual income (k$)", min_value=0.1, max_value=1000.0, value=60.0, step=1.0)
        spending = c.number_input("Spending score (1–100)", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("Predict and save", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter a customer name.")
        else:
            x = pd.DataFrame([[income, spending]], columns=FEATURES)
            x_scaled = bundle["scaler"].transform(x)
            cluster_id = int(bundle["model"].predict(x_scaled)[0])
            segment = names[cluster_id]
            save_prediction(name.strip(), int(age), float(income), float(spending), cluster_id, segment)
            st.success("Prediction saved to the database.")
            st.metric("Predicted segment", segment)
            st.write(f"Cluster ID: **{cluster_id}**")
            st.info("This is a cluster assignment, not a probability or a verified customer category.")

elif page == "Prediction History":
    st.subheader("Prediction History")
    history_df = load_history()
    if history_df.empty:
        st.info("No predictions have been saved yet. Use the Predict Segment page first.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("Download prediction history",
                           history_df.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv", mime="text/csv")

st.divider()
st.caption("Educational prototype. Cluster labels are descriptive summaries of learned centroids and should be validated before business use.")
