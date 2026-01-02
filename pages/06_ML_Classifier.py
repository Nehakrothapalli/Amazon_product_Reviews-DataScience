# pages/06_ML_Classifier.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from ml_classifier_utils import (
    add_rule_labels, train_tfidf_logreg, save_model, load_model, predict_texts, LABELS
)

st.set_page_config(page_title="ML Classifier (TF‑IDF + Logistic Regression)", layout="wide")
st.title("Sentiment Model Baseline")

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

# --- Data selection ---
st.sidebar.header("Data")
default_path = "attached_assets/Reviews.csv"
data_choice = st.sidebar.selectbox("Source", ["Default dataset path", "Upload CSV"])

df = None
if data_choice == "Default dataset path":
    if os.path.exists(default_path):
        df = load_csv(default_path)
    else:
        st.warning(f"Default path not found: {default_path}. Please upload a CSV instead.")
else:
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if up is not None:
        df = load_csv(up)

if df is None:
    st.stop()

# Choose text column
text_cols = [c for c in df.columns if df[c].dtype == "object"]
text_col = st.sidebar.selectbox("Text column", text_cols if text_cols else df.columns.tolist())

# Create rule-based labels (TextBlob) if not present
label_col = st.sidebar.text_input("Label column (will be created if missing)", "rule_label")
if label_col not in df.columns:
    with st.spinner("Labelling with TextBlob…"):
        df = add_rule_labels(df, text_col, label_col)

# --- Training controls ---
st.sidebar.header("Training")
sample_size = st.sidebar.slider("Sample size", min_value=5000, max_value=80000, value=30000, step=5000)
ngram_max = st.sidebar.radio("N‑gram range", [1, 2], index=1, horizontal=True)
max_features = st.sidebar.select_slider("Max features", options=[20000, 50000, 100000, 200000], value=100000)

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("🚀 Train model", use_container_width=True):
        with st.spinner("Training TF‑IDF + Logistic Regression…"):
            pipe, metrics = train_tfidf_logreg(
                df, text_col=text_col, label_col=label_col, sample_size=sample_size,
                ngram_max=int(ngram_max), max_features=int(max_features)
            )
            st.session_state["ml_pipe"] = pipe
            st.session_state["ml_metrics"] = metrics

with col_b:
    if st.button("💾 Save model", use_container_width=True, disabled=("ml_pipe" not in st.session_state)):
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "tfidf_logreg.joblib")
        save_model(st.session_state["ml_pipe"], save_path)
        st.success(f"Saved to {save_path}")

# --- Metrics ---
if "ml_metrics" in st.session_state:
    m = st.session_state["ml_metrics"]
    st.subheader("Evaluation")
    left, right = st.columns([1,1], vertical_alignment="top")

    with left:
        st.metric("Accuracy", f"{m['accuracy']:.3f}")
        cm = np.array(m["confusion_matrix"])
        df_cm = pd.DataFrame(cm, columns=m["labels"], index=m["labels"])
        fig = px.imshow(df_cm, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rep = pd.DataFrame(m["report"]).T.reset_index().rename(columns={"index":"label"})
        st.dataframe(rep, use_container_width=True, height=360)

# --- Predictor ---
st.subheader("Try the trained model")
txt = st.text_area("Enter a review", placeholder="Type a product review here…")
btn = st.button("Predict sentiment", disabled=("ml_pipe" not in st.session_state or len(txt.strip())==0))
if btn:
    pred = predict_texts(st.session_state["ml_pipe"], [txt])[0]
    st.success(f"**Prediction:** {pred}")