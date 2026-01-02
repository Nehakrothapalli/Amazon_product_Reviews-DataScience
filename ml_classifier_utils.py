# ml_classifier_utils.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import joblib

LABELS = ["Negative", "Neutral", "Positive"]

def textblob_label(text: str, pos_thresh: float = 0.1, neg_thresh: float = -0.1) -> str:
    if not isinstance(text, str):
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > pos_thresh:
        return "Positive"
    elif polarity < neg_thresh:
        return "Negative"
    return "Neutral"

def add_rule_labels(df: pd.DataFrame, text_col: str, label_col: str = "rule_label") -> pd.DataFrame:
    out = df.copy()
    out[label_col] = out[text_col].astype(str).map(textblob_label)
    return out

def prepare_text(df: pd.DataFrame, text_col: str) -> pd.Series:
    # Lightweight cleaning; keep as-is to match earlier app behavior
    return df[text_col].astype(str).fillna("").str.strip()

def train_tfidf_logreg(
    df: pd.DataFrame,
    text_col: str = "Text",
    label_col: str = "rule_label",
    sample_size: int = 30000,
    random_state: int = 42,
    ngram_max: int = 2,
    max_features: int = 100_000,
    class_weight: str | None = "balanced",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a TF-IDF + LogisticRegression classifier using rule-based labels as ground truth.
    We use class_weight='balanced' instead of SMOTE since TF-IDF is sparse.
    Returns: (pipeline, metrics_dict)
    """
    df = df.dropna(subset=[text_col]).copy()
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    X = prepare_text(df, text_col)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return pipe, metrics

def save_model(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)

def predict_texts(pipe: Pipeline, texts: List[str]) -> List[str]:
    return list(pipe.predict(texts))