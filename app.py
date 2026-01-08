# app.py
# AutoJudge – Programming Problem Difficulty Predictor

import joblib
import numpy as np
import re
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# ================================
# Load Models & Preprocessors
# ================================
save_path = "models/"

# Shared preprocessing
tfidf = joblib.load(save_path + "tfidf_vectorizer.pkl")
scaler = joblib.load(save_path + "numeric_scaler.pkl")

# Classification
rf = joblib.load(save_path + "rf_final_model.pkl")
selector_cls = joblib.load(save_path + "selectkbest.pkl")
label_encoder = joblib.load(save_path + "label_encoder.pkl")

# Regression (final model)
reg_model = joblib.load(save_path + "xgb_regressor.pkl")
selector_reg = joblib.load(save_path + "reg_selector.pkl")

# ================================
# Keyword Lists (Same as Training)
# ================================
easy_keywords = [
    "array", "loop", "print", "sum", "basic", "integer", "simple"
]

medium_keywords = [
    "binary search", "dfs", "bfs", "sorting",
    "greedy", "stack", "queue", "two pointers"
]

hard_keywords = [
    "dp", "dynamic programming", "bitmask",
    "segment tree", "union find", "flow",
    "shortest path", "graph theory"
]

# ================================
# Text Preprocessing
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_keywords(text, keywords):
    return sum(1 for kw in keywords if kw in text)

# ================================
# Feature Extraction
# ================================
def extract_features(text):
    text = clean_text(text)

    # basic numeric features
    text_length = len(text)
    word_count = len(text.split())

    nums = re.findall(r"\b\d+\b", text)
    max_constraint = max([int(n) for n in nums], default=0)

    # keyword-based features
    easy_kw_count = count_keywords(text, easy_keywords)
    medium_kw_count = count_keywords(text, medium_keywords)
    hard_kw_count = count_keywords(text, hard_keywords)

    numeric = np.array([[
        text_length,
        word_count,
        max_constraint,
        easy_kw_count,
        medium_kw_count,
        hard_kw_count
    ]])

    numeric_scaled = scaler.transform(numeric)
    numeric_sparse = csr_matrix(numeric_scaled)

    return text, numeric_sparse

# ================================
# Prediction Functions
# ================================
def predict_difficulty_class(text):
    text, numeric_sparse = extract_features(text)

    X_tfidf = tfidf.transform([text])
    X_sel = selector_cls.transform(X_tfidf)
    X_final = hstack([X_sel, numeric_sparse])

    pred = rf.predict(X_final)[0]
    return label_encoder.inverse_transform([pred])[0]

def predict_difficulty_score(text):
    text, numeric_sparse = extract_features(text)

    X_tfidf = tfidf.transform([text])
    X_sel = selector_reg.transform(X_tfidf)
    X_final = hstack([X_sel, numeric_sparse])

    score = reg_model.predict(X_final)[0]
    return round(float(score), 2)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge – Programming Problem Difficulty Predictor")

st.write(
    "Paste the problem details below and click **Predict** "
    "to get the difficulty class and difficulty score."
)

# Input fields
problem_desc = st.text_area(
    "Problem Description",
    height=180,
    placeholder="Paste the full problem description here..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Paste the input format here..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Paste the output format here..."
)

# Predict button
if st.button("Predict"):
    if not (problem_desc or input_desc or output_desc):
        st.warning("Please enter at least one field before predicting.")
    else:
        full_text = problem_desc + "\n" + input_desc + "\n" + output_desc

        difficulty_class = predict_difficulty_class(full_text)
        difficulty_score = predict_difficulty_score(full_text)

        st.subheader("Prediction Results")

        st.success(
            f"Predicted Difficulty Class: **{difficulty_class.upper()}**"
        )

        st.info(
            f"Predicted Difficulty Score: **{difficulty_score} / 10**"
        )
