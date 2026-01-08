# AutoJudge – Programming Problem Difficulty Prediction

## 📌 Project Overview
**AutoJudge** is a machine learning system that automatically predicts the difficulty of
programming problems using only their textual descriptions.

The system performs **two tasks**:
1. **Classification** – Predicts difficulty class (*Easy / Medium / Hard*)
2. **Regression** – Predicts a numerical difficulty score on a **1–10 scale**

The project uses **Natural Language Processing (NLP)** with classical machine learning
models and provides a **Streamlit-based web interface** for real-time predictions.

---

## 📂 Dataset
- **Source:** Competitive programming problem statements
- **Format:** JSONL
- **Fields:**
  - Title
  - Problem Description
  - Input Description
  - Output Description
  - Difficulty Class (Easy / Medium / Hard)
  - Difficulty Score (1–10)

The dataset contains **no missing values**, allowing direct preprocessing.

---

## 🛠️ Methodology

### 🔹 Data Preprocessing
- Combined all text fields into a single `full_text`
- Text cleaning:
  - Lowercasing
  - Removal of extra spaces and line breaks
- Same preprocessing pipeline used for both classification and regression

---

### 🔹 Feature Engineering

#### Text Features
- **TF-IDF Vectorization**
  - Unigrams and bigrams
  - High-dimensional sparse representation

#### Keyword-Based Difficulty Signals
To capture explicit difficulty-related cues present in problem statements, 
keyword-based features were engineered using domain knowledge:

- **Easy keywords:** array, loop, print, sum, basic, integer, simple  
- **Medium keywords:** binary search, DFS, BFS, sorting, greedy, stack, queue  
- **Hard keywords:** DP, dynamic programming, bitmask, segment tree, union find, flow, graph theory  

For each problem, the **frequency of Easy, Medium, and Hard keywords** was computed.
These features help the model capture **semantic hints of problem complexity** that
may not be fully represented by TF-IDF alone.

> During experimentation, it was observed that excessive reliance on explicit difficulty
keywords could introduce bias. Therefore, keyword features were carefully evaluated and
used only when they improved generalization without causing label leakage.

---

#### Numeric Features
- Text length
- Word count
- Maximum numeric constraint extracted from text

Numeric features are **standardized** and concatenated with TF-IDF features.

---

### 🔹 Feature Selection
To handle high-dimensional text data:
- **SelectKBest (Chi-square)** for classification
- **SelectKBest (F-regression)** for regression

Feature selection is applied **only on training data**, preventing data leakage and
improving generalization.

---

## 🤖 Models Used

### 🔸 Classification
- Logistic Regression (baseline)
- **Random Forest Classifier**
- Support Vector Machine (SVM)

**Why RF?**
- Performs well on sparse, high-dimensional TF-IDF features
- Effectively separates Easy, Medium, and Hard classes
- Robust to class imbalance

### 🔸 Regression
- Linear Regression
- Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression
- **XGBoost Regression** ✅ *(Final model)*

Regression models were trained **directly on the original 1–10 scale** for better
interpretability.

---

## 📊 Results

### 🔹 Classification Results
- **Accuracy:** ~55.16%
- **Best Model:** RF + SelectKBest + Optuna
- Feature selection and keyword-aware features improved macro F1-score

---

### 🔹 Regression Results
- **MAE:** ~1.6549
- **RMSE:** ~1.9969
- **R²:** ~0.1783
- **Best Model:** XGBoost Regression

The regression model predicts difficulty scores with an average error of
**less than one difficulty point**, which is strong given the subjective nature
of difficulty estimation.

---

## 🌐 Web Application
A lightweight web application was built using **Streamlit**.

### Inputs
- Problem description
- Input description
- Output description

### Outputs
- Predicted difficulty class (Easy / Medium / Hard)
- Predicted difficulty score (1–10 scale)

The web app uses the **same trained models and preprocessing pipeline**
as the final notebook.

---

## How to Run the Project Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/Vishalmeena0658/AutoJudge.git
cd AutoJudge

### Step 2: Install dependencies
pip install -r requirements.txt

###Step 3: Run the web app
python -m streamlit run app.py

##Project Folder Structure

AutoJudge/
│
├── app.py                      # Streamlit web application
├── AutoJudge_final.ipynb       # Final training & evaluation notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
│
├── data/
│   └── problems_data.jsonl     # Dataset
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── numeric_scaler.pkl
│   ├── selectkbest.pkl
│   ├── reg_selector.pkl
│   ├── rf_final_model.pkl
│   ├── xgb_regressor.pkl
│   └── label_encoder.pkl
│
└── Report.pdf    # Final project report

