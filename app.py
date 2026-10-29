"""
Streamlit app — Assignment 2 (ML Model Comparison)
Loads pre-trained (GridSearchCV-tuned) model pipelines and scores an uploaded test CSV.
No retraining happens here.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Income Classification - Model Comparison", layout="wide")

APP_VERSION = "v3-about"   # shown in the sidebar so you can confirm which code is deployed
BASE_DIR = Path(__file__).resolve().parent


def _find_model_dir():
    """Locate the model/ folder whether the app runs from the repo root or elsewhere."""
    for cand in (BASE_DIR / "model", Path.cwd() / "model", BASE_DIR.parent / "model"):
        if (cand / "logistic_regression.joblib").is_file():
            return cand
    return BASE_DIR / "model"


MODEL_DIR = _find_model_dir()

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "svm.joblib",
}

FEATURE_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
                'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'native_country']
TARGET_COL = "income"


@st.cache_resource
def load_model(path):
    path = Path(path)
    if not path.is_file():
        st.error(
            f"Model file not found: `{path}`\n\n"
            f"App folder: `{BASE_DIR}`  \nWorking dir: `{Path.cwd()}`  \n"
            f"Files in app folder: `{sorted(p.name for p in BASE_DIR.iterdir())}`  \n"
            f"Files in model folder: `{sorted(p.name for p in MODEL_DIR.iterdir()) if MODEL_DIR.is_dir() else 'model/ folder missing'}`\n\n"
            "Check that the `model/` folder with the .joblib files is committed and pushed "
            "to the same repo and branch this Streamlit app is deployed from."
        )
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_best_params(path=MODEL_DIR / "best_hyperparameters.csv"):
    try:
        return pd.read_csv(path, index_col=0)["Best hyperparameters"].to_dict()
    except Exception:
        return {}


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


UCI_URL = "https://archive.ics.uci.edu/dataset/2/adult"
REPO_URL = "https://github.com/Rakshanthamidhun/ML_assignment_2"

st.title("Income Classification — Model Comparison")
st.caption(
    f"[UCI Adult (Census Income) dataset]({UCI_URL}) · 5 classifiers tuned with 5-fold CV and "
    "trained offline · upload a test CSV in the sidebar to see how each model performs."
)

with st.expander("About this application", expanded=True):
    st.markdown(
        f"""
**What it does**

This app predicts whether a person's annual income is **above $50K** from census attributes
(age, education, occupation, marital status, hours worked and so on), and compares five
classification models on the same test data.

- **Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes (Gaussian) and SVM (RBF).
  Each is a scikit-learn pipeline (scaling + one-hot encoding + classifier), tuned with
  `GridSearchCV` using 5-fold stratified cross-validation and selected by F1.
- **Class imbalance:** about 75% `<=50K` / 25% `>50K`. The data is not resampled; class weighting
  is used where cross-validation showed it helps.
- **No retraining:** the app loads the saved models from `model/` and only scores the data you upload.

**How to use it**

1. Upload a test CSV in the sidebar, or download the sample `test_data.csv` (1,200 held-out rows).
2. Pick a model from the dropdown to see its Accuracy, AUC, Precision, Recall, F1 and MCC,
   plus the confusion matrix and classification report.
3. Scroll down to compare all five models side by side.

**Dataset**

- Name: UCI Adult (Census Income), 48,842 rows, binary target `income` (`>50K` = 1)
- Source: [UCI Machine Learning Repository]({UCI_URL})
- Used here: a stratified 6,000-row sample with 12 features, split 80/20 into train and test

**Source code:** [{REPO_URL}]({REPO_URL})
"""
    )

# ---------------------------------------------------------------------------
# Sidebar: upload + model selection
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
st.sidebar.markdown(f"**Dataset:** [UCI Adult (Census Income)]({UCI_URL})")
st.sidebar.caption(f"App build: {APP_VERSION}")

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV", type="csv",
    help="Must contain the 12 feature columns plus the 'income' target column (0/1)."
)

try:
    with open(BASE_DIR / "test_data.csv", "rb") as f:
        st.sidebar.download_button("Download sample test_data.csv", f, file_name="test_data.csv",
                                   mime="text/csv", help="Held-out 1,200-row test split from training.")
except FileNotFoundError:
    pass

selected_model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note on class imbalance:** the dataset was **not resampled** (no SMOTE). "
    "It keeps the real ~75/25 ratio. Imbalance is handled with `class_weight='balanced'` "
    "during training, chosen per model by 5-fold CV."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Expected columns:**\n\n" + ", ".join(FEATURE_COLS) + f", **{TARGET_COL}**"
)

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to get started. "
            "(Use the `test_data.csv` included in this repo to try it out.)")
    st.stop()

# ---------------------------------------------------------------------------
# Load + validate uploaded data
# ---------------------------------------------------------------------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required column(s): {', '.join(missing_cols)}")
    st.stop()

if not set(data[TARGET_COL].dropna().unique()) <= {0, 1}:
    st.error(f"'{TARGET_COL}' must be binary 0/1 (1 = >50K). Found: {sorted(data[TARGET_COL].unique())[:5]}")
    st.stop()

X = data[FEATURE_COLS]
y_true = data[TARGET_COL]

st.subheader("Uploaded Data Preview")
st.dataframe(data.head(10), width="stretch")
st.caption(f"{data.shape[0]} rows loaded.")

# ---------------------------------------------------------------------------
# Selected-model evaluation
# ---------------------------------------------------------------------------
st.header(f"Results — {selected_model_name}")

best_params = load_best_params()
if selected_model_name in best_params:
    st.caption(f"Tuned hyperparameters: `{best_params[selected_model_name]}`")

model = load_model(MODEL_FILES[selected_model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y_true, y_pred, y_proba)

metric_cols = st.columns(6)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

col_cm, col_report = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=["≤50K", ">50K"], yticklabels=["≤50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_report:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=["≤50K", ">50K"], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, width="stretch")

# ---------------------------------------------------------------------------
# All-models comparison on the same uploaded data
# ---------------------------------------------------------------------------
st.header("All Models on This Test Data")

all_results = {}
for name, path in MODEL_FILES.items():
    m = load_model(path)
    pred = m.predict(X)
    proba = m.predict_proba(X)[:, 1]
    all_results[name] = compute_metrics(y_true, pred, proba)

comparison_df = pd.DataFrame(all_results).T.round(4)
st.dataframe(
    comparison_df.style.highlight_max(axis=0, color="#2a9d8f44"),
    width="stretch",
)

ranking = comparison_df.sort_values(["F1", "MCC"], ascending=False)
winner = ranking.index[0]
st.success(
    f"**Winner on this data: {winner}** (F1 {ranking.loc[winner, 'F1']:.4f}, "
    f"MCC {ranking.loc[winner, 'MCC']:.4f}). Ranked by F1, then MCC — fair metrics for the "
    "imbalanced target. Ranking: " + " › ".join(ranking.index)
)

fig2, ax2 = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(comparison_df.columns))
width = 0.15
palette = sns.color_palette("viridis", len(comparison_df))
for i, (name, row) in enumerate(comparison_df.iterrows()):
    ax2.bar(x + i * width, row.values, width, label=name, color=palette[i])
ax2.set_xticks(x + width * (len(comparison_df) - 1) / 2)
ax2.set_xticklabels(comparison_df.columns)
ax2.set_ylabel("Score")
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax2.grid(axis="y", alpha=0.3)
st.pyplot(fig2)

st.caption("All models are pre-trained and loaded from disk — nothing is retrained in this app.")
