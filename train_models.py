"""
train_models.py — standalone version of ml2_assignment.ipynb (Sections 3-7).

Cleans the UCI Adult data, creates the stratified 6,000-row sample and 80/20 split,
tunes all five classifiers with 5-fold stratified GridSearchCV (refit on F1),
evaluates them on the held-out test set and saves the pipelines to model/.

Class imbalance: the data is NOT resampled. Class weighting is a tuned
hyperparameter, chosen per model by cross-validation.

Usage:
    python train_models.py
"""
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SAMPLE_SIZE = 6000
RAW_PATH = "data/adult_raw.csv"
MODEL_DIR = "model"

RAW_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"]
NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship", "race",
                    "sex", "native_country"]

FILE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path, header=None, names=RAW_COLS, na_values="?", skipinitialspace=True)
    df = df.drop(columns=["fnlwgt", "education"])
    df["native_country"] = np.where(df["native_country"] == "United-States",
                                    "United-States", "Other")
    df = df.dropna(subset=["workclass", "occupation"])
    df["income"] = (df["income"] == ">50K").astype(int)
    return df


def make_split(df):
    sample, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df["income"],
                                 random_state=RANDOM_STATE)
    X = sample[NUMERIC_COLS + CATEGORICAL_COLS]
    y = sample["income"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def model_grids():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10, 100],
             "model__class_weight": [None, "balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [4, 6, 8, 10, 12, None],
             "model__min_samples_leaf": [1, 5, 10, 20],
             "model__criterion": ["gini", "entropy"],
             "model__class_weight": [None, "balanced"]}),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11, 15, 21, 31, 41],
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]}),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": np.logspace(-9, 0, 10),
             "model__priors": [None, [0.5, 0.5]]}),
        "Support Vector Machine": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10],
             "model__gamma": ["scale", 0.01, 0.1],
             "model__class_weight": [None, "balanced"]}),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_and_clean()
    X_train, X_test, y_train, y_test = make_split(df)
    print(f"Cleaned rows: {len(df)} | train {len(X_train)} | test {len(X_test)} | "
          f"positive rate {y_train.mean():.3f}")

    test_data = X_test.copy()
    test_data["income"] = y_test.values
    test_data.to_csv("test_data.csv", index=False)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
         CATEGORICAL_COLS),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "mcc": make_scorer(matthews_corrcoef), "auc": "roc_auc",
               "accuracy": "accuracy", "precision": "precision", "recall": "recall"}

    os.makedirs(MODEL_DIR, exist_ok=True)
    results, params = {}, {}
    for name, (clf, grid) in model_grids().items():
        search = GridSearchCV(Pipeline([("preprocess", preprocessor), ("model", clf)]),
                              grid, scoring=scoring, refit="f1", cv=cv, n_jobs=-1)
        t0 = time.time()
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        metrics = evaluate(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
        metrics.update({"CV F1": search.best_score_,
                        "Tuning Time (s)": round(time.time() - t0, 1),
                        "Refit Time (s)": round(search.refit_time_, 3)})
        results[name] = metrics
        params[name] = ", ".join(f"{k.replace('model__', '')}={v}"
                                 for k, v in search.best_params_.items())
        joblib.dump(pipe, f"{MODEL_DIR}/{FILE_NAMES[name]}.joblib", compress=3)
        print(f"✓ {name:<24} test F1 {metrics['F1']:.4f} | {params[name]}")

    results_df = pd.DataFrame(results).T
    results_df.iloc[:, :7] = results_df.iloc[:, :7].astype(float).round(4)
    results_df.to_csv(f"{MODEL_DIR}/comparison_table.csv")
    pd.DataFrame({"Best hyperparameters": params}).to_csv(f"{MODEL_DIR}/best_hyperparameters.csv")
    print("\n", results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].to_string())
    print(f"\nWinner by F1: {results_df['F1'].astype(float).idxmax()}")


if __name__ == "__main__":
    main()
