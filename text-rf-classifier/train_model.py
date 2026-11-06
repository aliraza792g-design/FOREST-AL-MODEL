"""
Random Forest text classifier (scikit-learn).

Usage
-----
# 1) Your own CSV
python train_model.py --data data/my_data.csv --text-col text --label-col label

# 2) No dataset yet? Use the built-in demo (20 Newsgroups, 4 classes)
python train_model.py --demo
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from text_utils import clean_text

RANDOM_STATE = 42
OUT_DIR, MODEL_DIR = "outputs", "models"


def load_data(args):
    if args.demo:
        from sklearn.datasets import fetch_20newsgroups
        cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
        d = fetch_20newsgroups(subset="all", categories=cats,
                               remove=("headers", "footers", "quotes"))
        df = pd.DataFrame({"text": d.data,
                           "label": [d.target_names[i] for i in d.target]})
        return df, "text", "label"
    df = pd.read_csv(args.data)
    return df, args.text_col, args.label_col


def eda(df, tcol, lcol):
    print("=" * 60, "\nEXPLORATORY ANALYSIS\n" + "=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df[[tcol, lcol]].isna().sum()}")
    print(f"Duplicate texts: {df[tcol].duplicated().sum()}")
    print(f"Classes: {df[lcol].nunique()}")
    print(f"Class distribution:\n{df[lcol].value_counts()}")
    print(f"Avg words per doc: {df[tcol].astype(str).str.split().str.len().mean():.1f}")
    plt.figure(figsize=(7, 4))
    df[lcol].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Class distribution"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n-iter", type=int, default=15, help="random search iterations")
    args = ap.parse_args()
    if not args.demo and not args.data:
        ap.error("Provide --data CSV or use --demo")

    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load + EDA
    df, tcol, lcol = load_data(args)
    eda(df, tcol, lcol)
    df = df.dropna(subset=[tcol, lcol]).drop_duplicates(subset=[tcol])

    # 2. Preprocess
    print("\nPreprocessing text ...")
    df["clean"] = df[tcol].astype(str).apply(clean_text)
    df = df[df["clean"].str.strip() != ""]

    # 4. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df[lcol], test_size=0.2, stratify=df[lcol],
        random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 3 + 5. TF-IDF + RF in ONE pipeline so CV never leaks vocabulary/IDF
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                  min_df=2, max_df=0.9, sublinear_tf=True)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                      class_weight="balanced")),
    ])
    param_dist = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 20, 40, 60],
        "rf__min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(pipe, param_dist, n_iter=args.n_iter, cv=3,
                                scoring="f1_weighted", random_state=RANDOM_STATE,
                                n_jobs=-1, verbose=2)
    print("\nTuning hyperparameters ...")
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV weighted-F1: {search.best_score_:.4f}")
    best = search.best_estimator_

    # 6. Evaluate
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted",
                                                  zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nTest accuracy: {acc:.4f}\n\n{report}")

    labels = sorted(df[lcol].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150); plt.close()

    # 7. Top-20 features
    names = best.named_steps["tfidf"].get_feature_names_out()
    imp = best.named_steps["rf"].feature_importances_
    top = np.argsort(imp)[::-1][:20]
    top_feats = [(names[i], float(imp[i])) for i in top]
    print("Top 20 features:")
    for n, v in top_feats:
        print(f"  {n:25s} {v:.5f}")
    plt.figure(figsize=(8, 6))
    plt.barh([n for n, _ in top_feats][::-1], [v for _, v in top_feats][::-1],
             color="darkorange")
    plt.xlabel("Importance"); plt.title("Top 20 features"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150); plt.close()

    # Save artifacts (compress to keep file small for GitHub's 100 MB limit)
    metrics = {"accuracy": acc, "precision_weighted": p, "recall_weighted": r,
               "f1_weighted": f1, "best_params": search.best_params_,
               "cv_f1_weighted": search.best_score_, "top_features": top_feats,
               "n_train": len(X_train), "n_test": len(X_test)}
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    joblib.dump(best, f"{MODEL_DIR}/text_rf.joblib", compress=3)
    size = os.path.getsize(f"{MODEL_DIR}/text_rf.joblib") / 1e6
    print(f"\nSaved model ({size:.1f} MB) -> {MODEL_DIR}/text_rf.joblib")

    # 8. Summary
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"Accuracy {acc:.3f} | weighted P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    print("Improvements to try: more data, class balancing, char n-grams, "
          "linear models (LogReg/LinearSVC) as baseline, or transformer embeddings.")


if __name__ == "__main__":
    main()
