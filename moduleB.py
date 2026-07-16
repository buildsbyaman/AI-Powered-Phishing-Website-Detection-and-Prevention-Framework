"""
Module B — Content/NLP-Based Phishing Classifier
==================================================
Combines TF-IDF text features (visible page content) with structural/DOM
features (forms, iframes, brand mismatch, etc.) extracted via
ContentExtraction.py, and trains Logistic Regression and Linear SVM
classifiers.

Usage:
    python moduleB.py --data Datasets/urls_labeled.csv

Input CSV requires columns: url, label   (label: 1 = legitimate, 0 = phishing)
(This will trigger live scraping via ContentExtraction.py on first run,
then cache results — see ContentExtraction.py for details.)

Outputs:
    models/module_b_logreg.pkl
    models/module_b_linear_svm.pkl
    models/module_b_tfidf_vectorizer.pkl
    models/module_b_scaler.pkl
    outputs/module_b_metrics.csv
    outputs/module_b_confusion_matrices.png   (if matplotlib available)
    outputs/module_b_roc_curves.png           (if matplotlib available)
    outputs/module_b_top_terms.png            (if matplotlib available)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ContentExtraction import build_content_dataset

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[!] matplotlib not installed — skipping graphs. "
          "Install with: pip install matplotlib")

STRUCTURAL_FEATURE_COLS = [
    "num_forms", "has_password_field", "num_iframes", "num_scripts",
    "num_links", "external_form_action", "title_brand_mismatch",
    "favicon_mismatch", "has_meta_refresh", "right_click_disabled",
]


# --------------------------------------------------------------------------- #
# 1. Data Preparation
# --------------------------------------------------------------------------- #

def prepare_features(df: pd.DataFrame, vectorizer=None, scaler=None, fit=True):
    """Combine TF-IDF text features with scaled structural features into one matrix."""
    df["text"] = df["text"].fillna("")

    if fit:
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
        text_features = vectorizer.fit_transform(df["text"])
    else:
        text_features = vectorizer.transform(df["text"])

    struct = df[STRUCTURAL_FEATURE_COLS].fillna(0).values
    if fit:
        scaler = StandardScaler()
        struct_scaled = scaler.fit_transform(struct)
    else:
        struct_scaled = scaler.transform(struct)

    combined = hstack([text_features, csr_matrix(struct_scaled)])
    return combined, vectorizer, scaler


# --------------------------------------------------------------------------- #
# 2. Training
# --------------------------------------------------------------------------- #

def train_models(X_train, y_train):
    models = {}

    print("\n[+] Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    models["Logistic Regression"] = logreg

    print("[+] Training Linear SVM (calibrated for probability estimates)...")
    # LinearSVC has no predict_proba by default; CalibratedClassifierCV adds it
    # so we can compute ROC-AUC and combine this module's output with others later.
    base_svm = LinearSVC(class_weight="balanced", max_iter=5000)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    models["Linear SVM"] = svm

    return models


# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #

def evaluate_models(models, X_test, y_test):
    results = []
    predictions, probabilities = {}, {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        })

        print(f"\n{'=' * 60}\nClassification Report — {name}\n{'=' * 60}")
        print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n" + "=" * 60 + "\nMODEL COMPARISON SUMMARY\n" + "=" * 60)
    print(results_df.to_string(index=False))

    return results_df, predictions, probabilities


# --------------------------------------------------------------------------- #
# 4. Graphs (only generated if matplotlib is available — not forced)
# --------------------------------------------------------------------------- #

def plot_confusion_matrices(predictions, y_test, out_dir):
    if not PLOTTING_AVAILABLE:
        return
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Phishing", "Legitimate"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Phishing", "Legitimate"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, "module_b_confusion_matrices.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[+] Saved confusion matrices -> {path}")


def plot_roc_curves(probabilities, y_test, out_dir):
    if not PLOTTING_AVAILABLE:
        return
    plt.figure(figsize=(6, 6))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Module B Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "module_b_roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[+] Saved ROC curves -> {path}")


def plot_top_terms(logreg_model, vectorizer, out_dir, top_n=15):
    """Show the TF-IDF terms most indicative of phishing vs. legitimate pages."""
    if not PLOTTING_AVAILABLE:
        return
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
    except Exception:
        return

    coefs = logreg_model.coef_[0]
    n_text_features = len(feature_names)
    text_coefs = coefs[:n_text_features]  # structural features were appended after text features

    top_phishing_idx = np.argsort(text_coefs)[:top_n]        # most negative -> phishing class (label 0)
    top_legit_idx = np.argsort(text_coefs)[-top_n:]           # most positive -> legitimate class (label 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(feature_names[top_phishing_idx], text_coefs[top_phishing_idx], color="crimson")
    axes[0].set_title("Top Terms Indicating Phishing")
    axes[1].barh(feature_names[top_legit_idx], text_coefs[top_legit_idx], color="seagreen")
    axes[1].set_title("Top Terms Indicating Legitimate")

    plt.tight_layout()
    path = os.path.join(out_dir, "module_b_top_terms.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[+] Saved top-terms plot -> {path}")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train Module B content-based phishing classifiers")
    parser.add_argument("--data", type=str, default="Datasets/urls_labeled.csv",
                         help="CSV with columns: url, label")
    parser.add_argument("--cache-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Scrape (or load cached) content dataset
    df = build_content_dataset(args.data, cache_dir=args.cache_dir)
    if df.empty:
        raise RuntimeError("No usable pages were scraped — check network access and URL list.")

    # 2. Train/test split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=42
    )

    # 3. Feature engineering (TF-IDF + structural, fit on train only)
    X_train, vectorizer, scaler = prepare_features(train_df, fit=True)
    X_test, _, _ = prepare_features(test_df, vectorizer=vectorizer, scaler=scaler, fit=False)
    y_train, y_test = train_df["label"], test_df["label"]

    joblib.dump(vectorizer, os.path.join(args.models_dir, "module_b_tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(args.models_dir, "module_b_scaler.pkl"))

    # 4. Train
    models = train_models(X_train, y_train)

    # 5. Evaluate
    results_df, predictions, probabilities = evaluate_models(models, X_test, y_test)
    results_df.to_csv(os.path.join(args.out_dir, "module_b_metrics.csv"), index=False)
    print(f"\n[+] Saved metrics table -> {os.path.join(args.out_dir, 'module_b_metrics.csv')}")

    # 6. Save models
    for name, model in models.items():
        fname = f"module_b_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, os.path.join(args.models_dir, fname))
        print(f"[+] Saved model -> {os.path.join(args.models_dir, fname)}")

    # 7. Graphs — skipped automatically if matplotlib isn't installed
    plot_confusion_matrices(predictions, y_test, args.out_dir)
    plot_roc_curves(probabilities, y_test, args.out_dir)
    plot_top_terms(models["Logistic Regression"], vectorizer, args.out_dir)

    print("\n[+] Module B training and evaluation complete.")


if __name__ == "__main__":
    main()