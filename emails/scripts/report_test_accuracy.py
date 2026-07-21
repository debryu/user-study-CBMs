"""Report the emails CBM's end-to-end test-set accuracy: embedding ->
concept-extractor SVM (decision_function, tanh) -> GT-trained
LogisticRegression -> task label. Mirrors the canonical chain in
train_concept_extractor2.ipynb (same model classes/hyperparameters/data),
extracted into a standalone, non-notebook script for reporting purposes.

Example:
  cd emails && uv run scripts/report_test_accuracy.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "user_study_data.parquet")
    train_class = train[train["label"].isin([0, 1])]
    test_class = test[test["label"].isin([0, 1])]

    # Concept extractor: embedding -> 6 concepts (linear SVM, one per concept).
    X_train = np.stack(train["embedding"].values)
    y_train_concepts = np.stack(train["concept_gts_f"].values)
    concept_model = MultiOutputClassifier(SVC(kernel="linear", C=1.0))
    concept_model.fit(X_train, y_train_concepts)

    # Label predictor: GT concepts (rescaled {0,1} -> {-1,+1}) -> task label.
    train_concepts_pm1 = 2 * np.stack(train_class["concept_gts_f"].values) - 1
    train_labels = np.stack(train_class["label"].values)
    label_model = LogisticRegression(max_iter=1000, C=1, solver="lbfgs",
                                      fit_intercept=False, penalty=None, class_weight="balanced")
    label_model.fit(train_concepts_pm1, train_labels)

    # End-to-end test evaluation: embedding -> SVM decision_function -> tanh -> label predictor.
    X_test = np.stack(test_class["embedding"].values)
    logits = np.column_stack([est.decision_function(X_test) for est in concept_model.estimators_])
    concept_activations = np.tanh(logits)
    y_pred = label_model.predict(concept_activations)
    y_true = np.stack(test_class["label"].values)

    print(f"Emails CBM -- end-to-end test accuracy (embedding -> predicted concepts -> predicted label)")
    print(f"Test set: user_study_data.parquet, label in {{0,1}}, n={len(test_class)}")
    print()
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
