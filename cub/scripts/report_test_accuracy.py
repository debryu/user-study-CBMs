"""Report the CUB CBM's end-to-end test-set accuracy: CLIP embedding ->
concept-extractor SVM (decision_function, tanh) -> GT-trained
LogisticRegression -> task label, on the Le Conte vs. Savannah Sparrow
minimal pair. Mirrors the canonical chain in cub/notebooks/train_model.ipynb
(same model classes/hyperparameters/data), extracted into a standalone,
non-notebook script for reporting purposes.

The CBM's task is restricted to the sparrow pair (class ids 123/126) and 6
concepts (concept_mask below) -- this is the actual experiment design (see
cub/cub_preprocessing.md), not a subsample for convenience: the
concept-extractor is trained on every OTHER class (so it isn't fit to the
pair it will be evaluated on), and the label predictor is trained only on
the pair's own GT concepts.

Example:
  cd cub && uv run scripts/report_test_accuracy.py
"""

import argparse

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

NON_CONCEPT_COLS = {"sample_idx", "split", "label", "class_name", "image_path", "image", "embedding", "concepts"}
CONCEPT_MASK = [23, 44, 48, 69, 89, 103]  # striped breast, buff breast, white throat, buff nape, solid belly, brown crown
SPARROW_PAIR = [123, 126]  # Le Conte Sparrow, Savannah Sparrow


def split_tensors(ds, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.tensor(ds[split]["embedding"])
    embeddings = (embeddings - embeddings.mean(0, keepdim=True)) / embeddings.std(0, keepdim=True)
    concepts = torch.tensor(ds[split]["concepts"])
    labels = torch.tensor(ds[split]["label"])
    return embeddings, concepts, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="NWeak/cub-mirror")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.repo_id)

    train_embeddings, train_concepts, train_y = split_tensors(ds, "train")
    test_embeddings, test_concepts, test_y = split_tensors(ds, "test")

    train_other_subset = np.where(~np.isin(train_y.numpy(), SPARROW_PAIR))[0]
    train_pair_subset = np.where(np.isin(train_y.numpy(), SPARROW_PAIR))[0]
    test_pair_subset = np.where(np.isin(test_y.numpy(), SPARROW_PAIR))[0]

    # Concept extractor: CLIP embedding -> 6 masked concepts, trained on
    # every class EXCEPT the sparrow pair (rbf SVM, one per concept).
    concept_model = MultiOutputClassifier(SVC(kernel="rbf", C=1.0, class_weight="balanced"))
    concept_model.fit(
        train_embeddings[train_other_subset].numpy(),
        train_concepts[train_other_subset][:, CONCEPT_MASK].numpy(),
    )

    # Label predictor: GT concepts (rescaled {0,1} -> {-1,+1}) -> task label,
    # trained only on the sparrow pair's own train rows.
    train_pair_concepts_pm1 = 2 * train_concepts[train_pair_subset][:, CONCEPT_MASK].numpy() - 1
    train_pair_y = train_y[train_pair_subset].numpy()
    label_model = LogisticRegression(max_iter=1000, class_weight="balanced", fit_intercept=False)
    label_model.fit(train_pair_concepts_pm1, train_pair_y)

    # End-to-end test evaluation, on the pair's test rows only: embedding ->
    # SVM decision_function -> tanh -> label predictor.
    X_test = test_embeddings[test_pair_subset].numpy()
    logits = np.column_stack([est.decision_function(X_test) for est in concept_model.estimators_])
    concept_activations = np.tanh(logits)
    y_pred = label_model.predict(concept_activations)
    y_true = test_y[test_pair_subset].numpy()

    print("CUB CBM -- end-to-end test accuracy (image -> predicted concepts -> predicted label)")
    print(f"Test set: {args.repo_id}[test], Le Conte (123) vs. Savannah (126) Sparrow, n={len(y_true)}")
    print()
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
