#!/usr/bin/env python
# coding:utf-8

import pickle
import numpy as np


# ---------------------------------------------------------------------
# Load hierarchy
# ---------------------------------------------------------------------

ANCESTER_INDICES_PATH = 'ancestor_indices.pkl'

with open(ANCESTER_INDICES_PATH, "rb") as f:
    ancestor_indices = pickle.load(f)


# ---------------------------------------------------------------------
# Existing flat metric helper
# ---------------------------------------------------------------------

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, count of correct predictions
    :param predict: int, count of predictions
    :param total: int, count of ground-truth labels
    :return: precision, recall, f1
    """
    p, r, f = 0.0, 0.0, 0.0

    if predict > 0:
        p = float(right) / predict

    if total > 0:
        r = float(right) / total

    if p + r > 0:
        f = 2 * p * r / (p + r)

    return p, r, f


# ---------------------------------------------------------------------
# Hierarchical metrics
# ---------------------------------------------------------------------

def augment_with_ancestors(binary_matrix):
    """
    Add ancestor labels to every active label.

    Parameters
    ----------
    binary_matrix : np.ndarray, shape (N, L)
        Binary label matrix.

    Returns
    -------
    np.ndarray, shape (N, L)
        Binary matrix with ancestors activated.
    """
    augmented = binary_matrix.copy().astype(bool)

    for i, ancestors in enumerate(ancestor_indices):
        for anc_idx in ancestors:
            # If label i is active, activate ancestor anc_idx.
            augmented[:, anc_idx] |= binary_matrix[:, i]

    return augmented


def hierarchical_f1_micro(labels, predictions):
    """
    Compute micro-averaged hierarchical F1.
    """
    labels_aug = augment_with_ancestors(labels.astype(bool))
    preds_aug = augment_with_ancestors(predictions.astype(bool))

    tp = (labels_aug & preds_aug).sum(axis=1)
    pred_count = preds_aug.sum(axis=1)
    true_count = labels_aug.sum(axis=1)

    total_tp = np.sum(tp)
    total_pred = np.sum(pred_count)
    total_true = np.sum(true_count)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def hierarchical_f1_macro(labels, predictions):
    """
    Label-macro hierarchical F1.

    - Adds ancestor labels to ground truth and predictions.
    - Computes TP, FP, FN for each label.
    - Computes F1 separately for each label.
    - Excludes labels with no ground-truth support.
    - Macro-averages the remaining label-level F1 scores.
    """
    labels_aug = augment_with_ancestors(labels.astype(bool))
    preds_aug = augment_with_ancestors(predictions.astype(bool))

    tp = np.sum(labels_aug & preds_aug, axis=0)
    fp = np.sum(~labels_aug & preds_aug, axis=0)
    fn = np.sum(labels_aug & ~preds_aug, axis=0)

    # Labels that occur in the augmented ground truth.
    valid = (tp + fn) > 0

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) > 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) > 0
    )

    f1_per_label = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0
    )

    # Avoid mean(empty array) if evaluation data has no labels.
    return np.mean(f1_per_label[valid]) if np.any(valid) else 0.0


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(epoch_predicts, epoch_labels, id2label,
             threshold=0.5, top_k=None):
    """
    Evaluate flat and hierarchical multilabel metrics.

    Parameters
    ----------
    epoch_labels : List[List[int]]
        Ground-truth label IDs for each sample.

    epoch_predicts : List[List[float]]
        Predicted probability for every label.

    id2label : Dict[int, str]
        Mapping from label ID to label name.

    threshold : float
        Probability threshold used to select predicted labels.

    top_k : int or None
        If specified, consider at most the top-k predictions.

    Returns
    -------
    dict
        Flat precision/recall/F1 plus hierarchical F1 metrics.
    """

    assert len(epoch_predicts) == len(epoch_labels), (
        "mismatch between prediction and ground truth for evaluation"
    )

    num_samples = len(epoch_labels)
    num_labels = len(id2label)

    # -------------------------------------------------------------
    # Binary matrices needed for hierarchical metrics
    #
    # Shape:
    #     (number of samples, number of labels)
    # -------------------------------------------------------------

    gold_binary = np.zeros(
        (num_samples, num_labels),
        dtype=bool
    )

    pred_binary = np.zeros(
        (num_samples, num_labels),
        dtype=bool
    )

    # -------------------------------------------------------------
    # Existing flat metric counters
    # -------------------------------------------------------------

    confusion_count_list = [
        [0 for _ in range(num_labels)]
        for _ in range(num_labels)
    ]

    right_count_list = [0 for _ in range(num_labels)]
    gold_count_list = [0 for _ in range(num_labels)]
    predicted_count_list = [0 for _ in range(num_labels)]

    # -------------------------------------------------------------
    # Process each sample
    # -------------------------------------------------------------

    for sample_idx, (sample_predict, sample_gold) in enumerate(
        zip(epoch_predicts, epoch_labels)
    ):
        np_sample_predict = np.asarray(
            sample_predict,
            dtype=np.float32
        )

        # Highest probability first.
        sample_predict_descent_idx = np.argsort(-np_sample_predict)

        # IMPORTANT:
        # Don't modify top_k itself inside the loop.
        sample_top_k = (
            len(sample_predict)
            if top_k is None
            else min(top_k, len(sample_predict))
        )

        sample_predict_id_list = []

        for j in range(sample_top_k):
            label_id = sample_predict_descent_idx[j]

            if np_sample_predict[label_id] > threshold:
                sample_predict_id_list.append(label_id)

        # ---------------------------------------------------------
        # Fill binary matrices
        # ---------------------------------------------------------

        for gold_id in sample_gold:
            gold_binary[sample_idx, gold_id] = True

        for pred_id in sample_predict_id_list:
            pred_binary[sample_idx, pred_id] = True

        # ---------------------------------------------------------
        # Existing flat metric counting
        # ---------------------------------------------------------

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # Gold and correct predictions.
        for gold in sample_gold:
            gold_count_list[gold] += 1

            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # Predicted labels.
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    # -------------------------------------------------------------
    # Existing flat per-label metrics
    # -------------------------------------------------------------

    precision_dict = {}
    recall_dict = {}
    fscore_dict = {}

    right_total = 0
    predict_total = 0
    gold_total = 0

    for i, label in id2label.items():
        label_name = label + "_" + str(i)

        (
            precision_dict[label_name],
            recall_dict[label_name],
            fscore_dict[label_name]
        ) = _precision_recall_f1(
            right_count_list[i],
            predicted_count_list[i],
            gold_count_list[i]
        )

        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # -------------------------------------------------------------
    # Flat macro F1
    # -------------------------------------------------------------

    precision_macro = (
        sum(precision_dict.values()) / len(precision_dict)
        if precision_dict else 0.0
    )

    recall_macro = (
        sum(recall_dict.values()) / len(recall_dict)
        if recall_dict else 0.0
    )

    macro_f1 = (
        sum(fscore_dict.values()) / len(fscore_dict)
        if fscore_dict else 0.0
    )

    # -------------------------------------------------------------
    # Flat micro F1
    # -------------------------------------------------------------

    precision_micro = (
        float(right_total) / predict_total
        if predict_total > 0
        else 0.0
    )

    recall_micro = (
        float(right_total) / gold_total
        if gold_total > 0
        else 0.0
    )

    micro_f1 = (
        2 * precision_micro * recall_micro
        / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    # -------------------------------------------------------------
    # Hierarchical metrics
    # -------------------------------------------------------------

    hierarchical_micro_f1 = hierarchical_f1_micro(
        gold_binary,
        pred_binary
    )

    hierarchical_macro_f1 = hierarchical_f1_macro(
        gold_binary,
        pred_binary
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    return {
        "precision": precision_micro,
        "recall": recall_micro,

        # Standard/flat metrics
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,

        # Hierarchical metrics
        "hierarchical_micro_f1": hierarchical_micro_f1,
        "hierarchical_macro_f1": hierarchical_macro_f1,

        "full": [
            precision_dict,
            recall_dict,
            fscore_dict,
            right_count_list,
            predicted_count_list,
            gold_count_list,
        ],
    }