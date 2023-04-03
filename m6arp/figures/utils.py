from sklearn.utils import check_consistent_length, column_or_1d, check_array, assert_all_finite
from sklearn.utils.multiclass import type_of_target
import numpy as np
import warnings

# Copied from https://github.com/scikit-learn/scikit-learn/blob/2e481f114169396660f0051eee1bcf6bcddfd556/sklearn/metrics/_base.py#L202
def _check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.
    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.
    Parameters
    ----------
    pos_label : int, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.
    Returns
    -------
    pos_label : int
        If `pos_label` can be inferred, it will be returned.
    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if pos_label is None and (
        classes.dtype.kind in "OUS"
        or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        )
    ):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            "specified: either make y_true take value in {0, 1} or "
            "{-1, 1} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1

    return pos_label

def dumb_binary_clf_curve(y_true, y_score, thresholds, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.
    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    pos_label : int or str, default=None
        The label of the positive class.
    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    fps = []
    tps = []

    for t in thresholds:
        y_pred = y_score > t

        true_positives = np.logical_and(y_true, y_pred)
        n_true_positives = sum(true_positives)
        tps.append(n_true_positives)

        false_positives = np.logical_and(~y_true, y_pred)
        n_false_positives = sum(false_positives)
        fps.append(n_false_positives)

    return np.array(fps), np.array(tps)

def dumb_roc_curve(y_true, y_score, thresholds, *, pos_label=None, sample_weight=None):
    fps, tps = dumb_binary_clf_curve(
        y_true, y_score, thresholds, pos_label=pos_label, sample_weight=sample_weight
    )

    # print(thresholds.shape)
    # print(fps.shape)
    # print(tps.shape)

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    # tps = np.r_[0, tps]
    # fps = np.r_[0, fps]
    # thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless",
            UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr

def dumb_precision_recall_curve(y_true, y_score, thresholds, *, pos_label=None, sample_weight=None):
    fps, tps = dumb_binary_clf_curve(
        y_true, y_score, thresholds, pos_label=pos_label, sample_weight=sample_weight
    )

    # print(thresholds.shape)
    # print(fps.shape)
    # print(tps.shape)

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[0] == 0 <=> y_true == all negative labels
    if tps[0] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[0]

    # reverse the outputs so recall is decreasing
    return np.hstack((precision, 1)), np.hstack((recall, 0))
