import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics

import utils

###############################################################################
#                         experimental roc curve stuff                        #
###############################################################################

NUMBER_OF_FOLDS = 5
folder = model_selection.StratifiedKFold(NUMBER_OF_FOLDS, shuffle=True, random_state=855)
small_domain = np.linspace(0.0, 1.0, 100)

def moving_average(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def smooth_plot(ax, x_points, y_points, label):
    small_codomain = np.interp(small_domain, x_points, y_points)
    small_codomain = np.r_[small_codomain[0], moving_average(small_codomain), small_codomain[-1]]

    ax.plot(
        small_domain,
        small_codomain,
        lw=lw,
        label=label,
    )

def plot_smooth_roc_for_site(ax, decision_function, X_test, y_test, label=None):
    y_score = decision_function(X_test)

    fpr_arrs = []
    tpr_arrs = []

    # just using the folder to build a roc on four fifths of the data, then
    # averaging them
    for d0, _ in folder.split(X_test, y_test):
        y_d0, y_score_d0 = y_test.iloc[d0], y_score.iloc[d0]

        fpr, tpr, thresholds = metrics.roc_curve(y_d0, y_score_d0, drop_intermediate=False)

        fpr_arrs.append(fpr)
        tpr_arrs.append(tpr)

    final_fpr_arr = np.concatenate(fpr_arrs)
    final_tpr_arr = np.concatenate(tpr_arrs)

    i = np.argsort(final_fpr_arr)
    final_fpr_arr = final_fpr_arr[i]
    final_tpr_arr = final_tpr_arr[i]
    roc_auc = metrics.auc(final_fpr_arr, final_tpr_arr)

    smooth_plot(ax, final_fpr_arr, final_tpr_arr, label)

def plot_smooth_prc_for_site(ax, decision_function, X_test, y_test, label=None):
    y_score = decision_function(X_test)

    precision_arrs = []
    recall_arrs = []
    avg_precs = []

    # just using the folder to build a roc on four fifths of the data, then
    # averaging them
    for d0, _ in folder.split(X_test, y_test):
        y_d0, y_score_d0 = X_test.iloc[d0, :], y_test.iloc[d0], y_score.iloc[d0]

        precision, recall, thresholds = metrics.precision_recall_curve(y_d0, y_score_d0)
        avg_prec = metrics.average_precision_score(y_d0, y_score_d0)

        precision_arrs.append(precision)
        recall_arrs.append(recall)
        avg_precs.append(avg_prec)

    final_recall_arr = np.concatenate(recall_arrs)
    final_precision_arr = np.concatenate(precision_arrs)

    i = np.argsort(final_recall_arr)
    final_recall_arr = final_recall_arr[i]
    final_precision_arr = final_precision_arr[i]

    final_avg_prec = np.mean(avg_precs)

    if label is not None:
        label = label + f" (AP = %0.2f)" % final_avg_prec
    else:
        label = f"AP = %0.2f" % final_avg_prec

    smooth_plot(ax, final_recall_arr, final_precision_arr, label)

def plot_smooth_roc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    y_score = y_hat

    fpr_arrs = []
    tpr_arrs = []

    # just using the folder to build a roc on four fifths of the data, then
    # averaging them
    for d0, _ in folder.split(np.zeros_like(y_test), y_test):
        y_d0, y_score_d0 = y_test[d0], y_score[d0]

        fpr, tpr, thresholds = metrics.roc_curve(y_d0, y_score_d0, drop_intermediate=False)

        fpr_arrs.append(fpr)
        tpr_arrs.append(tpr)

    final_fpr_arr = np.concatenate(fpr_arrs)
    final_tpr_arr = np.concatenate(tpr_arrs)

    i = np.argsort(final_fpr_arr)
    final_fpr_arr = final_fpr_arr[i]
    final_tpr_arr = final_tpr_arr[i]
    roc_auc = metrics.auc(final_fpr_arr, final_tpr_arr)

    if label is not None:
        label = label + f" (AUC = %0.2f)" % roc_auc
    else:
        label = f"AUC = %0.2f" % roc_auc

    smooth_plot(ax, final_fpr_arr, final_tpr_arr, label)

def plot_smooth_prc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    y_score = y_hat

    precision_arrs = []
    recall_arrs = []
    avg_precs = []

    # just using the folder to build a roc on four fifths of the data, then
    # averaging them
    for d0, _ in folder.split(np.zeros_like(y_test), y_test):
        y_d0, y_score_d0 = y_test[d0], y_score[d0]

        precision, recall, thresholds = metrics.precision_recall_curve(y_d0, y_score_d0)
        avg_prec = metrics.average_precision_score(y_d0, y_score_d0)

        precision_arrs.append(precision)
        recall_arrs.append(recall)
        avg_precs.append(avg_prec)


    final_recall_arr = np.concatenate(recall_arrs)
    final_precision_arr = np.concatenate(precision_arrs)

    i = np.argsort(final_recall_arr)
    final_recall_arr = final_recall_arr[i]
    final_precision_arr = final_precision_arr[i]

    final_avg_prec = np.mean(avg_precs)

    if label is not None:
        label = label + f" (AP = %0.2f)" % final_avg_prec
    else:
        label = f"AP = %0.2f" % final_avg_prec

    smooth_plot(ax, final_recall_arr, final_precision_arr, label)

###############################################################################
#                              plotting all rocs                              #
###############################################################################

lw = 2

def plot_model_roc_for_site(ax, decision_function, X_test, y_test):
    y_hat = decision_function(X_test)

    fpr_arr, tpr_arr, thresholds = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr_arr, tpr_arr)

    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % roc_auc,
    )

def plot_roc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    fpr_arr, tpr_arr, thresholds = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr_arr, tpr_arr)

    if label is not None:
        label = label + f" (AUC = %0.2f)" % roc_auc
    else:
        label = f"AUC = %0.2f" % roc_auc

    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=lw,
        label=label,
    )

def plot_model_prc_for_site(ax, decision_function, X_test, y_test):
    y_hat = decision_function(X_test)

    precision_arr, recall_arr, thresholds = metrics.precision_recall_curve(y_test, y_hat)
    avg_prec = metrics.average_precision_score(y_test, y_hat)

    ax.plot(
        recall_arr,
        precision_arr,
        lw=lw,
        label=f"AP = %0.2f" % avg_prec,
    )

def plot_prc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    precision_arr, recall_arr, thresholds = metrics.precision_recall_curve(y_test, y_hat)
    avg_prec = metrics.average_precision_score(y_test, y_hat)

    if label is not None:
        label = label + f" (AP = %0.2f)" % avg_prec
    else:
        label = f"AP = %0.2f" % avg_prec

    ax.plot(
        recall_arr,
        precision_arr,
        lw=lw,
        label=label,
    )

def plot_rocs_for_site(ax, decision_functions, X_test, y_test):
    for df in decision_functions:
        plot_model_roc_for_site(ax, df, X_test, y_test)

    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

def plot_prcs_for_site(ax, decision_functions, X_test, y_test):
    for df in decision_functions:
        plot_model_prc_for_site(ax, df, X_test, y_test)

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    # ax.set_title("Precision-Recall Curve")
