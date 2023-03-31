import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics
from loading import *

import kim.interface
import utils

NUMBER_OF_FOLDS = 5
folder = model_selection.StratifiedKFold(NUMBER_OF_FOLDS, shuffle=True, random_state=855)

roc_fig, roc_axes = plt.subplots(1, 3, figsize=(15, 5))
prc_fig, prc_axes = plt.subplots(1, 3, figsize=(15, 5))

# Get data
for fig_col, site in enumerate([8078, 8974, 8988]):
    read_dirs = LABELLED_DATA_LIST[site]["read_dirs"]
    Xy_df = get_randomized_data(site)
    X, y = separate_data_and_labels(Xy_df)

    roc_ax = roc_axes[fig_col]
    prc_ax = prc_axes[fig_col]

    for model in ["kim", "m6anet", "nanom6a"]:
        tpr_arrs = []
        precision_arrs = []
        avg_precs = []
        base_fpr = None
        base_recall = None

        if model != "kim":
            with open(f"./{model}/{site}-pred_and_test.pickle", "rb") as pred_and_test_f:
                pred_and_test = pickle.load(pred_and_test_f)
                y_pred = pred_and_test.probability_modified.astype("float")
                y_test = pred_and_test.positive.astype("float")
        else:
            y_test = y

        for i, (train, test) in enumerate(folder.split(np.zeros_like(y_test), y_test)):
            if model == "kim":
                local_X_train = X.iloc[train]
                local_y_train = y.iloc[train]
                dec_func = kim.interface.train(local_X_train, local_y_train)
                local_X_pred = X.iloc[test]
                local_y_test = y.iloc[test]
                local_y_pred = dec_func(local_X_pred)
                model_label = f"model{site}"
            else:
                local_y_test = y_test[test]
                local_y_pred = y_pred[test]
                model_label = model

            fold_fpr, fold_tpr, fold_roc_thresholds = metrics.roc_curve(local_y_test, local_y_pred)
            fold_precision, fold_recall, fold_prc_thresholds = metrics.precision_recall_curve(local_y_test, local_y_pred)

            if base_fpr is None and base_recall is None:
                base_fpr = fold_fpr[::2]
                base_recall = fold_recall[::2]

            fold_interp_tpr = np.interp(base_fpr, fold_fpr, fold_tpr)
            tpr_arrs.append(fold_interp_tpr)

            fold_interp_precision = np.interp(base_recall, fold_recall[::-1], fold_precision[::-1])
            precision_arrs.append(fold_interp_precision)

            avg_prec = metrics.average_precision_score(local_y_test, local_y_pred)
            avg_precs.append(avg_prec)


        final_fpr_arr = base_fpr
        final_fpr_arr = np.hstack((final_fpr_arr, 1))
        final_tpr_arr = np.mean(np.stack(tpr_arrs), axis=0)
        final_tpr_arr = np.hstack((final_tpr_arr, 1))

        lw = 2
        roc_auc = round(metrics.auc(final_fpr_arr, final_tpr_arr), 2)

        roc_ax.plot(
            final_fpr_arr,
            final_tpr_arr,
            lw=lw,
            label=f"{model_label} ROC curve (area = %0.2f)" % roc_auc,
        )

        final_recall_arr = base_recall
        final_recall_arr = np.hstack((final_recall_arr, 0))
        final_precision_arr = np.mean(np.stack(precision_arrs), axis=0)
        final_precision_arr = np.hstack((final_precision_arr, 1))
        final_avg_prec = round(np.mean(np.array(avg_precs)), 2)

        prc_ax.plot(
            final_recall_arr,
            final_precision_arr,
            lw=lw,
            label=f"{model_label} AP = %0.2f" % final_avg_prec,
        )
        print(f"finished with {model}")

    roc_ax.legend(loc='lower right')
    prc_ax.legend(loc='lower left')

roc_fig.suptitle("Receiver Operating Characteristic Curves")
roc_fig.savefig(f"multi_model_rocs_smoothed.png")

prc_fig.suptitle("Precision-Recall Curves")
prc_fig.savefig(f"multi_model_prcs_smoothed.png")

# roc_fig.savefig(f"./smoothed-multi-model-roc-curve-{site}.png")
# prc_fig.savefig(f"./smoothed-multi-model-prc-curve-{site}.png")
