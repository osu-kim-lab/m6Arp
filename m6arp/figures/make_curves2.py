import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics
from loading import *
from plotting import *

###############################################################################
#                                  main code                                  #
###############################################################################

roc_fig, roc_axes = plt.subplots(1, 3, figsize=(15, 5))
prc_fig, prc_axes = plt.subplots(1, 3, figsize=(15, 5))

for fig_col, site in enumerate([8078, 8974, 8988]):
    roc_ax = roc_axes[fig_col]
    prc_ax = prc_axes[fig_col]
            
    for model in ["kim", "m6anet", "nanom6a"]:
        with open(f"./{model}/{site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{site}-test.pickle", "rb") as test_f:
            y_pred = pickle.load(pred_f)
            y_test = pickle.load(test_f)

        if y_pred.dtype == np.object:
            y_pred = y_pred.astype(np.float64)
            y_test = y_test.astype(np.float64)
            
        y_pred = np.nan_to_num(y_pred)
        y_test = np.nan_to_num(y_test)

        if model == "kim":
            model = f"model{site}"

        # plot_roc_for_site_by_y_hat(roc_ax, y_pred, y_test, label=model)
        # plot_prc_for_site_by_y_hat(prc_ax, y_pred, y_test, label=model)
        plot_smooth_roc_for_site_by_y_hat(roc_ax, y_pred, y_test, label=model)
        plot_smooth_prc_for_site_by_y_hat(prc_ax, y_pred, y_test, label=model)
        print(f"finished with {model}")

    roc_ax.legend(loc='lower right')
    prc_ax.legend(loc='lower left')

        
roc_fig.suptitle("Receiver Operating Characteristic Curves")
roc_fig.savefig(f"multi_model_rocs_smoothed.png")

prc_fig.suptitle("Precision-Recall Curves")
prc_fig.savefig(f"multi_model_prcs_smoothed.png")
