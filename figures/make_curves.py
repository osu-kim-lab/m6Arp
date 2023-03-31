import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics
from loading import *
from plotting import *

import kim.interface

other_models = ["m6anet", "nanom6a"]
            
train_cache = {}
test_cache = {}

for site in [8078, 8974, 8988]:
    Xy_df = get_randomized_data(site)
    X_train, y_train, X_test, y_test = test_train_split(Xy_df)
    train_cache[site] = (X_train.copy(), y_train.copy())
    test_cache[site] = (X_test.copy(), y_test.copy())
    print(f"saved data for {site}")

fig, axes = plt.subplots(3, 3, figsize=(13, 13))

for i, model_site in enumerate([8078, 8974, 8988]):
    X_train, y_train = train_cache[model_site]
    df = kim.interface.train(X_train, y_train)
    dfs = [df]
    for j, test_site in enumerate([8078, 8974, 8988]):
        X_test, y_test = test_cache[test_site]
        plot_rocs_for_site(axes[i, j], dfs, X_test, y_test)

        for model in other_models:
            with open(f"./{model}/{test_site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{test_site}-test.pickle", "rb") as test_f:
                model_y_pred_at_current_test_site = pickle.load(pred_f)
                model_y_test_at_current_test_site = pickle.load(test_f)

                if model_y_pred_at_current_test_site.dtype == np.object:
                    model_y_pred_at_current_test_site = model_y_pred_at_current_test_site.astype(np.float64)
                    model_y_test_at_current_test_site = model_y_test_at_current_test_site.astype(np.float64)

                model_y_pred_at_current_test_site = np.nan_to_num(model_y_pred_at_current_test_site)
                model_y_test_at_current_test_site = np.nan_to_num(model_y_test_at_current_test_site)

                plot_roc_for_site_by_y_hat(axes[i, j],
                                           model_y_pred_at_current_test_site,
                                           model_y_test_at_current_test_site,
                                           label=model)

        axes[i, j].legend(loc="lower right")

pad = 5 # in points

cols = ['At Site {}'.format(site) for site in [8078, 8974, 8988]]
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

rows = ['With model{}'.format(site) for site in [8078, 8974, 8988]]
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.suptitle("Receiver Operating Characteristic Curves")
fig.savefig(f"all_rocs.png")



fig, axes = plt.subplots(3, 3, figsize=(13, 13))

for i, model_site in enumerate([8078, 8974, 8988]):
    X_train, y_train = train_cache[model_site]
    df = kim.interface.train(X_train, y_train)
    dfs = [df]
    for j, test_site in enumerate([8078, 8974, 8988]):
        X_test, y_test = test_cache[test_site]
        plot_prcs_for_site(axes[i, j], dfs, X_test, y_test)

        for model in other_models:
            with open(f"./{model}/{test_site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{test_site}-test.pickle", "rb") as test_f:
                model_y_pred_at_current_test_site = pickle.load(pred_f)
                model_y_test_at_current_test_site = pickle.load(test_f)

                if model_y_pred_at_current_test_site.dtype == np.object:
                    model_y_pred_at_current_test_site = model_y_pred_at_current_test_site.astype(np.float64)
                    model_y_test_at_current_test_site = model_y_test_at_current_test_site.astype(np.float64)

                model_y_pred_at_current_test_site = np.nan_to_num(model_y_pred_at_current_test_site)
                model_y_test_at_current_test_site = np.nan_to_num(model_y_test_at_current_test_site)

                plot_prc_for_site_by_y_hat(axes[i, j],
                                           model_y_pred_at_current_test_site,
                                           model_y_test_at_current_test_site,
                                           label=model)

        axes[i, j].legend(loc="lower left")

pad = 5 # in points

cols = ['At Site {}'.format(site) for site in [8078, 8974, 8988]]
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

rows = ['With model{}'.format(site) for site in [8078, 8974, 8988]]
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.suptitle("Precision Recall Curves")
fig.savefig(f"all_prcs.png")
