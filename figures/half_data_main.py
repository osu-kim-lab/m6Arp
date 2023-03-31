import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics
from loading import *

# Get data
for site in [8078, 8974, 8988]:
    read_dirs = LABELLED_DATA_LIST[site]["read_dirs"]
    Xy_df = get_randomized_data(site)
    X_train, y_train, X_test, y_test = test_train_split(Xy_df)

    # prepare(*read_dirs, X_test, f"{site}")

    # import kim.interface
    # decision_function = kim.interface.train(X_train, y_train)
    # y_pred = decision_function(X_test.values)

    # with open(f"./kim/{site}-predictions.pickle", "wb") as pred_f, open(f"./kim/{site}-test.pickle", "wb") as test_f:
    #     pickle.dump(y_pred, pred_f)
    #     pickle.dump(y_test.values, test_f)

    # import m6anet.interface
    # prob_read_name_df = m6anet.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/data.indiv_proba.csv.gz",
    #                                                 f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/alignment-summary.txt",
    #                                                 site, y_test)
    # y_pred_and_test = prob_read_name_df.join(y_test)

    # y_pred = y_pred_and_test["probability_modified"].values
    # y_test_values = y_pred_and_test["positive"].values

    # with open(f"./m6anet/{site}-predictions.pickle", "wb") as pred_f, open(f"./m6anet/{site}-test.pickle", "wb") as test_f:
    #     pickle.dump(y_pred, pred_f)
    #     pickle.dump(y_test.values, test_f)

    import nanom6a.interface
    read_name_prob_df = nanom6a.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/extract.reference.bed12",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/sam_parse2.txt",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/total_mod.tsv",
                                                    site)

    y_pred_and_test = read_name_prob_df.join(y_test)
    y_pred = y_pred_and_test["probability"].values
    y_test_vals = y_pred_and_test["positive"].values

    with open(f"./nanom6a/{site}-predictions.pickle", "wb") as pred_f, open(f"./nanom6a/{site}-test.pickle", "wb") as test_f:
        pickle.dump(y_pred, pred_f)
        pickle.dump(y_test_vals, test_f)
