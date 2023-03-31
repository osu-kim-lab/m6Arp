import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics
from loading import *

import kim.interface
import m6anet.interface
import nanom6a.interface

# Get data
for site in [8078, 8974, 8988]:
    read_dirs = LABELLED_DATA_LIST[site]["read_dirs"]
    Xy_df = get_randomized_data(site)
    X, y = separate_data_and_labels(Xy_df)

    # prepare(*read_dirs, X_test, f"{site}")
    # decision_function = kim.interface.train(X_train, y_train)
    # y_pred = decision_function(X_test.values)

    # with open(f"./kim/{site}-predictions.pickle", "wb") as pred_f, open(f"./kim/{site}-test.pickle", "wb") as test_f:
    #     pickle.dump(y_pred, pred_f)
    #     pickle.dump(y_test.values, test_f)

    ###########################################################################
    #              step 1: prepare inputs for m6anet and nanom6a              #
    ###########################################################################


    # m6anet.interface.prepare(*read_dirs, X, f"{site}")

    ###########################################################################
    #                 step 2: run m6anet and nanom6a pipelines                #
    ###########################################################################
    # done 2023-02-07

    ###########################################################################
    #    step 3: read m6anet and nanom6a output and dump in friendly format   #
    ###########################################################################

    y_test = y
    prob_read_name_df = m6anet.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/data.indiv_proba.csv.gz",
                                                    f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/alignment-summary.txt",
                                                    site)
    y_pred_and_test = prob_read_name_df.join(y_test)
    y_pred_and_test.columns = ["probability_modified", "positive"]
    y_pred = y_pred_and_test["probability_modified"].values
    y_test_values = y_pred_and_test["positive"].values

    with open(f"./m6anet/{site}-predictions.pickle", "wb") as pred_f, open(f"./m6anet/{site}-test.pickle", "wb") as test_f, open(f"./m6anet/{site}-pred_and_test.pickle", "wb") as pred_and_test_f:
        pickle.dump(y_pred, pred_f)
        pickle.dump(y_test.values, test_f)
        pickle.dump(y_pred_and_test, pred_and_test_f)

    read_name_prob_df = nanom6a.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/results/{site}/prediction_results/extract.reference.bed12",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/results/{site}/prediction_results/sam_parse2.txt",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/results/{site}/prediction_results/total_mod.tsv",
                                                    site)

    y_pred_and_test = read_name_prob_df.join(y_test)
    y_pred_and_test.columns = ["probability_modified", "positive"]
    y_pred = y_pred_and_test["probability_modified"].values
    y_test_vals = y_pred_and_test["positive"].values

    with open(f"./nanom6a/{site}-predictions.pickle", "wb") as pred_f, open(f"./nanom6a/{site}-test.pickle", "wb") as test_f, open(f"./nanom6a/{site}-pred_and_test.pickle", "wb") as pred_and_test_f:
        pickle.dump(y_pred, pred_f)
        pickle.dump(y_test_vals, test_f)
        pickle.dump(y_pred_and_test, pred_and_test_f)
