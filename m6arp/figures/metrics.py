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

train_cache = {}
test_cache = {}

for site in [8078, 8974, 8988]:
    Xy_df = get_randomized_data(site)
    X_train, y_train, X_test, y_test = test_train_split(Xy_df)
    train_cache[site] = (X_train.copy(), y_train.copy())
    test_cache[site] = (X_test.copy(), y_test.copy())
    print(f"saved data for {site}")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, model_site in enumerate([8078, 8974, 8988]):
    X_train, y_train = train_cache[model_site]
    classifier = kim.interface.trained_classifier(X_train, y_train)

    print(f"Coefficients and intercept for model{model_site}")
    print(classifier.coef_)
    print(classifier.intercept_)

    for j, test_site in enumerate([8078, 8974, 8988]):
        X_test, y_test = test_cache[test_site]
        # y_pred = classifier.predict(X_test)

        metrics.plot_confusion_matrix(classifier, X_test, y_test, ax=axes[i, j])
        axes[i, j].set_title(f"model{model_site} at {test_site}")

fig.savefig("confusion_matrix.png")
