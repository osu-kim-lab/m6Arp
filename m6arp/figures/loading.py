import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics

###############################################################################
#                                     data                                    #
###############################################################################

LABELLED_DATA_LIST = {
    8078: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8079pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8079pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
    8974: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8975pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
    8988: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8989pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8989pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
}

RANGE_OF_BASES_TO_INCLUDE = (-4, 1) # inclusive

def load_csv(filepath):
    '''Load per-read stats from a CSV file into a Pandas DataFrame'''
    retval = pd.read_csv(filepath, header=0, index_col=0).rename_axis('pos_0b', axis=1)
    retval.columns = retval.columns.astype(int)
    retval.index = [x[2:-1] if x[0:2] == "b'" and x[-1] == "'" else x for x in retval.index]
    retval = retval.rename_axis('read_id')
    return retval

def longify(df):
    '''Convert dataframe output of load_csv to a long format'''
    return df.stack().rename('pval').reset_index()

def prepare_labelled_data(site, labelled_data_list=LABELLED_DATA_LIST):
    """The Kim Model makes a prediction about a modified site based on the Tombo
    MSC values surrounding that site in the read."""

    data_file_pair = labelled_data_list[site]
    to_concat = []
    for filepath, positive in [(data_file_pair['positive'], True),
                               (data_file_pair['negative'], False)]:
        # read data and create useful columns
        df = ( # pylint: disable=invalid-name
            longify(load_csv(filepath))
            .assign(
                positive = positive,
                site_0b = site
            ).assign(
                delta = lambda x: x['pos_0b'] - x['site_0b']
            )
        )
        # remove unnecessary positions
        df = df.loc[
            (RANGE_OF_BASES_TO_INCLUDE[0] <= df['delta'])
            & (df['delta'] <= RANGE_OF_BASES_TO_INCLUDE[1])
        ]
        to_concat.append(df)
        del df

    labelled_df = pd.concat(to_concat)
    pivoted_labelled_df = labelled_df.pivot(
        index=['positive', 'read_id'],
        columns='delta',
        values='pval'
    ).dropna()

    return pivoted_labelled_df

def get_randomized_data(site):
    pivoted_labelled_df = prepare_labelled_data(site)
    Xy_df = pivoted_labelled_df.reset_index(level=0).astype("float64")
    Xy_df = Xy_df[[*Xy_df.columns[1:], Xy_df.columns[0]]]

    rs = RandomState(MT19937(SeedSequence(123456789)))
    Xy_df = Xy_df.sample(frac = 1, random_state=rs)

    return Xy_df

def separate_data_and_labels(Xy_df):
    col_point = Xy_df.shape[1] - 1
    X = Xy_df.iloc[:, 0:col_point]
    y = Xy_df.iloc[:, col_point]
    return X, y

def test_train_split(Xy_df):
    half_point = Xy_df.shape[0] // 2
    col_point = Xy_df.shape[1] - 1

    Xy_train = Xy_df.iloc[0:half_point, :]
    X_train = Xy_train.iloc[:, 0:col_point]
    y_train = Xy_train.iloc[:, col_point]

    Xy_test = Xy_df.iloc[0:half_point, :]
    X_test = Xy_test.iloc[:, 0:col_point]
    y_test = Xy_test.iloc[:, col_point]

    return X_train, y_train, X_test, y_test
