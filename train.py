#!/usr/bin/env python
import shlex
import sys
import argparse
import pathlib
import pandas as pd
import pickle

from m6arp.models.ours import interface as m6arp
from m6arp.loading import get_randomized_data, separate_data_and_labels

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--site',
                        required=True,
                        type=int,
                        help='The site for which the model should learn to make predictions.')

    parser.add_argument('--positive-csv',
                        required=True,
                        type=pathlib.Path,
                        help='The csv file with positive example reads and values per position for training.')

    parser.add_argument('--negative-csv',
                        required=True,
                        type=pathlib.Path,
                        help='The csv file with negative example reads and values per position for training.')

    parser.add_argument('--output',
                        type=pathlib.Path,
                        help='The path at which to output the trained model.')

    args = parser.parse_args()
    return args


def main() -> int:
    """Run model with given arguments."""
    args = parse_args()
    Xy_df = get_randomized_data(args.site, args.positive_csv, args.negative_csv)
    X, y = separate_data_and_labels(Xy_df)
    classifier = m6arp.trained_classifier(X, y)

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(classifier, f)
    else:
        print(classifier)

    return 0

if __name__ == '__main__':
    sys.exit(main())
