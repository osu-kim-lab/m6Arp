#!/usr/bin/env python
import shlex
import sys
import argparse
import pathlib
import pandas as pd

from m6arp.models.ours import interface as m6arp
from m6arp.loading import load_input_csv

def parse_args():
    parser = argparse.ArgumentParser(description='Run a trained model.')
    parser.add_argument('--model',
                        required=True,
                        type=pathlib.Path,
                        help='A pickle containing a trained model.')

    parser.add_argument('--site',
                        required=True,
                        type=int,
                        help='The site at which the model should make a prediction.')

    parser.add_argument('--csv',
                        required=True,
                        type=pathlib.Path,
                        help='The csv file with reads and values per position for making a prediction.')

    parser.add_argument('--output',
                        type=pathlib.Path,
                        help='The path at which to ouput csv file of predictions.')


    args = parser.parse_args()
    return args


def main() -> int:
    """Run model with given arguments."""
    args = parse_args()
    model = m6arp.from_pickle(args.model)
    df = load_input_csv(args.csv, args.site)
    X = df.values
    y_hat = model.predict(X)

    predictions = pd.Series(y_hat, index=df.index, name="predictions")

    if args.output:
        predictions.to_csv(args.output)
    else:
        print(predictions)

    return 0

if __name__ == '__main__':
    sys.exit(main())
