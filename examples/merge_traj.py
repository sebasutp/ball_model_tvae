""" Merges separate trajectory files
"""

import argparse
import os
import json
import time

import numpy as np


def main(args):
    Xreal = []
    Times = []
    for x in args.data:
        with open(x,'rb') as f:
            data = np.load(f, encoding='latin1')
            Xreal.extend([x for x in data['X'] if len(x)>args.min_duration and len(x)<args.max_duration])
            Times.extend([x for x in data['Times'] if len(x)>args.min_duration and len(x)<args.max_duration])
    print("Loaded {} ball trajectories".format(len(Times)))
    with open(args.out,'wb') as f:
        np.savez(f, X=Xreal, Times=Times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs='+', help="Files with the stored trajectories")
    parser.add_argument('--out', default='out.npz', help="File where the processed data is saved")
    parser.add_argument('--min_duration', type=int, default=100, help="Minimal length of time series to consider. Too short are usually outliers")
    parser.add_argument('--max_duration', type=int, default=500, help="Maximal length of time series to consider. Too long are usually outliers")
    args = parser.parse_args()
    main(args)
