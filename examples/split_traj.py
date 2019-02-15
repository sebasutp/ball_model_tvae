""" Split randomly a trajectory file
"""

import argparse
import os
import json
import time

import numpy as np


def main(args):
    with open(args.in_data,'rb') as f:
        data = np.load(f, encoding='latin1')
        Times = data['Times']
        X = data['X']
    N = len(Times)
    ix = np.random.permutation(N)
    last_ix = 0
    for i,p in enumerate(args.p):
        next_ix = last_ix + int(p*N)
        if next_ix > N:
            print("Warning with index computation: The provided percentages might not sum to one")
        nTimes = [Times[ix[i]] for i in range(last_ix,next_ix)]
        nX = [X[ix[i]] for i in range(last_ix,next_ix)]
        fname = args.out_fmt.format(i)
        with open(fname,'wb') as out_f:
            np.savez(out_f, X = nX, Times = nTimes)
        print("{} trajectories saved to {}".format(next_ix-last_ix, fname))
        last_ix = next_ix
    if last_ix < N:
        print("Warning: {} trajectory was not used".format(N-last_ix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('in_data', help="File with the stored trajectories")
    parser.add_argument('--p', nargs='+', type=float, help="Percentages of the trajectories for each output file")
    parser.add_argument('--out_fmt', default='split{}.npz', help="File where the processed data is saved")
    args = parser.parse_args()
    main(args)
