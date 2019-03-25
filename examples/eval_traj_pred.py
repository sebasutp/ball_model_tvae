
import traj_pred.trajectory as traj
import traj_pred.utils as utils
import numpy as np
import matplotlib.pyplot as plt

import argparse

def flatten_list(L):
    ans = []
    for l in L:
        ans.extend(l)
    return ans

def main(args):
    with open(args.data,'rb') as f:
        data = np.load(f, encoding='latin1')
        X = data['X']
        Times = data['Times']
    N = len(Times) if not args.N else args.N
    Times = [t - t[0] for t in Times]
    pred = traj.load_model(args.model)
    m = traj.traj_pred_error(pred, Times[0:N], X[0:N], nobs=args.nobs, noise=args.noise)
    m['avg_err'] = np.mean( flatten_list(m['distances']) )
    m['avg_mlh'] = np.mean( flatten_list(m['c_marg']) )
    mean_dist, std_dist = traj.comp_traj_dist(Times[0:N], m['distances'], length=args.length, deltaT=args.dt)
    mean_llh, std_llh = traj.comp_traj_dist(Times[0:N], m['c_marg'], length=args.length, deltaT=args.dt)
    m['mean_dist'] = mean_dist.reshape(-1)
    m['std_dist'] = std_dist.reshape(-1)
    m['mean_llh'] = mean_llh.reshape(-1)
    m['std_llh'] = std_llh.reshape(-1)
    m['avg_latency'] = np.mean(m['latencies'])
    m['model'] = args.model 
    m['data'] = args.data
    m['dt'] = args.dt
    m['noise'] = args.noise, 
    m['length'] = args.length
    m['N'] = args.N
    m['nobs'] = args.nobs 
    with open(args.result,'wb') as f:
        np.savez(f, **m)
    print("Results successfully saved. avg_dist={}".format(m['avg_err']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="File with the normalized training/validation data")
    parser.add_argument('model', help="Path where the model is stored")
    parser.add_argument('result', help="File where the resulting computations are stored")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Delta Time")
    parser.add_argument('--noise', type=float, default=0.01, help="Sensor noise")
    parser.add_argument('--length', type=int, default=100, help="Cut-Length of the time series")
    parser.add_argument('--N', type=int, help="Number of instances in which to evaluate performance")
    parser.add_argument('--nobs', type=int, default=30, help="Number of observations given to the model before asking for predictions")
    args = parser.parse_args()
    main(args)
