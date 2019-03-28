
import traj_pred.trajectory as traj
import traj_pred.utils as utils
import numpy as np
import matplotlib.pyplot as plt

import argparse

def main(args):
    with open(args.data,'rb') as f:
        data = np.load(f, encoding='latin1')
        X = data['X']
        Times = data['Times']
    pred = traj.load_model(args.model)
    D = 3

    pred_dur = int(round(args.t/args.dt))
    #pred_times = np.arange(0,pred_dur)*args.dt
    for i,tabs in enumerate(Times):
        t = tabs - tabs[0]
        T = len(t)
        if T >= pred_dur: continue

        x = np.array(X[i])
        prev_times = t[T-args.n:T]
        prev_obs = x[T-args.n:T]
        pred_times = np.arange(T,pred_dur)*args.dt

        print(i ,len(prev_times), len(pred_times))
        try:
            pred_mean, pred_cov = pred.traj_dist(prev_times, prev_obs, pred_times)
        except:
            print("Error processing {}. Skipping".format(i))
            continue
        pred_std = utils.cov_to_std(pred_cov)
        if False:
            plt.figure(1)
            for d in range(D):
                plt.subplot(D,1,d+1)
                plt.plot(t, x[:,d], 'r.')
                y_test_mean = pred_mean[:,d]
                y_test_std = pred_std[:,d] 
                plt.plot(pred_times, y_test_mean, 'b')
                plt.fill_between(pred_times, y_test_mean - 2*y_test_std, y_test_mean + 2*y_test_std,
                        color='b', alpha=0.5)
            plt.show()
        Times[i] = np.concatenate((Times[i],Times[i][0]+pred_times), axis=0)
        X[i] = np.concatenate((X[i], pred_mean), axis=0)

    with open(args.out,"wb") as f:
        np.savez(f, Times=Times, X=X)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="File with the normalized training/validation data")
    parser.add_argument('model', help="Model used to complete trajectories")
    parser.add_argument('--n', type=int, default=20, help="Last k observations to use to feed the model")
    parser.add_argument('--t', type=float, default=1.5, help="Time all observations are completed to")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Delta Time")
    parser.add_argument('--out', default="out.npz", help="Where the output is stored")
    args = parser.parse_args()
    main(args)
