
import numpy as np
import matplotlib.pyplot as plt
import slpp_pb2
import argparse
import struct
import traj_pred.utils as utils
from traj_pred_server import pbtensor2numpy, numpy2pbtensor

def main(args):
    nbstr = struct.Struct('<I')
    req = slpp_pb2.BallPredictRequest()
    res = slpp_pb2.BallPredictResponse()

    f = open(args.log,'rb')
    for i in range(args.n):
        tmp = f.read(nbstr.size)
        if not tmp: break
        s1 = nbstr.unpack(tmp)[0]
        req_msg = f.read(s1)
        s2 = nbstr.unpack(f.read(nbstr.size))[0]
        res_msg = f.read(s2)
        req.ParseFromString(req_msg)
        res.ParseFromString(res_msg)

        prev_times = pbtensor2numpy(req.prev_times)
        prev_obs = pbtensor2numpy(req.prev_obs)
        pred_times = pbtensor2numpy(req.pred_times)
        pred_mean = pbtensor2numpy(res.means)
        pred_cov = pbtensor2numpy(res.covs)
        pred_std = utils.cov_to_std(pred_cov)

        plt.figure(1)
        D = pred_mean.shape[1]
        for d in range(D):
            plt.subplot(D,1,d+1)
            plt.plot(prev_times,prev_obs[:,d],'g.')
            y_test_mean = pred_mean[:,d]
            y_test_std = pred_std[:,d] 
            plt.plot(pred_times, y_test_mean, 'b')
            plt.fill_between(pred_times, y_test_mean - 2*y_test_std, y_test_mean + 2*y_test_std,
                    color='b', alpha=0.5)
        plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', help="File with the normalized training/validation data")
    parser.add_argument('n', type=int, help="Number of messages to plot")
    args = parser.parse_args()
    main(args)
