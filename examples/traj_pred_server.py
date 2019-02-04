
import traj_pred.trajectory as traj
import traj_pred.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import zmq
import slpp_pb2
import argparse

def pbtensor2numpy(pb_tensor):
    shape = np.array(pb_tensor.shape, dtype=np.int32)
    tensor = np.array(pb_tensor.data).reshape(shape)
    return tensor

def numpy2pbtensor(pb_tensor, tensor):
    pb_tensor.shape[:] = tensor.shape
    pb_tensor.data[:] = tensor.flatten().tolist()

def main(args):
    pred = traj.load_model(args.model)
    ctx = zmq.Context()
    skt = ctx.socket(zmq.REP)
    skt.bind("tcp://*:%s" % args.port)
    req = slpp_pb2.BallPredictRequest()
    res = slpp_pb2.BallPredictResponse()
    while True:
        msg = skt.recv()
        req.ParseFromString(msg)
        prev_times = pbtensor2numpy(req.prev_times)
        prev_obs = pbtensor2numpy(req.prev_obs)
        pred_times = pbtensor2numpy(req.pred_times)

        pred_mean, pred_cov = pred.traj_dist(prev_times, prev_obs, pred_times)
        numpy2pbtensor(res.means, pred_mean)
        numpy2pbtensor(res.covs, pred_cov)
        msg = res.SerializeToString()
        skt.send(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="Path where the model is stored")
    parser.add_argument('--port', default="7671", help="Number of instances to plot")
    args = parser.parse_args()
    main(args)