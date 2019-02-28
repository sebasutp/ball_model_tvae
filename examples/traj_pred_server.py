
import traj_pred.trajectory as traj
import traj_pred.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import time
import zmq
import slpp_pb2
import argparse
import struct

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
        req_msg = skt.recv()
        t0 = time.time()
        req.ParseFromString(req_msg)
        prev_times = pbtensor2numpy(req.prev_times)
        prev_obs = pbtensor2numpy(req.prev_obs)
        pred_times = pbtensor2numpy(req.pred_times)
        t1 = time.time()

        pred_mean, pred_cov = pred.traj_dist(prev_times, prev_obs, pred_times)

        t2 = time.time()
        numpy2pbtensor(res.means, pred_mean)
        numpy2pbtensor(res.covs, pred_cov)
        res_msg = res.SerializeToString()
        t3 = time.time()
        skt.send(res_msg)
        print("Responding request. Decoding: {} ms, Computing: {} ms, Encoding: {} ms, Total: {} ms".format(
            1000*(t1-t0), 1000*(t2-t1), 1000*(t3-t2), 1000*(t3-t0)))
        if args.msglog:
            nbstr = struct.Struct('<I')
            with open(args.msglog,'ab') as f:
                f.write(nbstr.pack(len(req_msg)))
                f.write(req_msg)
                f.write(nbstr.pack(len(res_msg)))
                f.write(res_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="Path where the model is stored")
    parser.add_argument('--port', default="7671", help="Number of instances to plot")
    parser.add_argument('--msglog', help="File where all the queries and responses are stored")
    args = parser.parse_args()
    main(args)
