
import numpy as np
import keras
import tensorflow as tf
import traj_pred.utils as utils
import json
import pickle
import os
import matplotlib.pyplot as plt


class Trajectory:
    """ Trajectory modeling with CVAE

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __init__(self, encoder, decoder, normalizer=None, samples=30, z_size=16, 
            length=200, deltaT=1.0/180.0, default_Sigma_y=1e2):
        self.encoder = encoder
        self.decoder = decoder
        self.normalizer = normalizer
        self.deltaT = deltaT
        self.length = length
        self.z_size = z_size
        self.samples = samples
        self.default_Sigma_y = default_Sigma_y

    def traj_llh(self, times, obs):
        #TODO: Not implemented yet, think well the math first
        #X, Xobs = utils.encode_fixed_dt([times],[obs], self.length,self.deltaT)
        return 0.0

    def traj_dist(self, prev_times, prev_obs, pred_times):
        batch_size = 1
        #1) First normalize and then encode. Very important!
        Xn, Xobs = utils.encode_fixed_dt([prev_times],[self.normalizer.transform(prev_obs)], 
                self.length, self.deltaT)
        z = np.random.normal(loc=0.0, scale=1.0, size=(self.samples,batch_size,self.z_size))
        y_n = np.array([self.decoder.predict([Xn,Xobs,z[i]]) for i in range(self.samples)])
        y = utils.apply_scaler(self.normalizer.inverse_transform, y_n)
        ixs = [int(round((x - prev_times[0])/self.deltaT)) for x in pred_times]
        means = [] 
        covs = []
        for i in ixs:
            if i < self.length:
                y_mu = np.mean(y[:,0,i,:], axis=0)
                y_Sigma = np.cov(y[:,0,i,:], rowvar=False)
                bias_scale = self.samples/(self.samples-1)
                y_Sigma = bias_scale*y_Sigma
            else:
                y_mu = np.mean(y[:,0,-1,:], axis=0)
                y_Sigma = self.default_Sigma_y*np.eye(y.shape[-1])
            means.append(y_mu)
            covs.append(y_Sigma)
        return np.array(means), np.array(covs)

def load_traj_cvae(path):
    """ Loads trained trajectory CVAE
    """
    extra = json.load( open(os.path.join(path,'conf.json'), 'r') )
    extra.setdefault('deltaT', 1.0/180.0)
    extra.setdefault('samples', 30)
    extra.setdefault('default_Sigma_y', 1e2)
    decoder = keras.models.load_model( os.path.join(path,'decoder.h5') )
    encoder = keras.models.load_model( os.path.join(path,'encoder.h5') )
    norm = pickle.load(open(os.path.join(path,'norm.pickle'), 'rb'))
    return Trajectory(encoder, decoder, norm['xscaler'], samples=extra['samples'], 
            z_size=extra['z_size'], length=extra['in_size'], 
            deltaT=extra['deltaT'], 
            default_Sigma_y=extra['default_Sigma_y'])


