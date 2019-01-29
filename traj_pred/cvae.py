
import numpy as np
import keras
import tensorflow as tf

class Trajectory:
    """ Trajectory modeling with CVAE

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __init__(self, config):
        pass

    def traj_llh(self, times, obs):
        pass

    def traj_dist(self, prev_times, prev_obs, pred_times):
        pass

class TSAug(keras.utils.Sequence):
    """ Augments the data for time series modeling
    """

    def __init__(self, X, Xobs, batch_size=32, ds_mult=16, shuffle=True):
        self.X = X
        self.Xobs = Xobs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ds_mult = ds_mult
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.repeat(np.arange(len(self.X)), self.ds_mult)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids):
        Y = self.X[list_ids]
        Yobs = self.Xobs[list_ids]
        
        N,T,K = Y.shape        
        ts_lens = np.random.randint(low=0, high=T, size=N)
        is_obs = np.array([np.arange(T) < x for x in ts_lens])
        Xobs = Yobs*is_obs.reshape((self.batch_size,T,1))
        X = Xobs*Y

        return X, Xobs, Y, Yobs

    def __len__(self):
        x = np.floor(len(self.indexes) / self.batch_size)
        return int(x)

    def __getitem__(self, index):
        list_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, Xobs, Y, Yobs = self.__data_generation(list_ids)
        Ymask = np.concatenate((Y,Yobs),axis=-1)
        return [X,Xobs,Y], [Ymask]

def cvae_loss(mu, log_sigma, batch_size):
    def loss(y_mask, y_decoded_mean):
        y = y_mask[:,:,0:-1]
        mask = y_mask[:,:,-1]
        d = y - y_decoded_mean
        d_sq = keras.backend.sum( keras.backend.square(d), axis=-1 )
        d_sq_masked = d_sq * mask
        kl = 0.5 * keras.backend.sum(keras.backend.exp(log_sigma) + 
                keras.backend.square(mu) - 1. - log_sigma)
        rec_loss = keras.backend.sum( d_sq_masked )
        return (rec_loss + kl) / batch_size
    return loss

