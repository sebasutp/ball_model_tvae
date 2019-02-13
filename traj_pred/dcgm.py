
import numpy as np
import keras
import tensorflow as tf
import traj_pred.utils as utils
import json
import pickle
import os
import matplotlib.pyplot as plt


class Trajectory:
    """ Trajectory modeling with Deep Conditional Generative Model

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

def load_traj_model(path):
    """ Loads trained trajectory model
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


class BatchDCGM(keras.utils.Sequence):
    """ Creates mini-batches for a deep conditional generative model for trajectories

    Given a sequence of pairs (time, X) and a particular deltaT and length, returns a sequence
    of tensors (X,Xobs,Y,Yobs) with the same time length and deltaT.
    """

    def __init__(self, batch_sampler, length, deltaT):
        self.batch_sampler = batch_sampler
        self.length = length
        self.deltaT = deltaT
        self.on_epoch_end()

    def on_epoch_end(self):
        self.batch_sampler.on_epoch_end()

    def __data_generation(self, times, X):
        assert( len(times) == len(X) )
        Y, Yobs = utils.encode_fixed_dt(times, X, self.length, self.deltaT)
       
        N,T,K = Y.shape        
        ts_lens = np.random.randint(low=0, high=T, size=N)
        is_obs = np.array([np.arange(T) < x for x in ts_lens])
        Xobs = Yobs*is_obs.reshape((self.batch_size,T,1))
        X = Xobs*Y

        return X, Xobs, Y, Yobs

    def __len__(self):
        return len(self.batch_sampler)

    def __getitem__(self, index):
        times, X = self.batch_sampler[index]
        X, Xobs, Y, Yobs = self.__data_generation(times, X)
        Ymask = np.concatenate((Y,Yobs),axis=-1)
        return [X,Xobs,Y], [Ymask]

def dcgm_loss(mu, log_sigma, log_sig_y):
    def loss(y_mask, y_decoded_mean):
        y_mask_shape = keras.backend.shape(y_mask)
        batch_size = y_mask_shape[0]
        y = y_mask[:,:,0:-1]
        mask = y_mask[:,:,-1]
        sig_y = keras.backend.exp(log_sig_y)
        d = y - y_decoded_mean
        d_sq = keras.backend.square(d)
        d_mah = tf.math.divide(d_sq, sig_y)
        log_det_sig_y = keras.backend.sum(log_sig_y) #has to be a scalar
        d_mah_sum = keras.backend.sum(d_mah, axis=-1 ) + log_det_sig_y
        d_sq_masked = d_mah_sum * mask
        kl = 0.5 * keras.backend.sum(keras.backend.exp(log_sigma) + 
                keras.backend.square(mu) - 1. - log_sigma)
        rec_loss = 0.5 * keras.backend.sum( d_sq_masked )
        return (rec_loss + kl) / tf.to_float(batch_size)
    return loss

class TrajDCGM:
    """ A deep conditional generative model for trajectory generation
    """

    def __build_graph(self, encoder, cond_generator, log_sig_y, length, D, z_size):
        self.encoder = encoder
        self.cond_generator = cond_generator
        x = keras.layers.Input(shape=(length,D))
        x_obs = keras.layers.Input(shape=(length,1))
        y = keras.layers.Input(shape=(length,D))
        y_obs = keras.layers.Input(shape=(length,1))
        
        #self.z_in = keras.layers.Input(shape=(z_size,))
        mu_z, log_sig_z = encoder([y,y_obs])

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = keras.backend.random_normal(shape=(batch_size, z_size))
            return z_mean + tf.exp(z_log_sigma) * epsilon

        z_sampler = keras.layers.Lambda(sampling, output_shape=(z_size,))
        z = z_sampler([mu_z, log_sig_z])
        y_pred = cond_generator([x,x_obs,z])

        self.full_tree = keras.models.Model(inputs=[x,x_obs,y,y_obs], outputs=[y_pred])
        my_loss = dcgm_loss(mu_z, log_sig_z, log_sig_y)
        self.full_tree.compile(optimizer='adam', loss=my_loss)

    def fit_generator(self, generator, validation_data, epochs, use_multiprocessing, workers, callbacks):
        self.full_tree.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                use_multiprocessing=use_multiprocessing, workers=workers, callbacks=callbacks)

    def __init__(self, encoder, cond_generator, log_sig_y, length, D, z_size):
        """ Constructs a Trajectory Deep Conditional Model

        Parameters
        ----------

        encoder : Model with inputs=[y,y_obs] and outputs=[mu_z, log_sig_z]
        cond_generator : Model with inputs=[x,x_obs,z] and output=[y]
        log_sig_y : Variable representing the sensor noise. If it is trainable, it will be optimized.
        length : Maximum number of time samples of each trajectory
        D : Dimensionality of the observations
        z_size : Dimensionality of the hidden state
        """
        self.__build_graph(encoder, cond_generator, log_sig_y, length, D, z_size)



