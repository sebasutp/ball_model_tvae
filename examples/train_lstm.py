""" Trains LSTM for time series forecasting
"""

import argparse
import os
import json
import time
import math

import numpy as np
import sklearn.preprocessing as prep
import traj_pred.utils as utils
#import traj_pred.dcgm as dcgm
import traj_pred.lstm as traj_lstm
import keras

def create_lstm(length, D, hidden_size):
    y = keras.layers.Input(shape=(length-1,D), name='y_input')
    y_obs = keras.layers.Input(shape=(length-1,1), name='y_obs_input')
    noise = keras.layers.Input(shape=(length-1,D), name='y_noise')
    
    y_conc = keras.layers.concatenate([y, y_obs])
    my_lstm = keras.layers.LSTM(hidden_size, input_shape=(D+1,), return_sequences=True, return_state=True)
    my_dense = keras.layers.Dense(D, input_shape=(hidden_size,))

    lstm_out, lstm_state_h, lstm_state_c = my_lstm(y_conc)
    td_layer = keras.layers.TimeDistributed(my_dense, input_shape=(length,hidden_size))
    out = td_layer(lstm_out)

    # Test time model
    '''
    lstm_out_test = []
    for i in range(length-1):
        if i == 0:
            lstm_out_i, lstm_state_h_i, lstm_state_c_i = my_lstm(y_conc[i:i+1,:])
        else:
            lstm_out_i, lstm_state_h_i, lstm_state_c_i = my_lstm(y_conc[i:i+1,:], 
                    initial_state=[lstm_state_h_i, lstm_state_c_i])
        lstm_out_test.append(my_dense(lstm_out_i[0]))
    test_model = keras.models.Model(inputs=[y,y_obs,noise], outputs=[lstm_out_test])
    '''


    #rnn = keras.models.Sequential()
    #rnn.add( my_lstm )
    #rnn.add( keras.layers.TimeDistributed(keras.layers.Dense(D)) )
    #out = rnn(y_conc)

    return keras.models.Model(inputs=[y,y_obs], outputs=[out])


def train_ball_lstm(X, Times, Xval, Time_v, length, deltaT, z_size=64, batch_size=64, epochs=100, samples=30):
    x_scaler = utils.train_std_scaler(X)
    x_transform = lambda x: x_scaler.transform( utils.transform_ball_traj(x,(-30,30),((-0.3,-0.3,0.0),(0.3,0.3,0.0))) )
    epoch_size = 1000
    N = len(X)
    D = len(X[0][0])
    ds_mult = int( math.ceil(epoch_size / N) )

    my_mb = utils.TrajMiniBatch(Times, X, batch_size=batch_size, ds_mult=ds_mult,
            shuffle=True, x_transform=x_transform)
    lstm_mb = traj_lstm.BatchLSTM(my_mb, length, deltaT)
    my_mb_val = utils.TrajMiniBatch(Time_v, Xval, batch_size=batch_size, ds_mult=1,
            shuffle=True, x_transform=x_transform)
    lstm_mb_val = traj_lstm.BatchLSTM(my_mb_val, length, deltaT)

    hidden_size = 128
    #rnn = keras.models.Sequential()
    #rnn.add( keras.layers.LSTM(hidden_size, input_shape=(length,D+1), return_sequences=True) )
    #rnn.add( keras.layers.TimeDistributed(keras.layers.Dense(D)) )
    rnn = create_lstm(length, D, hidden_size)
    rnn.compile(loss='mse', optimizer='adam')

    model = traj_lstm.TrajectoryLSTM(model=rnn, normalizer=x_scaler, samples=samples, deltaT=deltaT, length=length)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, verbose=1, min_lr=1e-7)
    early_st = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, mode='auto')
    #model_cp = keras.callbacks.ModelCheckpoint("/tmp/weights-{epoch:02d}.h5", monitor='val_loss', save_best_only=True, save_weights_only=True, mode="auto")
    callback_list = [reduce_lr, early_st]
    model.fit_generator(generator=lstm_mb, validation_data=lstm_mb_val, epochs=epochs, use_multiprocessing=True, workers=8,
            callbacks=callback_list)
    print("Finishing training")

    return model


def main(args):
    with open(args.training_data,'rb') as f:
        data = np.load(f, encoding='latin1')
        Times = data['Times']
        X = data['X']
    N = len(Times)
    ntrain = int(round(N*args.p))
    nval = N-ntrain
    Xt = X[0:ntrain]
    Time_t = Times[0:ntrain]
    Xval = X[ntrain:]
    Time_val = Times[ntrain:]

    #model, t_pred, log_sig_y = ...
    model = train_ball_lstm(Xt, Time_t, Xval, Time_val, length=args.length, deltaT=1.0/180.0, batch_size=args.batch_size, epochs=args.epochs)
    print("Model trained")
    model.save(args.model)
    print("Model saved")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('training_data', help="File with the stored training trajectories")
    parser.add_argument('model', help="Path where the resulting model is saved")
    parser.add_argument('--p', type=float, default=0.1, help="Percentage of the trajectories used for validation")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Delta Time")
    parser.add_argument('--length', type=int, default=200, help="Length of the time series to model")
    parser.add_argument('--batch_size', type=int, default=128, help="Size of each minibatch")
    args = parser.parse_args()
    main(args)
