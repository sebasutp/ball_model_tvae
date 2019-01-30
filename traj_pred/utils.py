
import numpy as np

def encode_fixed_dt(Times, Xreal, length, deltaT=1.0/180.0):
    """ Encode time series as with fixed time and missing observations

    Assume a time series with an appoximate fixed deltaT is measured, 
    but we have observations with their time stamps and possibly some 
    observations are missing. This function returns an array of size
    (N, length, D), where N is the number of trajectories, length is
    an argument passed representing the length of each trajectory, and
    D is the dimensionality of each observation.
    """

    N = len(Times)
    assert(N>0 and len(Xreal) == N)
    D = len(Xreal[0][0])

    X = np.zeros((N,length,D))
    Xobs = np.zeros((N,length,1))
    for n in range(N):
        assert(len(Xreal[n]) == len(Times[n]))
        X[n,0] = Xreal[n][0]
        Xobs[n,0] = 1
        Tn = len(Xreal[n])
        t_ix = -1
        for i in range(1,Tn):
            assert(Times[n][i] > Times[n][i-1])
            t = Times[n][i] - Times[n][0]
            t_ix = int(round(t / deltaT))
            if t_ix >= length:
                break
            X[n,t_ix] = Xreal[n][i]
            Xobs[n,t_ix] = 1
    return X, Xobs

def cov_to_std(covs):
    """ Changes a tensor of covariance matrices to standard deviations

    Given an array X[...,d,d] representing covariance matrices, returns an array
    Y[...,d] representing the standard deviations (diagonal) corresponding to X.
    """
    var = np.diagonal(covs, axis1=-2,axis2=-1)
    std = np.sqrt(var)
    return std

def apply_scaler(f, tensor):
    """ Apply the scaler function to the tensor
    """
    shape = np.shape(tensor)
    tr = tensor.reshape(-1,shape[-1])
    st = f(tr)
    return st.reshape(shape)
