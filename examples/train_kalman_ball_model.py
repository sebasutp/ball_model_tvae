""" Train a ball model with a extended Kalman Filter
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import traj_pred.kalman as kalman
import json

class BallPhysicsA:

    def __init__(self, quadAirDrag, bounceVec, deltaT, bounceZ):
        self.deltaT = deltaT
        self.bounceZ = bounceZ
        self.quadAirDrag = quadAirDrag
        self.bounceVec = bounceVec
        self.gravity = 9.807
        self.num_bounces = 0
        #self.maxBounces = maxBounces

    def flyMat(self, state, params=None, **opt_par):
        params = params if params is not None else self.params
        deltaT = self.deltaT
        vx = state['prev_z'][1,3,5] 
        xdd = -self.quadAirDrag*np.linalg.norm(vx) - self.gravity
        Adim = []
        for d in xrange(3):
            Adim.append( np.array([[1, deltaT + xdd*deltaT**2],[0,xdd*deltaT]]) )
        Afly = scipy.linalg.block_diag(Adim[0],Adim[1],Adim[2])
        return Afly

    def bounceMat(self, state, params=None, **opt_par):
        deltaT = self.deltaT 
        bounceFac = self.bounceVec
        Abounce = np.array([[1,deltaT - 0.5*deltaT**2,0,0,0,0],[0,bounceFac[0],0,0,0,0], \
            [0,0,1,deltaT - 0.5*deltaT**2,0,0], [0,0,0,bounceFac[1],0,0], \
            [0,0,0,0,1,0], [0,0,0,0,0,-bounceFac[2]]])
        return Abounce

    def isBounce(self, state):
        if 'isBounce' in self.optional and state['n']<len(self.optional['isBounce']) and state['t']<len(self.optional['isBounce'][state['n']]):
            return self.optional['isBounce'][state['n']][state['t']]
        Afly = self.flyMat(state)
        Bfly = state['B'].flyMat(state)
        next_state = np.dot(Afly, state['prev_z']) + np.dot(Bfly, state['action'])
        next_z = next_state[4]
        if next_z <= self.bounceZ:
            next_x = next_state[0]
            next_y = next_state[2]
            return True #(self.num_bounces < self.maxBounces) 
        return False

    def mat(self, state, params=None, **opt_par):
        if self.isBounce(state):
            self.num_bounces += 1
            return self.bounceMat(state, params, **opt_par)
        return self.flyMat(state,params,**opt_par)

    def n_params(self):
        return len(self.__params)

    def shape(self):
        return 6,6

    def deriv(self, state, param_id, params=None, **opt_par):
        return None

class Ball_B_Mat:
    def __init__(self, deltaT, bounceZ, bounceFac):
        self.deltaT = deltaT
        self.gravety = -9.8
        self.bounceZ = bounceZ
        self.bounceFac = bounceFac
        self.params = 0 #don't optimize anything here for the moment

    def flyMat(self, state, params=None, **opt_par):
        deltaT = self.deltaT if not 'deltaT' in opt_par else opt_par['deltaT']
        Bfly = np.array([[0],[0],[0],[0],[0.5*deltaT**2*self.gravety],[deltaT*self.gravety]])
        return Bfly

    def bounceMat(self, state, params=None, **opt_par):
        bounceZ = self.bounceZ
        bfac = self.bounceFac #state['A'].params[5]
        Bbounce = np.array([[0],[0],[0],[0],[bounceZ*(1+bfac)],[0]])
        #Bbounce = np.array([[0],[0],[0],[0],[0],[0]])
        return Bbounce

    def mat(self, state, params=None, **opt_par):
        if (state['A'].isBounce(state)):
            return self.bounceMat(state, params, **opt_par)
        return self.flyMat(state, params, **opt_par)

    def n_params(self):
        return len(self.__params)

    def shape(self):
        return 6,1

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__params = params

    def deriv(self, state, param_id, params, **opt_par):
        raise NotImplementedError
