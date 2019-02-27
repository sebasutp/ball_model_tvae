
import numpy as np

class BallInitState:
    """ Estimates the initial state of the ball trajectory
    """

    def __init__(self):
        pass

    def fit(self, time, obs):
        pass

    def sample(self, n):
        pass

class BallTrajectory:
    """ Ball trajectory modeling with differential equations

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __init__(self, air_drag, bounce_fac, max_bounces, 
            init_state_dist, deltaT,
            filter_window=50, n_samples=30):
        self.air_drag = air_drag
        self.bounce_fac = bounce_fac
        self.max_bounces = max_bounces
        self.filter_window = filter_window
        self.init_state_dist = init_state_dist
        self.deltaT = deltaT
        self.n_samples = 30

    def get_traj_sample(self, init_state, init_time, pred_times, deltaT=None):
        if deltaT is None: deltaT = self.deltaT
        pass

    def traj_dist(self, sess, prev_times, prev_obs, pred_times):
        """ Find the initial state distribution and sample from it
        """
        if len(prev_times) > self.filter_window:
            #N = len(prev_times)
            prev_times = prev_times[:self.filter_window]
            prev_obs = prev_obs[:self.filter_window]
        pt = prev_times - prev_times[0]
        init_state_dist = self.init_state_dist.fit(pt, prev_obs)
        states = init_state_dist.sample(self.n_samples)


