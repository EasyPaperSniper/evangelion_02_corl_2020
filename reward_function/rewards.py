import numpy as np


class velocity_tracking_reward():
    def __init__(self, target, reward_factor):
        self.total_reward = 0
        self.target = target
        self.reward_factor = reward_factor
    
    def set_target(self, target):
        self.target = target

    def calc_instant_reward(self, state, action):
        self.reward = -self.reward_factor[0]*np.linalg.norm(self.target[0:2] - state['base_velocity'][0:2]) - \
                         self.reward_factor[1]*np.linalg.norm(self.target[2]- state['base_pos'][2])
        self.total_reward += self.reward
        return self.reward

    def calc_additional_reward(self, **kargs):
        return 0