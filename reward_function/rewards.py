import numpy as np

MAX_JOINT_EFFORT = np.array([20., 55., 55.])
MAX_JOINT_VEL = np.array([52.4, 28.6, 28.6])

def normalize_joint_state(joint_state, norm_factor):
    state = np.reshape(joint_state, (4,3))
    state_norm = state / norm_factor
    print(state_norm)
    return np.reshape(state_norm, (12))

class expert_reward_optimize():
    def __init__(self, target, velocity_factor, height_factor, power_factor, effort_factor):
        self.total_reward = 0
        self.target = target
        self.velocity_factor = velocity_factor
        self.height_factor = height_factor
        self.power_factor = power_factor
        self.effort_factor = effort_factor
    
    def set_target(self, target):
        self.target = target

    def calc_instant_reward(self, state, action):
        norm_effort = normalize_joint_state(state['j_eff'], MAX_JOINT_EFFORT)
        norm_vel = normalize_joint_state(state['j_vel'], MAX_JOINT_VEL)
        self.reward = - (self.velocity_factor * np.linalg.norm(self.target[0:2] - state['base_velocity'][0:2]) + \
                        self.height_factor * np.linalg.norm(self.target[2]- state['base_pos'][2]) + \
                        self.power_factor * np.dot(norm_effort, norm_vel) + \
                        self.effort_factor *  np.linalg.norm(norm_effort) )
        self.total_reward += self.reward
        return self.reward

    def calc_additional_reward(self, **kargs):
        return 0