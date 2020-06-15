import os

import numpy as np
import torch
import hydra

from unitree_toolkit.unitree_robot_env import unitree_robot_API

from test_sample_gaits import experiment_rollout_simple
from optimization.optimization_methods import run_optimization
from reward_function.rewards import velocity_tracking_reward



def init_env(cfg):
    env = unitree_robot_API(render=False, robot =cfg.robot_type, control_mode=cfg.control_mode)
    return env

def init_low_level_controller(cfg):
    low_level_control = hydra.utils.instantiate(cfg.low_level_control)
    return low_level_control

def init_reward(cfg):
    reward = hydra.utils.instantiate(cfg.reward)
    return reward


@hydra.main(config_path='../config/cma_cpg_simple.yaml', strict=False)
def main(cfg):
    # define env
    env = init_env(cfg)

    # define low level controller
    low_level_control = init_low_level_controller(cfg)

    # define reward
    reward = init_reward(cfg)

    # define experiment
    exp_rollout = experiment_rollout_simple(env, low_level_control, reward, cfg.episode_length)

    # optimization 
    best_solution, best_cost = run_optimization(cfg, exp_rollout, cfg.init_parameter)
    print(best_solution, best_cost)

    

if __name__ == '__main__':
    main()