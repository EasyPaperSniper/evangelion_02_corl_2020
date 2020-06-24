import numpy as np

from low_level_control import cpg_low_level_controller
from unitree_toolkit.unitree_robot_env import unitree_robot_API

def main():
    env = unitree_robot_API(render=True, robot ='a1',control_mode='position')
    init_state = env.reset()
    EXP_LENGTH = 2000
    low_level_control = 'cpg'

    if low_level_control =='cpg':
        parameter_array = [1.,         1.,         0.5,        
                0.0, 
                0.0, 0.0, 0.0, 
                0.5, 0.36, 0.833, 
                0.0, 0.0,]

        low_level_control = cpg_low_level_controller(a_dim=12, 
                            num_legs=4, 
                            parameter_dim=12, 
                            max_frequency=2,
                            action_mid_point=[0.0, 1.5, -1.8] , 
                            action_scale=[1.6, 5.0, 1.8])

        low_level_control.update_policy_parmeters(parameter_array)


    for _ in range(50):
        action = init_state['j_pos']
        env.step(action)

    for _ in range(EXP_LENGTH):
        action = low_level_control.get_action()
        state = env.step(action)


if __name__ == '__main__':
    main()