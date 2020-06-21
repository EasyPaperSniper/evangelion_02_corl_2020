import numpy as np

from low_level_control import cpg_low_level_controller
from unitree_toolkit.unitree_robot_env import unitree_robot_API

def main():
    env = unitree_robot_API(render=True, robot ='a1',control_mode='position')
    init_state = env.reset()
    EXP_LENGTH = 2000
    low_level_control = 'cpg'

    if low_level_control =='cpg':
        parameter_array = [9.25029394e-03, 5.50869296e-01, 9.95786376e-01, 2.46700981e-01,
                            4.29118643e-05, 1.42139178e-02, 1.97832198e-01, 5.01789469e-01,
                            4.32138066e-01, 9.98640527e-01, 4.79130381e-05, 1.51229524e-02,]

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
        # print(action)
        state = env.step(action)


if __name__ == '__main__':
    main()