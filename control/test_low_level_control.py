import numpy as np

from low_level_control import cpg_low_level_controller
from unitree_toolkit.unitree_robot_env import unitree_robot_API

def main():
    env = unitree_robot_API(render=True, robot ='a1',control_mode='position')
    state = env.reset()
    EXP_LENGTH = 1000
    low_level_control = 'cpg'

    if low_level_control =='cpg':
        parameter_array = np.array([
            np.pi, np.pi, 0,         # phase difference between every base/hip motor
            1,
            0, 0.2, 0.3,            
            0, 0, 0,
            0, 0
        ])

        low_level_control = cpg_low_level_controller(a_dim=12, 
                            num_legs=4, 
                            parameter_dim=12, 
                            action_mid_point=[0.0, 1.5, -1.8] , 
                            action_scale=[0.8, 2.5, 0.8])

        low_level_control.update_policy_parmeters(parameter_array)

    for _ in range(EXP_LENGTH):
        action = low_level_control.get_action()
        state = env.step(action)


if __name__ == '__main__':
    main()