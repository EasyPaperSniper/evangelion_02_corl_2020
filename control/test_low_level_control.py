import numpy as np

from low_level_control import cpg_low_level_controller
from unitree_toolkit.unitree_robot_env import unitree_robot_API

def main():
    env = unitree_robot_API(render=True, robot ='a1',control_mode='position')
    init_state = env.reset()
    EXP_LENGTH = 2000
    low_level_control = 'cpg'

    if low_level_control =='cpg':
        parameter_array = np.array([0.81052535, 0.77606666, 0.6355018 , 0.00095107, 0.33033718,
       0.00519624, 0.03436627, 0.92213039, 0.36761444, 0.72378643,
       0.39245134, 0.19365534])

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