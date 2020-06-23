import numpy as np

import controllers.cpg_controller.gaits as cpg_robot # from daisy-simulation
import controllers.cpg_controller.cpg_oscillator as cpg

class low_level_controller:
    def __init__(self, a_dim , num_legs, action_mid_point=None, action_scale=None):
        self.a_dim = a_dim
        self.num_legs = num_legs
        self.num_joints_per_leg = int(a_dim/num_legs)

        if action_mid_point is None:
            self.action_mid_point = np.zeros(self.a_dim)
        else:
            self.action_mid_point = np.array([action_mid_point]*self.num_legs).reshape(self.a_dim)
            
        if action_scale is None:
            self.action_scale = np.ones(self.a_dim)
        else:
            self.action_scale = np.array([action_scale]*self.num_legs).reshape(self.a_dim)

    def get_action(self, ):
        action = self._get_action()
        action = np.multiply((action - 0.5) , self.action_scale)
        action = action + self.action_mid_point
        return action


    def _get_action(self,):
        raise NotImplementedError("Subclass must implement")

        

class cpg_low_level_controller(low_level_controller):
    def __init__(self, a_dim, num_legs, parameter_dim=48, max_frequency = 5, action_mid_point=None, action_scale=None):
        super().__init__(a_dim=a_dim, num_legs=num_legs, action_mid_point=action_mid_point, action_scale=action_scale)

        self.parameter_dim = parameter_dim
        self.max_frequency =  max_frequency
        self.parameter_array = np.clip(0.5*np.random.randn(parameter_dim),0, 1)        
        self.robot = cpg_robot.Robot(n_legs=num_legs, n_joint_per_leg=self.num_joints_per_leg)
        self.update_policy_parmeters(self.parameter_array)

    def gen_cpg_policy(self, ):
        gait = cpg_robot.Gait(robot=self.robot, 
                        relative_phase=self.relative_phases,
                        v=self.frequency,
                        R=self.amplitude,
                        amp_offset=self.amp_offset,
                        phase_offset=self.phase_offset,
                        a=self.constant, # don't know what is this parameter
                        )
        self.policy = cpg.CpgController(gait)
        self.action_step = 0

    def _get_action(self, ):
        action = np.zeros(self.a_dim)
        self.policy.update()
        motor_idx = 0
        for j in range(self.num_legs):
            for k in range(self.num_joints_per_leg):
                action[motor_idx+k]= -self.policy.y_data[k*self.policy.n_legs+j][self.action_step]
            motor_idx += 3

        self.action_step += 1
        return action
         

    def gen_parameter_matrix_quadruped(self):
        '''
        self.parameter_array:
        [0:3]: phase difference between every base/hip motor, the first leg is always 0, thus 4-1=3 dim
        [3]: frequency of legs, set the same for every leg
        [4:16]: amplitute for each motor
        [16:28]: amp_offset for each motor
        [28:36]: phase_offset for each leg's motor relative to its base/hip motor (3-1)*4=8 dim
        [36:48]:  unknown parameter
        '''
        phases = np.append([0.5], self.parameter_array[0:3])
        phases = phases * 2 * np.pi - np.pi  # Between -pi to pi
        self.relative_phases = np.zeros((self.num_legs, self.num_legs))
        for i in range(self.num_legs):
            for j in range(self.num_legs):
                self.relative_phases[i][j] = phases[j] - phases[i]

        self.frequency = self.max_frequency*self.parameter_array[3] * np.ones((self.num_joints_per_leg, self.num_legs))
        self.amplitude = np.reshape(self.parameter_array[4:16],(self.num_joints_per_leg, self.num_legs))
        self.amp_offset = np.reshape(self.parameter_array[16:28], (self.num_joints_per_leg, self.num_legs))
        self.phase_offset = np.reshape(self.parameter_array[28:36], (self.num_joints_per_leg-1, self.num_legs))
        self.phase_offset = np.vstack(np.zeros(self.num_legs), self.phase_offset[0], self.phase_offset[1])
        self.constant = np.reshape(self.parameter_array[36:48], (self.num_joints_per_leg, self.num_legs))

    def gen_parameter_matrix_quadruped_simplified(self):
        '''
        self.parameter_array:
        [0:3]: phase difference between every base/hip motor, the first leg is always 0, thus 4-1=3 dim
        [3]: frequency of legs, set the same for every leg
        [4:7]: amplitute for each motor
        [7:10]: amp_offset for each motor
        [10:12]: phase_offset for each leg's motor relative to its base/hip motor (3-1)*4=8 dim
        '''
        phases = np.append([0.5], self.parameter_array[0:3])
        phases = phases * 2 * np.pi - np.pi  # Between -pi to pi
        self.relative_phases = np.zeros((self.num_legs, self.num_legs))
        for i in range(self.num_legs):
            for j in range(self.num_legs):
                self.relative_phases[i][j] = phases[j] - phases[i]

        self.frequency = self.max_frequency*self.parameter_array[3] * np.ones((self.num_joints_per_leg, self.num_legs))
        self.amplitude = np.tile(np.reshape(self.parameter_array[4:7],(self.num_joints_per_leg, 1)), self.num_legs)
        self.amp_offset = np.tile(np.reshape(self.parameter_array[7:10],(self.num_joints_per_leg, 1)), self.num_legs)
        self.phase_offset = np.append([0], self.parameter_array[10:12])
        self.phase_offset =  np.tile(np.reshape(self.phase_offset,(self.num_joints_per_leg, 1)), self.num_legs)
        self.constant = 0.5* np.ones((self.num_joints_per_leg, self.num_legs))


    def update_policy_parmeters(self, parameter_array):
        self.parameter_array = parameter_array
        if self.num_legs==4:
            if self.parameter_dim == 48:
                self.gen_parameter_matrix_quadruped()
            else:
                self.gen_parameter_matrix_quadruped_simplified()
        self.gen_cpg_policy()



    