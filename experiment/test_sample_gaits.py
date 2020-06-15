
class experiment_rollout_simple():
    def __init__(self, env, low_level_control, reward,  episode_length=500):
        self.env = env
        self.low_level_control = low_level_control
        self.reward = reward
        self.episode_length = episode_length

    def test_sample_gait(self):
        self.reward.total_reward = 0
        for _ in range(self.episode_length):
            action = self.low_level_control.get_action()
            state = self.env.step(action)
            self.reward.calc_instant_reward(state, action)
        return self.reward.total_reward

    def evaluate_parameter_reward(self, parameter):
        self.low_level_control.update_policy_parmeters(parameter)
        total_reward = self.test_sample_gait()
        return total_reward

    def evaluate_parameter_cost(self, parameter):
        return - self.evaluate_parameter_reward(parameter)
