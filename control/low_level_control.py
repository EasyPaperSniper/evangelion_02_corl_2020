class low_level_controller:
    def __init__(self, parameter_list):
        pass

    def get_action(self, parameter_list):
        pass

    def _get_action(self, parameter_list):
        raise NotImplementedError

    def set_action_limits(self, parameter_list):
        raise NotImplementedError


class cpg_low_level_controller(low_level_controller):
    def __init__(self, parameter_list):
        pass

    def _get_action(self, parameter_list):
        pass


    