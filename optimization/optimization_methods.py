import cma

def run_optimization(cfg, experiment_rollout, init_parameter=None):
    if not init_parameter:
        init_parameter = [0.5] * cfg.low_level_control.params.parameter_dim
    
    if  cfg.optimizer =='cma':
        return run_cma(cfg,  experiment_rollout, init_parameter)
        

def run_cma(cfg, experiment_rollout, init_parameter):
    opt_dim = cfg.low_level_control.params.parameter_dim
    result = cma.fmin(
        objective_function=experiment_rollout.evaluate_parameter_cost,
        x0=init_parameter,
        sigma0=0.25,
        options={'bounds': [[0] * opt_dim, [1] * opt_dim], 'maxfevals': cfg.num_trial},
    )
    best_solution = result[0]
    best_cost = result[1]
    return best_solution, best_cost