from virtual_lab.test_tool import do_experiment
class ExperimentAgent:
    def __init__(self, parameter_space, noise_level=0.05):
        self.parameter_space = parameter_space

    def run_with_params(self, params):
        results = do_experiment(**params)
        if results['status'] == 'success':
            return results['predicted_g_factor']
