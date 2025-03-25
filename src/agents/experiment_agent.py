from copy import deepcopy
import openai, ast
from src.utils.logger import setup_logger
from virtual_lab.test_tool import do_experiment

class ExperimentAgent:
    """
    ExperimentAgent for handling experiment-related tasks, such as running
    simulations, exploring parameter spaces, and evaluating results.
    """

    def __init__(self, config):
        """
        Initialize the ExperimentAgent with configuration settings.

        Args:
            config (dict): Configuration parameters for the agent.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.model = config["agents"]["experiment"]["model_name"]
        self.client = openai.OpenAI(api_key=config['agents']['experiment']['api_key'],
                                    base_url=config['agents']['experiment']['base_url'])
        self.parameter_space = config.get("parameter_space")
        self.current_parameters = {}
        self.var = ""
        self.varRecord = []
        self.experiment_results = []
        self.param_history = []

    def initialize(self, hypothesis):
        """Initialize the experiment agent."""
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.experiment_results = []
        self.varRecord = []

        if not self.current_parameters:
            self.current_parameters = {
                param: (values[0] + values[1]) / 2
                for param, values in self.parameter_space.items()
            }
            self.current_parameters["n_turns"] = int(self.current_parameters["n_turns"])

        prompt = [
            {"role": "system",
             "content": "Now you are an expert in designing scientific experiments. Your task is to identify the experimental variables to be tested based on a given hypothesis. The output must include the parameter names and their proposed initial values from the hypothesis."},
            {"role": "user", "content": """Here, I need you to identify the experimental variables from a hypothesis. Please follow these REQUIREMENTS:
                1. Extract the specific parameters to be tested from the hypothesis, and the initial values of these parameters MUST be NUMERICAL values.
                2. ONLY output the parameter names and their initial values in the format: `{'variables': ['parameter_name1', 'parameter_name2'], 'values': [initial_value1, initial_value2]}`, do not include anything else.
                3. Ensure the output is concise and actionable for conducting experiments.
                4. The variable and corresponding parameter you are suggesting MUST lie in the pre-defined parameter space:\n
                parameter_space:
                  angle: [0.123160654, 1.009814211]
                  curl: [0.628318531, 8.078381109]
                  fiber_radius: [20, 60]
                  height: [43.32551229, 954.9296586]
                  helix_radius: [20, 90]
                  n_turns: [3, 10]
                  pitch: [60, 200]
                  total_fiber_length: [303.7757835, 1127.781297]
                  total_length: [300, 650].
                5. If there is no initial variables and values in the hypothesis, DECIDE BY YOURSELF and give the structured output!
            """},
            {"role": "assistant",
             "content": "Understood! Could you please provide the hypothesis for which I need to generate the experimental variables?"},
            {"role": "user",
             "content": """Hypothesis: At a curl value of 2.5 and a pitch value of 120, ligand binding at low pH enhances the elongation of gold nanorod tips during room temperature synthesis, maintaining stable growth within a reaction time of under 2 hours."""},
            {"role": "assistant", "content": """{'variables': ['curl', 'pitch'], 'values': [2.5, 120]}"""},
            {"role": "user",
             "content": """Hypothesis: At a fiber_radius of 40, increasing capping agent concentrations during seed-mediated synthesis at 25°C will enhance the crystallinity of silver nanoparticles, provided the reaction time remains under 3 hours."""},
            {"role": "assistant", "content": """{'variables': ['fiber_radius'], 'values': [40]}"""},
            {"role": "user",
             "content": """Hypothesis: At a pitch value of 120, the mechanical strength of helical nanofibers improves due to enhanced alignment and reduced defects within the given parameter constraints."""},
            {"role": "assistant", "content": """{'variables': ['pitch'], 'values': [120]}"""},
            {"role": "user", "content": f"""Hypothesis: {hypothesis}"""}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=250,
            temperature=0,
            stream=False
        ).choices[0].message.content.strip()

        try:
            variables = ast.literal_eval(response)['variables']
            init_values = ast.literal_eval(response)['values']
        except Exception as e:
            self.logger.error(f"Failed to parse response from openai: {response}")
            variables = input("Please enter the parameter names for experiment (comma-separated, NO SPACES):").split(",")
            init_values = input("Please enter the initial values for variables (comma-separated, NO SPACES):").split(",")

        self.logger.info(f"Setting the experiment variable as {variables} and its initial value as {init_values}")
        for var, val in zip(variables, init_values):
            if var == "n_turns":
                self.current_parameters[var] = int(val)
            else:
                self.current_parameters[var] = float(val)
        self.var = variables

        for param in self.parameter_space.keys():
            if param not in self.current_parameters:
                self.current_parameters[param] = (self.parameter_space[param][0] + self.parameter_space[param][1]) / 2
                self.logger.warning(
                    f"Parameter '{param}' was missing and set to default value: {self.current_parameters[param]}")

        self.param_history.append(deepcopy(list(self.current_parameters.values())))
        self.logger.info(f"Final initialized parameters: {self.current_parameters}")

    def run(self):
        """
        Run the main logic of the ExperimentAgent.

        Returns:
            dict: The results of the experiments.
        """
        self.logger.info(f"Running experiment for hypothesis:")

        try:
            results = do_experiment(
                angle=self.current_parameters['angle'],
                curl=self.current_parameters['curl'],
                fiber_radius=self.current_parameters['fiber_radius'],
                height=self.current_parameters['height'],
                helix_radius=self.current_parameters['helix_radius'],
                n_turns=self.current_parameters['n_turns'],
                pitch=self.current_parameters['pitch'],
                total_fiber_length=self.current_parameters['total_fiber_length'],
                total_length=self.current_parameters['total_length']
            )
            if results['status'] == 'success':
                self.param_history.append(deepcopy(list(self.current_parameters.values())))
                return results['predicted_g_factor']
            else: return None
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources for experiment agent."""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        self.varRecord = []
        self.experiment_results = []

    def getVariable(self):
        """Return the Variable in the experiment."""
        return self.var

    def getVarRecord(self):
        """Return the record of variable during the experiment"""
        return self.varRecord

    def get_gfactors(self):
        """Return the record of g-factor values during the experiment"""
        return self.experiment_results