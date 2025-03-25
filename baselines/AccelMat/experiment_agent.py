import ast
from virtual_lab.test_tool import do_experiment
from openai import OpenAI


class ExperimentAgent:
    """
    ExperimentAgent sets up experiments based on a hypothesis by extracting 
    experimental variables and running the experiment via a provided test tool.
    """

    def __init__(self):
        self.parameter_space = {
            "angle": [0.123160654, 1.009814211],
            "curl": [0.628318531, 8.078381109],
            "fiber_radius": [20, 60],
            "height": [43.32551229, 954.9296586],
            "helix_radius": [20, 90],
            "n_turns": [3, 10],
            "pitch": [60, 200],
            "total_fiber_length": [303.7757835, 1127.781297],
            "total_length": [300, 650]
        }
        self.client = OpenAI(
            base_url="https://35.aigcbest.top/v1",
            api_key="sk-AM41JeYzqBTluCfs8XpqoiTbrrmdA6QGMtypVm2rGwSBBmFj",
        )
        self.params = {}
        self.param_history = []
        self.var = []
        self.g_factor = []

    def initialize(self, hypothesis: dict) -> None:
        """
        Initialize experimental parameters using a given hypothesis.
        
        The hypothesis is passed to a GPT model to extract experimental variables
        and their initial numerical values. If the extraction fails, the user is prompted
        for manual input.
        """
        self.params = {
            param: (values[0] + values[1]) / 2
            for param, values in self.parameter_space.items()
        }
        self.params["n_turns"] = int(self.params["n_turns"])

        prompt = [
            {
                "role": "system",
                "content": (
                    "Now you are an expert in designing scientific experiments. Your task is to identify "
                    "the experimental variables to be tested based on a given hypothesis. The output must "
                    "include the parameter names and their proposed initial values from the hypothesis."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here, I need you to identify the experimental variables from a hypothesis. Please follow these "
                    "REQUIREMENTS:\n"
                    "1. Extract the specific parameters to be tested from the hypothesis, and the initial values of these "
                    "parameters MUST be NUMERICAL values.\n"
                    "2. ONLY output the parameter names and their initial values in the format: "
                    "`{'variables': ['parameter_name1', 'parameter_name2'], 'values': [initial_value1, initial_value2]}`. Do "
                    "not include anything else.\n"
                    "3. Ensure the output is concise and actionable for conducting experiments.\n"
                    "4. The variable and corresponding parameter you are suggesting MUST lie in the pre-defined parameter space:\n"
                    "   angle: [0.123160654, 1.009814211]\n"
                    "   curl: [0.628318531, 8.078381109]\n"
                    "   fiber_radius: [20, 60]\n"
                    "   height: [43.32551229, 954.9296586]\n"
                    "   helix_radius: [20, 90]\n"
                    "   n_turns: [3, 10]\n"
                    "   pitch: [60, 200]\n"
                    "   total_fiber_length: [303.7757835, 1127.781297]\n"
                    "   total_length: [300, 650]\n"
                    "5. If there is no initial variable and values in the hypothesis, DECIDE BY YOURSELF and give the structured output!"
                ),
            },
            {
                "role": "assistant",
                "content": "Understood! Could you please provide the hypothesis for which I need to generate the experimental variables?"
            },
            {
                "role": "user",
                "content": (
                    "Hypothesis: At a curl value of 2.5 and a pitch value of 120, ligand binding at low pH enhances the elongation "
                    "of gold nanorod tips during room temperature synthesis, maintaining stable growth within a reaction time "
                    "of under 2 hours."
                ),
            },
            {
                "role": "assistant",
                "content": "{'variables': ['curl', 'pitch'], 'values': [2.5, 120]}"
            },
            {
                "role": "user",
                "content": (
                    "Hypothesis: At a fiber_radius of 40, increasing capping agent concentrations during seed-mediated synthesis at "
                    "25°C will enhance the crystallinity of silver nanoparticles, provided the reaction time remains under 3 hours."
                ),
            },
            {
                "role": "assistant",
                "content": "{'variables': ['fiber_radius'], 'values': [40]}"
            },
            {
                "role": "user",
                "content": (
                    "Hypothesis: At a pitch value of 120, the mechanical strength of helical nanofibers improves due to enhanced "
                    "alignment and reduced defects within the given parameter constraints."
                ),
            },
            {
                "role": "assistant",
                "content": "{'variables': ['pitch'], 'values': [120]}"
            },
            {"role": "user", "content": f"Hypothesis: {hypothesis}"},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=250,
            temperature=0,
            stream=False,
        ).choices[0].message.content.strip()

        try:
            variables = ast.literal_eval(response)["variables"]
            init_values = ast.literal_eval(response)["values"]
        except Exception:
            variables = input(
                "Please enter the parameter names for experiment (comma-separated, NO SPACES):"
            ).split(",")
            init_values = input(
                "Please enter the initial values for variables (comma-separated, NO SPACES):"
            ).split(",")

        for var, val in zip(variables, init_values):
            if var == "n_turns":
                self.params[var] = int(val)
            else:
                self.params[var] = float(val)
        self.var = variables

    def run_experiment(self):
        """
        Run the experiment using the current parameters and record the parameters in the history.
        
        Returns:
            The predicted g-factor if the experiment is successful, otherwise None.
        """
        try:
            self.param_history.append(self.params.copy())
            results = do_experiment(**self.params)
            if results.get("status") == "success":
                self.g_factor.append(results.get("predicted_g_factor"))
                return results.get("predicted_g_factor")
            else:
                return None
        except Exception:
            return None