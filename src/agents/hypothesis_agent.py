from src.utils.logger import setup_logger
import openai

class HypothesisAgent:
    """
    HypothesisAgent for handling hypothesis-related tasks, such as generating,
    validating, and refining hypotheses for material science research.
    """

    def __init__(self, config):
        """
        Initialize the HypothesisAgent with configuration settings.

        Args:
            config (dict): Configuration parameters for the agent.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.model = config["agents"]["hypothesis"]["model_name"]
        self.client = openai.OpenAI(api_key=config['agents']['hypothesis']['api_key'],
                                    base_url=config['agents']['hypothesis']['base_url'])
        self.config = config

    def generate_hypothesis(self, research_goal, research_constr, literature_insights):
        """
        Generate a hypothesis based on the research goal and the literature review results.

        Args:
            goal (str): The research goal or problem statement.

        Returns:
            str: Generated hypothesis.
        """
        self.logger.info(f"Running hypothesis generation for goal: {research_goal}")

        prompt = [
            {"role": "system",
             "content": "You are an expert in materials science with a focus on helical structures and chiral properties. "
                        "Your task is to generate clear, specific, and testable hypotheses for nanohelices research. "
                        "Each hypothesis should be grounded in scientific principles of helix geometry, chirality, and "
                        "material behavior, and it must guide the design of experiments to evaluate these properties. "
                        "Incorporate insights from literature, ensure alignment with research goals and constraints, and "
                        "propose parameters within the defined space for virtual experiments."},
            {"role": "user", "content": """Here, I would like to generate a clear and testable hypothesis based on the provided research information. Please follow these REQUIREMENTS:
                1. Ensure the hypothesis aligns with the given research goal.
                2. Address all the specified research constraints.
                3. Incorporate insights or patterns identified in the provided literature review.
                4. Specifically consider the principles of helix geometry and chirality in the hypothesis.
                5. Focus on testing one parameter from the provided parameter space that is most relevant to the research goal.
                6. Include the parameter label and an initial value for the experiment, supported by literature or logical reasoning.
                7. Format the output as a single CONCISE hypothesis statement.
            """},
            {"role": "assistant",
             "content": "Understood! Could you please provide the research goal, constraints, literature review insights, and parameter space?"},
            {"role": "user", "content": f"""Research Goal: {research_goal}
                Constraints: {research_constr}
                Literature Review Summary: {literature_insights}
                Parameter Space: {self.config["parameter_space"]}"""}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=200,
            temperature=0,
            stream=False
        ).choices[0].message.content.strip()
        self.logger.info(f"Generated Hypothesis: {response}")
        return response

    def revise_hypothesis(self, research_report):
        """
        Revise the hypothesis based on the recent chatting history, including
        former hypotheses, experiment results, and research reports.

        Args:
            research_report (str): Research report from the previous iteration.

        Returns:
            str: Revised hypothesis.
        """
        self.logger.info("Revising hypothesis based on the research report of last iteration.")

        prompt = [
            {"role": "system",
             "content": (
                 f"""You are an expert in refining hypotheses for nanohelices research. Your primary task is to enhance hypotheses by incorporating insights from the research report of the previous iteration of experiments, and theoretical principles related to helix structure and chirality. The refined hypothesis must be precise, TESTABLE, and explicitly address the research objectives, constraints, and experimental outcomes. Pay special attention to the interplay between helix geometry (e.g., pitch, n_turns, helix_radius) and material properties, such as mechanical strength, optical activity, and chirality. Where applicable, use the Circular Dichroism (CD) spectrum as a guiding factor. Propose adjustments for future experiments to validate the hypothesis and explore hidden connections among parameters."""
             )},

            {"role": "user",
             "content": (
                 f"""Based on the following research report from the previous iteration of experiments, refine the hypothesis to better align with the research goal, constraints, and experimental outcomes. The hypothesis you revised MUST be CONCISE!
                 Research Report:
                {research_report}

                 The refined hypothesis must:
                 - Be CONCISE and focused on a specific parameter from the given parameter space:
                     angle: [0.123160654, 1.009814211]
                     curl: [0.628318531, 8.078381109]
                     fiber_radius: [20, 60]
                     height: [43.32551229, 954.9296586]
                     helix_radius: [20, 90]
                     n_turns: [3, 10] (integer values only)
                     pitch: [60, 200]
                     total_fiber_length: [303.7757835, 1127.781297]
                     total_length: [300, 650]

                 - Clearly articulate how the selected parameter influences material properties and contributes to achieving the research goal.
                 - You may suggest parameters within the defined space for virtual experiments.
                 - Apart from the experiment variables from the past iteration, you are encouraged to consider other parameters from the parameter space.
                 - Suggest specific values or adjustments for the parameter based on supporting evidence from experiments or literature.
                 - Explore potential hidden connections or interdependencies among parameters and propose hypotheses to investigate them.
                 - Format the output as a single CONCISE hypothesis statement."""
             )}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_tokens=200,
                temperature=0.5,
                stream=False
            ).choices[0].message.content.strip()

            self.logger.info(f"Revised Hypothesis: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error revising hypothesis: {e}")
            return None