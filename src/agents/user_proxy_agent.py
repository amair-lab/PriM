import openai
from src.utils.logger import setup_logger

class UserProxyAgent:
    """
    UserProxyAgent for handling interactions between the user and the system.
    It interprets user queries and provides meaningful responses.
    """

    def __init__(self, config):
        """
        Initialize the UserProxyAgent with configuration settings.

        Args:
            config (dict): Configuration parameters for the agent.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.model = config["agents"]["user_proxy"]["model_name"]
        self.client = openai.OpenAI(api_key=config['agents']['user_proxy']['api_key'],
                                    base_url=config['agents']['user_proxy']['base_url'])

    def get_research_goal(self):
        """
        Clarify the research goal using ChatGPT API
        :return: The clarified research goal by ChatGPT
        """
        query = self.getUserPrompt("Enter Research Goal: ")
        if query == "":
            return "Find the structural parameters corresponding to the strongest chirality (g-factor characteristics) in the nanoheliex material system."
        self.logger.info(f"Utilizing ChatGPT API to formalize research goal: {query}")
        prompt = [
            {"role": "system",
             "content": "Now you are an expert in research goal refinement for scientists. You should clarify and summarize research goals to make them precise and suitable for querying scientific databases like the Semantic Scholar API."},
            {"role": "user", "content": f"""Here, I would like to refine a given research goal for clarity and specificity. I need you to **generate a clarified research goal** by following these REQUIREMENTS:
                1. Maintain all critical scientific details and domain-specific terminology.
                2. Ensure the clarified goal is concise and uses keywords relevant to the research context.
                3. Remove extraneous or general descriptive phrases.
                4. Align the clarified goal with requirements for effective querying using scientific databases (e.g., Semantic Scholar).
                5. Format the clarified goal as a concise, keyword-focused statement."""},
            {"role": "assistant",
             "content": "Of course, I will help you refine the research goal following these requirements! Could you please provide the specific research goal?"},
            {"role": "user",
             "content": f"""Research Goal Provided: Investigate how ligand binding affects the growth morphology of gold nanorods under different pH conditions."""},
            {"role": "assistant",
             "content": "ligand binding, growth morphology, gold nanorods, pH-dependent synthesis, surface chemistry"},
            {"role": "user",
             "content": f"""Research Goal Provided: Explore the effect of different capping agents on the crystallinity of silver nanoparticles during seed-mediated synthesis."""},
            {"role": "assistant",
             "content": "capping agents, crystallinity, silver nanoparticles, seed-mediated synthesis, morphology control"},
            {"role": "user", "content": f"""Research Goal Provided: {query}"""}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=250,
            temperature=0,
            stream=False
        ).choices[0].message.content.strip()
        self.logger.info(f"Generated response: {response}")
        return response

    def get_research_constraints(self):
        """
        Clarify the research constraints using ChatGPT API
        :return: The clarified research constraints by ChatGPT
        """
        query = self.getUserPrompt("Enter Research Constraints:")
        if query == "":
            return "Explicitly show the underlying physicochemical principles regarding the structure and property relationships."
        self.logger.info(f"Utilizing ChatGPT API to formalize the constraints: {query}")
        prompt = [
            {"role": "system",
             "content": "Now you are an expert in research constraint refinement for scientists. You should clarify and summarize research constraints to make them precise and suitable for querying scientific databases like the Semantic Scholar API."},
            {"role": "user", "content": f"""Here, I would like to refine the constraints of a research project for clarity and specificity. I need you to **generate clarified research constraints** by following these REQUIREMENTS:
                1. Identify and emphasize the key limitations and boundaries of the research project.
                2. Ensure the clarified constraints are concise and use domain-specific terminology.
                3. Remove redundant or overly general phrases that do not contribute to a specific understanding of the constraints.
                4. Align the clarified constraints with requirements for effective querying using scientific databases (e.g., Semantic Scholar).
                5. Format the clarified constraints as a concise, keyword-focused statement."""},
            {"role": "assistant",
             "content": "Of course, I will help you refine the research constraints following these requirements! Could you please provide the specific constraints?"},
            {"role": "user",
             "content": f"""Research Constraints Provided: The synthesis process must occur at room temperature, and the reaction time must not exceed two hours. Additionally, the process requires the use of non-toxic and environmentally friendly reagents."""},
            {"role": "assistant",
             "content": "room temperature synthesis, reaction time < 2 hours, non-toxic reagents, eco-friendly processes"},
            {"role": "user",
             "content": f"""Research Constraints Provided: The material must exhibit high thermal conductivity and be compatible with existing silicon-based fabrication techniques. Additionally, the production cost must be below $50 per unit."""},
            {"role": "assistant",
             "content": "high thermal conductivity, silicon fabrication compatibility, production cost < $50/unit"},
            {"role": "user", "content": f"""Research Constraints Provided: {query}"""}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=250,
            temperature=0,
            stream=False
        ).choices[0].message.content.strip()
        self.logger.info(f"Generated response: {response}")
        return response

    def getUserPrompt(self, instructions):
        """
        Retrieve the prompt from the user to guide the whole workflow.

        Input:
            instructions: String, the instructions to guide the user to enter the prompt.

        Returns:
            Any: Prompt input by user.
        """
        return input(instructions)