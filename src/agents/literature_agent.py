import openai
from src.utils.logger import setup_logger
from src.utils.semantic_scholar_api import get_semantic_scholar_data

class LiteratureAgent:
    """LiteratureAgent for handling literature related tasks."""

    def __init__(self, config):
        """
        Initialize the LiteratureAgent with the necessary configuration and user proxy agent.

        Args:
            config (dict): Configuration settings for the agent.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.model = config["agents"]["literature"]["model_name"]
        self.client = openai.OpenAI(api_key=config['agents']['literature']['api_key'],
                                    base_url=config['agents']['literature']['base_url'])
        self.semantic_scholar_api = config["agents"]["literature"]["semantic_scholar_api"]

    def search_literature(self, research_goal, research_constr):
        """
        Perform a literature search using the Semantic Scholar API
        and summarize the reviewed articles using a language model.

        Returns:
            str: A summary of the literature review results.
        """
        query_prompt = [
            {"role": "system",
             "content": "Now you are an expert in generating search keywords for scientific database queries. Your task is to use refined research goals and research constraints to create precise and effective search queries for the Semantic Scholar API in the required format."},
            {"role": "user", "content": """Here, I need you to generate search queries for a literature review. Please follow these REQUIREMENTS:
                1. Use the provided clarified research goal and constraints to identify relevant search queries.
                2. Ensure the output is formatted as a single search string separated by commas, suitable for the Semantic Scholar API.
                3. Maintain brevity and precision, using domain-specific terms.
                4. Ensure the search terms cover both the research goal and constraints effectively.
                5. The query words you suppose should be AS LESS AS POSSIBLE, as Semantic Scholar may cannot find some enough literatures with too many constraints."""},
            {"role": "user", "content": f"""Clarified Research Goal: {research_goal}
                Clarified Research Constraints: {research_constr}"""}
        ]
        query = self.client.chat.completions.create(
            model=self.model,
            messages=query_prompt,
            max_tokens=250,
            temperature=0,
            stream=False
        ).choices[0].message.content.strip()

        self.logger.info(f"Generating literature review")
        self.logger.info(f"Searching queries: {query}")
        review_results = get_semantic_scholar_data(query, self.semantic_scholar_api)

        self.logger.info(f"Summarizing Literature Review Results")
        prompt = [
            {"role": "system",
             "content": "You are an expert in summarizing literature review results from scientific database searches. Your task is to process and summarize results retrieved from the Semantic Scholar API, focusing on the **mechanisms** by which various factors affect nanohelices materials."},
            {"role": "user",
             "content": """Here, I would like to summarize the search results from a literature review. The summaries should focus on the **mechanisms** and their **impact on nanohelices materials**. Please adhere to the following REQUIREMENTS:
                1. Include the article title, authors, and publication year.
                2. Provide a 1-2 sentence summary of the article's focus on **mechanisms**, specifically how different factors or processes affect **nanohelices materials**.
                3. Use precise scientific language to ensure clarity and relevance.
                4. Avoid including unrelated details; prioritize findings directly tied to the effects on nanohelices materials.
                5. Format the summaries for easy reference and further exploration."""},
            {"role": "user",
             "content": f"""Search Results Provided: {review_results}"""}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=1000,
            temperature=0
        ).choices[0].message.content.strip()
        self.logger.info(f"Literature review summary results: {response}")
        return response