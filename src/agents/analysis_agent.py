import openai
from src.utils.logger import setup_logger
from src.utils.data_analysis_tools import data_analysis_tool
import json, ast


class AnalysisAgent:
    """
    AnalysisAgent for handling the analysis workflow:
    - Data distribution analysis using scikit-learn.
    - Trend analysis with polynomial fitting.
    - Critical value identification.
    - Generating workflow reports using OpenAI API.
    """

    def __init__(self, config):
        self.logger = setup_logger(self.__class__.__name__)
        self.model = config["agents"]["analysis"]["model_name"]
        self.client = openai.OpenAI(api_key=config['agents']['analysis']['api_key'],
                                    base_url=config['agents']['analysis']['base_url'])

    def run(self, var_records, g_factor_results,
            research_goal="", research_constr="", literature_insights="", hypothesis=""):
        """
        Perform the complete analysis workflow for multi-variable optimization.

        Args:
            var_records (list of dict): List of variable value combinations tested during experiments.
            g_factor_results (list): Corresponding g-factor results.
            research_goal (str): Research goal.
            research_constr (str): Research constraints.
            literature_insights (str): Literature insights.
            hypothesis (str): Hypothesis.

        Returns:
            dict: Summary report of analysis.
        """
        self.logger.info("Running AnalysisAgent...")

        try:
            flat_var_records = {var: [record[var] for record in var_records] for var in var_records[0].keys()}
        except Exception as e:
            last_record_key = set(var_records[-1].keys())
            filtered_var, filtered_g = [], []
            for record, g_factor in zip(var_records, g_factor_results):
                if set(record.keys()) == last_record_key:
                    filtered_var.append(record)
                    filtered_g.append(g_factor)
            flat_var_records = {var: [record[var] for record in filtered_var] for var in filtered_var[0].keys()}
            g_factor_results = filtered_g

        self.data_analysis_results = self.data_analysis(flat_var_records, g_factor_results)
        self.logger.info(f"Analysis results from toolbox: {self.data_analysis_results}.")

        report = self.generate_report(self.data_analysis_results, research_goal, research_constr, literature_insights,
                                      hypothesis)
        self.logger.info(f"Experiment report:\n {report}")
        return report

    def data_analysis(self, var_records, g_factor_results):
        """
        Perform data analysis for multi-variable optimization.

        Args:
            var_records (dict): Dictionary of variable values for each variable.
            g_factor_results (list): Corresponding g-factor results.

        Returns:
            dict: Summary report of analysis.
        """
        self.logger.info("Running analysis on experiment data...")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "data_analysis_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "var_records": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            },
                            "g_factor_results": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "degree": {"type": "integer"}
                        },
                        "required": ["var_record", "g_factor_results", "degree"]
                    },
                },
            },
        ]

        prompt = [{"role": "user", "content": "Analyze the experiment data:\n"
                                              f"The variable changes record are {var_records}\n"
                                              f"The g-factor experiment results are {g_factor_results}\n\n"

                                              "You may suggest parameter `degree` (degree of polynomial fitting) based on the experiment data provided."}]

        try:
            completion = self.client.chat.completions.create(model=self.model, messages=prompt, tools=tools,
                                                             max_tokens=3000, stream=False)

            call = completion.choices[0].message.tool_calls[0]
            self.logger.info(f"Selected tool calls: {call}")

            arguments = call.function.arguments
            self.logger.info(f"Invoking Data Analysis Tool with arguments: {arguments}")

            params = ast.literal_eval(arguments)
            results = data_analysis_tool(var_records=var_records, g_factor_results=g_factor_results, degree=params["degree"])
            return results
        except Exception as e:
            self.logger.error(f"Failed to complete analysis: {e}")
            return {"error": "Analysis failed."}

    def generate_report(self, data_analysis_results, research_goal="", research_constr="", literature_insights="",
                        hypothesis=""):
        """
        Generate a structured workflow report using OpenAI API.

        Args:
            data_analysis_results (dict): Data analysis results using the tools from the toolbox.
            research_goal (str): Research goal.
            research_constr (str): Research constraints.
            literature_insights (str): Literature insights.
            hypothesis (str): Hypothesis.

        Returns:
            str: Generated report.
        """
        prompt = f"""
        You are a research report writer specializing in materials science experiments. You are now requested to compile a comprehensive research report based on our research settings, experiment results, and analysis.

        **Research Context**:
        - **Research Goal**: {research_goal}
        - **Constraints**: {research_constr}
        - **Literature Review Summary**: {literature_insights}
        - **Hypothesis**: {hypothesis}

        **Data Analysis**:
        The analysis results obtained from the data analysis tools are attached below: {data_analysis_results}.

        **Requirements for the Report**:
        1. Provide a **concise summary** of the experimental results.
        2. Highlight important **insights** from the data and analysis.
        3. Include **tables** summarizing experimental setups, key parameters, and results.
        4. Suggest **next steps** for the research based on the current findings.

        The report should be saved as a structured markdown file. AND THE REPORT MUST BE CONCISE!

        Make sure the report is well-structured, easy to read, and conveys the necessary details for further analysis and replication.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a research report writer specializing in materials science."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0,
                stream=False
            ).choices[0].message.content.strip()
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate report using OpenAI: {e}")
            return "Report generation failed."

    def getAnalysisResults(self):
        return self.data_analysis_results