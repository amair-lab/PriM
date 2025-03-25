import click, yaml, json
from openai import OpenAI
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.logger import setup_logger
from src.utils.g_v_ite import plot_g_factor_results
from src.utils.exploration_rate import average_exploration_distance
from src.agents import (
    UserProxyAgent,
    LiteratureAgent,
    HypothesisAgent,
    ExperimentAgent,
    AnalysisAgent,
    OptimizerAgent
)

logger = setup_logger('CLI')

class Planner:
    def __init__(self, config, history_manager):
        self.history_manager = history_manager
        self.client = OpenAI(api_key=config['planner']['api_key'],
                                    base_url=config['planner']['base_url'])
        self.model = config["planner"]["model_name"]

    def query_llm(self):
        chat_history = self.history_manager.get_chat_history()
        if len(chat_history) > 10:
            chat_history = chat_history[-9:]

        prompt = (
            f"""You are an expert in scientific research workflows for multi-agent systems discovering structure-attribute relationships in nanohelices materials.

            Your task is to decide the next agent based on the recent chatting history and agent abilities. Here are the agent roles:
            - 'user_proxy': Defines the research goals and constraints.
            - 'literature': Extracts and summarizes relevant information from scientific literature, even if there is no relevant literature, you may still guide to the 'hypothesis' agent.
            - 'hypothesis': Formulates hypotheses based on research goal+constraint and literature insights or research report from the last iteration.
            - 'experiment': Conducts experiments based on the hypothesis on the current iteration.
            - 'analysis': After conducting the experiments, analyzes experimental data and determines whether the hypothesis is valid.

            ONLY RESPOND WITH THE NAME OF THE NEXT AGENT: 'literature', 'hypothesis', 'experiment', 'analysis', or 'None'.
            f"Chat History:\n{chat_history}"""
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an intelligent planning assistant under the settings of materials science research."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def determine_next_agent_llm(self) -> Dict[str, Any]:
        chat_history = self.history_manager.get_chat_history()
        if not chat_history:
            return {"next_agent": "user_proxy"}

        last_message = chat_history[-1]
        sender = last_message['sender']
        if sender == 'analysis':
            feedback = ExpControl()
            if feedback == "CONTINUE":
                return {"next_agent": "hypothesis"}
            else:
                return {"next_agent": None}

        try:
            next_agent = self.query_llm()
            if next_agent not in ["user_proxy", "literature", "hypothesis", "experiment", "analysis", "optimizer",
                                  "None"]:
                logger.error(f"Invalid agent suggested by LLM: {next_agent}")
                return {"next_agent": None}
            return {"next_agent": next_agent if next_agent != "None" else None}
        except Exception as e:
            logger.error(f"Error in LLM-based planning: {str(e)}", exc_info=True)
            return {"next_agent": None}

    def determine_next_agent(self):
        chat_history = self.history_manager.get_chat_history()
        if not chat_history:
            return {"next_agent": "user_proxy"}

        last_message = chat_history[-1]
        logger.info(f"Planner analyzing history: last_message={last_message}")

        sender = last_message["sender"]
        feedback = human_feedback(sender)
        if not feedback:
            return {"next_agent": sender}

        if sender == "user_proxy":
            return {"next_agent": "literature"}
        elif sender == "literature":
            return {"next_agent": "hypothesis"}
        elif sender == "hypothesis":
            return {"next_agent": "experiment"}
        elif sender == "experiment":
            return {"next_agent": "analysis"}
        elif sender == "optimizer":
            return {"next_agent": "analysis"}
        else:
            feedback = ExpControl()
            if feedback == "CONTINUE":
                return {"next_agent": "hypothesis"}
            else:
                return {"next_agent": None}


class ChattingHistoryManager:
    def __init__(self, config, mode):
        self.chat_history = {"chat_history": []}
        self.save_path = Path(config['file_paths']['chat_history'])
        self.planner = Planner(config, self)
        self.mode = mode

    def record_message(self, sender: str, receiver: str, message: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "message": message
        }
        logger.info(f"Recording message: {entry}")
        self.chat_history["chat_history"].append(entry)
        self.save_to_file()

    def get_chat_history(self) -> Dict[str, Any]:
        return self.chat_history["chat_history"]

    def get_recent_chat_history(self) -> Dict[str, Any]:
        try:
            recent_chat_history = self.chat_history["chat_history"][-5:]
        except Exception as e:
            recent_chat_history = self.chat_history["chat_history"]
        return recent_chat_history

    def analyze_history(self) -> Dict[str, Any]:
        if self.mode.lower() == "llm" or self.mode == "":
            return self.planner.determine_next_agent_llm()
        else:
            return self.planner.determine_next_agent()

    def save_to_file(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.chat_history, f, indent=4)


def human_feedback(AgentName):
    '''
    :param AgentName: The name of current agent
    :return: (bool) TRUE if agent response is accepted, FALSE otherwise
    '''
    if AgentName == 'analysis':
        fdb = input("Do you wish to continue with an another experiment? \n Respond Y or [N]:")
        if fdb.lower() not in ["y", "n", ""]:
            fdb = input("Invalid Input! \n Do you wish to continue with an another experiment? \n Respond [Y] or N:")
        if fdb.lower() in ["n", ""]:
            return "STOP"
        else:
            return "CONTINUE"

    feedback = input(
        f"Please enter your feedback of the output of {AgentName}. \n You can decide whether to accept (Y) the output or not. \n Respond [Y] or N:")
    if feedback.lower() not in ["y", "n", ""]:
        feedback = input(
            f"Invalid input!\n Please enter your feedback of the output of {AgentName}. \n Respond Y or N:")
    if feedback.lower() == "y" or feedback == "":
        return True
    else:
        return False


def ExpControl():
    fdb = input("Do you wish to continue with an another experiment? \n Respond Y or [N]:")
    if fdb.lower() not in ["y", "n", ""]:
        fdb = input("Invalid Input! \n Do you wish to continue with an another experiment? \n Respond [Y] or N:")
    if fdb.lower() in ["n", ""]:
        return "STOP"
    else:
        return "CONTINUE"


class NoteTaker:
    def __init__(self, config):
        self.notes = {}
        self.save_path = Path(config['file_paths']['notes'])

    def record(self, key: str, value: Any):
        self.notes[key] = value
        self.save_to_file()

    def get_notes(self):
        return self.notes

    def save_to_file(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.notes, f, indent=4)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_agents(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all enabled agents with required arguments."""
    agents = {}
    agents['user_proxy'] = UserProxyAgent(config)
    agents['literature'] = LiteratureAgent(config)
    agents['hypothesis'] = HypothesisAgent(config)
    agents['experiment'] = ExperimentAgent(config)
    agents['analysis'] = AnalysisAgent(config)
    agents['optimizer'] = OptimizerAgent(config, agents['experiment'])

    return agents


def run_autonomous_workflow(agents: Dict[str, Any], note_taker: NoteTaker, history_manager: ChattingHistoryManager):
    g_factors = []
    param_space = {}
    param_history = []
    while True:
        analysis = history_manager.analyze_history()
        next_agent_name = analysis.get("next_agent")
        if not next_agent_name:
            if g_factors:
                plot_g_factor_results(g_factors, "data/fig.png")
            if param_history:
                print(f"Exploration rate: {average_exploration_distance(param_history)}")
            print(f"g-factor results over iterations: {g_factors}")
            print(f"Optimal parameter space discovered: {param_space}")
            logger.info("Workflow complete.")
            break

        agent = agents.get(next_agent_name)
        if not agent:
            logger.error(f"Agent {next_agent_name} not found.")
            break

        logger.info(f"Running {next_agent_name} agent...")
        if next_agent_name == "user_proxy":
            research_goal = agent.get_research_goal()
            research_constraints = agent.get_research_constraints()
            history_manager.record_message(next_agent_name, "System",
                                           f"Goal: {research_goal}, Constraints: {research_constraints}")
            note_taker.record("research_goal", research_goal)
            note_taker.record("research_constraints", research_constraints)
        elif next_agent_name == "literature":
            literature_insights = agent.search_literature(note_taker.get_notes().get("research_goal"),
                                                          note_taker.get_notes().get("research_constraints"))
            history_manager.record_message(next_agent_name, "System", f"Insights: {literature_insights}")
            note_taker.record("literature_insights", literature_insights)
        elif next_agent_name == "hypothesis":
            if not g_factors:
                hypothesis = agent.generate_hypothesis(note_taker.get_notes().get("research_goal"),
                                                       note_taker.get_notes().get("research_constraints"),
                                                       note_taker.get_notes().get("literature_insights"))
            else:
                hypothesis = agent.revise_hypothesis(note_taker.get_notes().get("Research Experiment Report"))
            history_manager.record_message(next_agent_name, "System", f"Generated hypothesis: {hypothesis}")
            note_taker.record("hypothesis", hypothesis)
        elif next_agent_name == "experiment":
            agent.initialize(note_taker.get_notes().get("hypothesis"))
            suggested_value = agents['optimizer'].optimize_experiment(agent.getVariable())
            param_history += agent.param_history
            param_space = agent.current_parameters
            history_manager.record_message("experiment", "System",
                                           f"Variable change record in experiment: {agent.getVarRecord()}")
            history_manager.record_message("experiment", "System",
                                           f"g-factor change record in experiment: {agent.get_gfactors()}")
            history_manager.record_message("optimizer", "ExperimentAgent", f"Suggested value: {suggested_value}")
            history_manager.record_message("experiment", "System", "Experiment run completed")
            note_taker.record("Variable Change Record", agent.getVarRecord())
            note_taker.record("g-factor Change Record", agent.get_gfactors())
            note_taker.record("Experiment Variable", agent.getVariable())
            note_taker.record("Suggested Value", suggested_value)
        elif next_agent_name == "analysis":
            report = agent.run(
                note_taker.get_notes().get("Variable Change Record"),
                note_taker.get_notes().get("g-factor Change Record"),
                note_taker.get_notes().get("research_goal"),
                note_taker.get_notes().get("research_constraints"),
                note_taker.get_notes().get("literature_insights"),
                note_taker.get_notes().get("hypothesis")
            )
            g_factors.append(note_taker.get_notes().get("g-factor Change Record")[-1])
            note_taker.record("Analysis result", agent.getAnalysisResults())
            note_taker.record("Research Experiment Report", report)
            history_manager.record_message(next_agent_name, "System", "Analysis completed")
            history_manager.record_message(next_agent_name, "System", f"Research Experiment Report: {report}")


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Run the multi-agent synthesis system with the specified config."""
    global logger
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(Path(config_path))

    # Setup logging based on config
    if 'logging' in config:
        logger.setLevel(config['logging'].get('level', 'INFO'))
        if 'log_file' in config['logging']:
            logger = setup_logger(
                'CLI',
                Path(config['logging']['log_file']),
                level=config['logging'].get('level', 'INFO')
            )

    mode = input(
        """Enter <LLM> for LLM-based planning or <otherwise> for pre-defined planning logic with user feedback:""")
    logger.info("Initializing ChattingHistoryManager...")
    history_manager = ChattingHistoryManager(config, mode)

    logger.info("Initializing NoteTaker...")
    note_taker = NoteTaker(config)

    logger.info("Initializing agents...")
    agents = initialize_agents(config)

    try:
        logger.info("Executing workflow...")
        run_autonomous_workflow(agents, note_taker, history_manager)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

    logger.info("Multi-agent system execution completed")

if __name__ == '__main__':
    main()