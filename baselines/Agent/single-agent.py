import click
import numpy as np
import yaml
import json
import openai
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import setup_logger
from src.utils.exploration_rate import average_exploration_distance
from src.agents import (
    ExperimentAgent,
    OptimizerAgent
)

logger = setup_logger('CLI')

class Planner:
    def __init__(self, config, history_manager):
        self.history_manager = history_manager
        self.client = openai.OpenAI(api_key=config['planner']['api_key'],
                                    base_url=config['planner']['base_url'])
        self.model = config["planner"]["model_name"]

    def determine_next_agent(self):
        chat_history = self.history_manager.get_chat_history()
        if not chat_history:
            return {"next_agent": "experiment"}

        last_message = chat_history[-1]
        sender = last_message["sender"]
        if sender == "experiment":
            return {"next_agent": None}
        else:
            return {"next_agent": "experiment"}

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
    agents['experiment'] = ExperimentAgent(config)
    agents['optimization'] = OptimizerAgent(config, agents['experiment'])
    return agents

def run_autonomous_workflow(agents: Dict[str, Any], note_taker: NoteTaker, history_manager: ChattingHistoryManager):
    param_history = []
    while True:
        analysis = history_manager.analyze_history()
        next_agent_name = analysis.get("next_agent")
        if not next_agent_name:
            dist = average_exploration_distance(param_history)
            print(f"Exploration rate: {dist}")
            history_manager.record_message("System", "System", f"Avg exploring dist: {dist}")
            break

        agent = agents.get(next_agent_name)
        if not agent:
            logger.error(f"Agent {next_agent_name} not found.")
            break

        logger.info(f"Running {next_agent_name} agent...")
        if next_agent_name == "experiment":
            agent.initialize("Find the structural parameters corresponding to the strongest chirality (g-factor characteristics) in the nanohelix material system.")
            suggested_value = agents['optimization'].optimize_experiment(agent.getVariable())
            param_history.extend(agent.param_history)

            g_factors = agent.get_gfactors()
            best_g = max(g_factors) if g_factors else None
            print(best_g)
            history_manager.record_message("experiment", "System", "Experiment run completed")


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Run the multi-agent synthesis system with the specified config."""
    global logger
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(Path(config_path))

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