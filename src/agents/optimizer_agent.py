import math, random
from src.utils.logger import setup_logger
from anytree import NodeMixin
import numpy as np

class CustomNode(NodeMixin):
    def __init__(self, name, values=None, parent=None, visits=0, reward=0.0):
        self.name = name
        self.values = values
        self.visits = visits
        self.reward = reward
        self.parent = parent

class OptimizerAgent:
    """
    OptimizationAgent for handling optimization-related tasks,
    such as optimizing material parameters to meet desired objectives.
    """

    def __init__(self, config, experiment_agent):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = config
        self.experiment_agent = experiment_agent
        self.tree_root = CustomNode("Root", values=None, visits=0, reward=0.0)

    def optimize_experiment(self, variables):
        """
        Suggest an optimized value for the given variable.

        Args:
            variables (list): List of variable names to optimize.

        Returns:
            dict: Suggested values for variables.
        """
        self.logger.info(f"Running optimization for parameter: {variables}")

        best_node = self.monte_carlo_tree_search(self.tree_root, variables, iterations=20)
        suggested_values = best_node.values if best_node.values is not None else self.experiment_agent.current_parameters

        self.logger.info(f"Suggested value for parameter {variables}: {suggested_values}")
        return suggested_values

    def monte_carlo_tree_search(self, node, variables, iterations=50):
        """
        Perform Monte Carlo Tree Search to find the optimal value for the variable.

        Args:
            node (Node): The root node to begin the search from.
            variables (list): List of variable names to optimize.
            iterations (int): Number of MCTS iterations to perform.

        Returns:
            Node: The node with the highest evaluation score.
        """
        best_node = None
        self.experiment_agent.experiment_results = []
        self.experiment_agent.varRecord = []
        for _ in range(iterations):
            leaf = self.expand(node, variables)
            reward = self.simulate(leaf, variables)
            self.backpropagate(leaf, reward)

            best_node = self.best_child(node, exploration_weight=1.414)
            if best_node is None or best_node.values is None:
                continue
            g_fact = self.evaluate_g_factor(best_node.values)
            self.experiment_agent.experiment_results.append(g_fact)
            self.experiment_agent.varRecord.append(best_node.values)
        return best_node

    def expand(self, node, variables):
        """Expand a leaf node by adding a new child."""
        try:
            new_values = {}
            for var in variables:
                min_val, max_val = self.experiment_agent.parameter_space[var]
                if var == "n_turns":
                    new_values["n_turns"] = random.randint(min_val, max_val)
                else: new_values[var] = random.uniform(min_val, max_val)
            new_child = CustomNode(name=new_values, values=new_values, parent=node, visits=0, reward=0.0)
            self.logger.info(f"Expanding node with new values: {new_values}")
            return new_child
        except Exception as e:
            self.logger.info(f"Failed to expand node: {str(e)}")
            return None

    def simulate(self, node, variables):
        """
        Simulate an experiment and return a reward based on parameter performance.

        Args:
            node (Node): The node representing the current parameter value.
            variables (list): List of variable names.

        Returns:
            float: Reward based on the simulated experiment outcome.
        """
        if node.values is None:
            return -float('inf')

        g_factor = self.evaluate_g_factor(node.values)
        self.logger.info(f"Simulating node with values {node.values} for {variables}, reward (g-factor): {g_factor}")
        return g_factor

    def evaluate_g_factor(self, parameter_values):
        """
        Evaluate the g-factor for a given parameter value.

        Args:
            parameter_values (dict): The value of the parameter being evaluated.

        Returns:
            float: The computed g-factor.
        """
        for var in parameter_values.keys() :
            self.experiment_agent.current_parameters[var] = parameter_values[var]
        return self.experiment_agent.run()

    def backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def best_child(self, node, exploration_weight=1.0):
        """Select the best child node based on the UCT value."""
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            if child.visits == 0:
                score = math.inf
            else:
                score = child.reward / child.visits + exploration_weight * np.sqrt(np.log(node.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
