import math
import random
import copy
from typing import List, Dict

class Node:
    def __init__(self, state: Dict[str, float], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def uct(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)


def select(node: Node) -> Node:
    current = node
    while current.children:
        current = max(current.children, key=lambda child: child.uct())
    return current


def expand(node: Node, parameter_space: Dict[str, List[float]]) -> Node:
    new_state = copy.deepcopy(node.state)
    param = random.choice(list(parameter_space.keys()))
    low, high = parameter_space[param]
    perturbation = (high - low) * 0.1 * (random.random() - 0.5)
    new_value = new_state[param] + perturbation
    new_state[param] = min(max(new_value, low), high)
    child_node = Node(new_state, parent=node)
    node.children.append(child_node)
    return child_node


def simulate(agent, state: Dict[str, float]) -> float:
    agent.params = state
    reward = agent.run_experiment()
    if reward is None:
        reward = 0.0
    return reward


def backpropagate(node: Node, reward: float) -> None:
    current = node
    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent


def run_mcts(agent, iterations: int = 100) -> None:
    root = Node(copy.deepcopy(agent.params))
    for i in range(iterations):
        leaf = select(root)
        if leaf.visits > 0:
            child = expand(leaf, agent.parameter_space)
        else:
            child = leaf
        reward = simulate(agent, child.state)
        backpropagate(child, reward)
    if root.children:
        best = max(root.children, key=lambda child: child.visits)
        agent.params = best.state

def compute_average_exploration_distance(param_history: List[Dict[str, float]]) -> float:
    N = len(param_history)
    if N < 2:
        return 0.0

    total_distance = 0.0
    count = 0
    keys = list(param_history[0].keys())
    for i in range(N):
        for j in range(i + 1, N):
            diff_squared = 0.0
            for key in keys:
                diff = param_history[i][key] - param_history[j][key]
                diff_squared += diff * diff
            distance = math.sqrt(diff_squared)
            total_distance += distance
            count += 1

    average_distance = (total_distance * 2) / (N * (N - 1))
    return average_distance
