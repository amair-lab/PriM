import math
from .node import MCTSNode

class MCTSHypothesisSearch:
    def __init__(self, hypothesis_agent, experiment_agent, uct_c=1.414):
        self.hypothesis_agent = hypothesis_agent
        self.experiment_agent = experiment_agent
        self.uct_c = uct_c
        self.param_history = []
        self.g_history = []

    def run_search(self, root_node, iterations=100):
        best_node = root_node
        for _ in range(iterations):
            leaf = self.selection(root_node)
            if leaf:
                child = self.expansion(leaf)
                if child:
                    reward = self.simulation(child)
                    self.backpropagate(child, reward)
                    if child.average_reward > best_node.average_reward:
                        best_node = child
        return best_node

    def selection(self, node):
        current = node
        while current.children:
            best_score = -float('inf')
            best_child = None
            for c in current.children:
                score = self.uct_score(c, current)
                if score > best_score:
                    best_score = score
                    best_child = c
            current = best_child
        return current

    def uct_score(self, node, parent):
        if node.visits == 0:
            return float('inf')
        return node.average_reward + self.uct_c * math.sqrt(math.log(parent.visits) / node.visits)

    def expansion(self, node):
        new_params, reasoning = self.hypothesis_agent.suggest_params(node.params)
        child = MCTSNode(
            name=f"Child_{len(node.children)}",
            params=new_params,
            reasoning=reasoning,
            parent=node
        )
        return child

    def simulation(self, node):
        self.param_history.append(node.params)
        g = self.experiment_agent.run_with_params(node.params)
        self.g_history.append(g)
        return g

    def backpropagate(self, node, reward):
        current = node
        while current:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
