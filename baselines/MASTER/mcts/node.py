from anytree import NodeMixin

class MCTSNode(NodeMixin):
    def __init__(self, name, params=None, reasoning="", parent=None):
        super().__init__()
        self.name = name
        self.params = params or {}
        self.reasoning = reasoning  # Hypothesis from LLM
        self.parent = parent
        self.visits = 0
        self.total_reward = 0.0

    @property
    def average_reward(self):
        return 0.0 if self.visits == 0 else self.total_reward / self.visits
