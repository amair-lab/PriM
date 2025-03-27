import sys, os, math, json, yaml
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcts.node import MCTSNode
from mcts.search import MCTSHypothesisSearch
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def average_exploration_distance(param_history):
    """
    Computes the average pairwise Euclidean distance between all distinct pairs
    of parameter vectors in param_history.

    param_history: List of dicts, where each dict represents a parameter vector.
    Returns: A float representing the average pairwise distance.
    """
    N = len(param_history)
    if N < 2:
        return 0.0

    total_dist = 0.0

    for x, y in combinations(param_history, 2):
        dist_sq = sum((x_val - y_val) ** 2 for x_val, y_val in zip(x.values(), y.values()))
        total_dist += math.sqrt(dist_sq)

    return (2 * total_dist) / (N * (N - 1))

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = load_config(config_path)
    research_goal = "Find the structural parameters corresponding to the strongest chirality (g-factor characteristics) in the nanohelix material system."
    research_constr = "Explicitly show the underlying physicochemical principles regarding the structure and property relationships."

    # Agents
    hypothesis_agent = HypothesisAgent(
        config['openai_api_key'],
        config['openai_base_url'],
        config['parameter_space'],
        research_goal,
        research_constr,
    )
    experiment_agent = ExperimentAgent(
        parameter_space=config['parameter_space'],
    )

    # Initialize MCTS
    init_params = {k: (v[0] + v[1]) / 2 for k, v in config['parameter_space'].items()}
    root = MCTSNode("Root", params=init_params)
    mcts = MCTSHypothesisSearch(
        hypothesis_agent=hypothesis_agent,
        experiment_agent=experiment_agent,
        uct_c=config['mcts']['uct_constant']
    )

    # Run MCTS
    best_node = mcts.run_search(root, config['mcts']['iterations'])

    # Results
    print("\n=== BEST NODE FOUND ===")
    print(f"Params: {best_node.params}")
    print(f"Param history: {mcts.param_history}")
    print(f"g-factor history: {mcts.g_history}")
    exploration_dist = average_exploration_distance(mcts.param_history)
    print(f"Average exploration distance: {exploration_dist}")

    g_factor = experiment_agent.run_with_params(best_node.params)
    if g_factor is not None:
        print(f"g-factor (from virtual_lab): {g_factor}")
    else:
        print("Experiment failed to return g-factor.")

    results_dict = {
        "BestNodeParams": best_node.params,
        "ParamHistory": mcts.param_history,
        "GFactorHistory": mcts.g_history,
        "AverageExplorationDistance": exploration_dist,
        "FinalGFactor": g_factor
    }

    output_path = os.path.join(os.path.dirname(__file__), "master_experiments.json")
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n[INFO] Results saved to {output_path}")

if __name__ == "__main__":
    main()
