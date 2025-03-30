import optuna
import os, sys, json, math
from itertools import combinations
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from virtual_lab.test_tool import do_experiment


# Parameter space
parameter_space = {
    "angle": [0.123160654, 1.009814211],
    "curl": [0.628318531, 8.078381109],
    "fiber_radius": [20, 60],
    "height": [43.32551229, 954.9296586],
    "helix_radius": [20, 90],
    "n_turns": [3, 10],
    "pitch": [60, 200],
    "total_fiber_length": [303.7757835, 1127.781297],
    "total_length": [300, 650]
}


experiment_log = []
param_history = []
best_rewards_over_time = []

def average_exploration_distance(param_history):
    N = len(param_history)
    if N < 2:
        return 0.0
    total_distance = 0.0
    for x, y in combinations(param_history, 2):
        dist_sq = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        total_distance += math.sqrt(dist_sq)
    return (2 * total_distance) / (N * (N - 1))


def objective(trial):
    parameters = {
        param: trial.suggest_float(param, bounds[0], bounds[1])
        for param, bounds in parameter_space.items()
    }

    param_vector = [parameters[k] for k in parameter_space.keys()]
    param_history.append(param_vector)

    try:
        results = do_experiment(**parameters)
        reward = results['predicted_g_factor'] if results['status'] == 'success' else -1.0
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        reward = -1.0

    experiment_log.append({
        "trial": trial.number + 1,
        "parameters": parameters,
        "reward": reward
    })

    # track best reward so far
    if not best_rewards_over_time:
        best_rewards_over_time.append(reward)
    else:
        best_rewards_over_time.append(max(best_rewards_over_time[-1], reward))

    return reward


# 🔁 Run Optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    avg_distance = average_exploration_distance(param_history)

    epsilon = 1e-3
    best_reward = best_rewards_over_time[-1]
    convergence_trial = next(
        (i for i, r in enumerate(best_rewards_over_time) if abs(r - best_reward) <= epsilon),
        len(best_rewards_over_time)
    )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"optuna_trials_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump({
            "best_reward": study.best_value,
            "best_params": study.best_params,
            "trials": experiment_log,
            "average_exploration_distance": avg_distance,
            "convergence_trial": convergence_trial
        }, f, indent=4)

    print("\nFinal Results:")
    print(f"Best Reward (g-factor): {study.best_value}")
    print("Optimal Parameters:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Average Exploration Distance: {avg_distance:.4f}")
    print(f"Convergence Trial (approx): {convergence_trial}")
