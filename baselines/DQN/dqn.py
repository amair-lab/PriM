import random, os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import combinations
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from virtual_lab.test_tool import do_experiment

# -------- PARAMETER SETUP --------
parameter_space = {
    "angle": [0.123, 1.01],
    "curl": [0.63, 8.08],
    "fiber_radius": [20, 60],
    "height": [43, 955],
    "helix_radius": [20, 90],
    "n_turns": [3, 10],
    "pitch": [60, 200],
    "total_fiber_length": [304, 1128],
    "total_length": [300, 650]
}

def discretize_space(parameter_space, num_bins=5):
    return {param: np.linspace(bounds[0], bounds[1], num_bins) for param, bounds in parameter_space.items()}

def get_convergence_trial(rewards, best_reward, epsilon=1e-3, patience=3):
    count = 0
    for i in range(len(rewards)):
        if abs(rewards[i] - best_reward) <= epsilon:
            count += 1
            if count >= patience:
                return i - patience + 1
        else:
            count = 0
    return len(rewards)

def average_exploration_distance(param_history):
    N = len(param_history)
    if N < 2: return 0.0
    total_distance = 0.0
    for x, y in combinations(param_history, 2):
        dist_sq = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        total_distance += math.sqrt(dist_sq)
    return (2 * total_distance) / (N * (N - 1))


# -------- NETWORK --------
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -------- AGENT --------
class DQNAgent:
    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=500)
        self.batch_size = 16

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            q_target = reward
            if not done:
                q_target += self.gamma * torch.max(self.model(next_state_tensor))
            q_values = self.model(state_tensor)
            q_values = q_values.clone().detach()
            q_values[action] = q_target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), q_values)
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -------- BASELINE WRAPPER --------
class BaselineDQN:
    def __init__(self, parameter_space, num_bins=5):
        self.parameter_space = parameter_space
        self.discrete_space = discretize_space(parameter_space, num_bins)
        self.num_bins = num_bins
        self.state_size = len(parameter_space)
        self.action_size = len(parameter_space) * num_bins
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.best_reward = float('-inf')
        self.best_parameters = None
        self.trial_log = []
        self.param_history = []
        self.reward_history = []

    def state_from_parameters(self, parameters):
        state = []
        for param, bounds in self.parameter_space.items():
            value = parameters[param]
            state.append((value - bounds[0]) / (bounds[1] - bounds[0]))
        return state

    def parameters_from_action(self, action):
        parameters = {}
        for i, param in enumerate(self.parameter_space.keys()):
            bin_index = action % self.num_bins
            action //= self.num_bins
            parameters[param] = self.discrete_space[param][bin_index]
        return parameters

    def run(self, episodes=20, max_steps=3):
        trial_counter = 0

        for episode in range(episodes):
            parameters = {param: random.uniform(bounds[0], bounds[1]) for param, bounds in self.parameter_space.items()}
            state = self.state_from_parameters(parameters)

            for step in range(max_steps):
                action = self.agent.act(state)
                next_parameters = self.parameters_from_action(action)

                try:
                    results = do_experiment(**next_parameters)
                    reward = results['predicted_g_factor'] if results['status'] == 'success' else -1.0
                except Exception as e:
                    print(f"Experiment failed: {str(e)}")
                    reward = -1.0

                next_state = self.state_from_parameters(next_parameters)
                done = reward > 0
                self.agent.remember(state, action, reward, next_state, done)

                self.param_history.append([next_parameters[k] for k in self.parameter_space])
                self.reward_history.append(reward)

                self.trial_log.append({
                    "trial": trial_counter,
                    "parameters": next_parameters,
                    "reward": reward
                })
                trial_counter += 1

                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_parameters = next_parameters

                state = next_state
                if done:
                    break

            self.agent.replay()

        # Post-run analysis
        self.save_results()
        
    def save_results(self):
        avg_distance = average_exploration_distance(self.param_history)
        convergence_trial = get_convergence_trial(self.reward_history, self.best_reward)

        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"dqn_trials_{timestamp}.json")

        with open(output_path, "w") as f:
            json.dump({
                "best_reward": self.best_reward,
                "best_params": self.best_parameters,
                "trials": self.trial_log,
                "average_exploration_distance": avg_distance,
                "convergence_trial": convergence_trial
            }, f, indent=4)

        print("\nFinal Results:")
        print(f"Best Reward (g-factor): {self.best_reward}")
        print("Best Parameters:")
        for param, value in self.best_parameters.items():
            print(f"{param}: {value}")
        print(f"Average Exploration Distance: {avg_distance:.4f}")
        print(f"Convergence Trial (approx): {convergence_trial}")


# -------- EXECUTION --------
if __name__ == "__main__":
    baseline_dqn = BaselineDQN(parameter_space)
    baseline_dqn.run(episodes=20)
