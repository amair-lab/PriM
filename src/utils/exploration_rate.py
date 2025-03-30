import math
from itertools import combinations

def average_exploration_distance(param_history):
    """
    Computes the average pairwise Euclidean distance between all distinct pairs
    of parameter vectors in param_history.

    param_history: List of lists or tuples, where each inner list/tuple is a parameter vector.
    Returns: A float representing the average pairwise distance.
    """
    N = len(param_history)
    if N < 2:
        return 0.0

    total_distance = 0.0

    for x, y in combinations(param_history, 2):
        dist_sq = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        total_distance += math.sqrt(dist_sq)

    return (2 * total_distance) / (N * (N - 1))
