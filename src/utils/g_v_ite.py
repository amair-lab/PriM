import matplotlib.pyplot as plt

def plot_g_factor_results(g_factor_results, file_path):
    """
    Plots the g-factor results over iterations.

    Parameters:
    g_factor_results (list): A list of g-factor values, where the index represents the iteration.
    file_path (str): The path to save the plot.
    """
    iterations = range(1, len(g_factor_results) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, g_factor_results, marker='o', linestyle='-', color='b', label='G-Factor Values')

    plt.xlabel('Iteration')
    plt.ylabel('G-Factor Value')
    plt.title('G-Factor Values Over Iterations')
    plt.grid(True)

    plt.legend()
    plt.savefig(file_path)
    plt.close()