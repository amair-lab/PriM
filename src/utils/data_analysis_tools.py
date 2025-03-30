import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, kendalltau

def analyze_data_distribution(var_records, g_factor_results):
    """
    Analyze the data distribution using StandardScaler.

    Args:
        var_records (dict): Dictionary of variable values for each variable.
        g_factor_results (list): Corresponding g-factor results.

    Returns:
        dict: Distribution statistics.
    """
    distribution_stats = {}
    scaler = StandardScaler()

    for var, values in var_records.items():
        scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1))
        distribution_stats[var] = {
            "mean": np.mean(scaled_values),
            "std": np.std(scaled_values)
        }

    scaled_g_factor = scaler.fit_transform(np.array(g_factor_results).reshape(-1, 1))
    distribution_stats["g_factor"] = {
        "mean": np.mean(scaled_g_factor),
        "std": np.std(scaled_g_factor)
    }

    return distribution_stats

def correlation_analysis(var_records, g_factor_results):
    """
    Analyze the correlation between each variable and g-factor results.

    Args:
        var_records (dict): Dictionary of variable values for each variable.
        g_factor_results (list): Corresponding g-factor results.

    Returns:
        dict: Correlation analysis results for each variable.
    """
    correlation_stats = {}

    for var, values in var_records.items():
        pearson_corr, pearson_p = pearsonr(values, g_factor_results)
        spearman_corr, spearman_p = spearmanr(values, g_factor_results)
        kendall_corr, kendall_p = kendalltau(values, g_factor_results)

        correlation_stats[var] = {
            "pearson": {"correlation": pearson_corr, "p_value": pearson_p},
            "spearman": {"correlation": spearman_corr, "p_value": spearman_p},
            "kendall": {"correlation": kendall_corr, "p_value": kendall_p}
        }

    return correlation_stats

def identify_critical_values(var_records, g_factor_results):
    """
    Identify the min, max, and largest g-factor with corresponding parameters.

    Args:
        var_records (dict): Dictionary of variable values for each variable.
        g_factor_results (list): Corresponding g-factor results.

    Returns:
        dict: Critical values summary.
    """
    max_g_factor_idx = np.argmax(g_factor_results)
    max_g_factor = g_factor_results[max_g_factor_idx]
    optimal_params = {var: values[max_g_factor_idx] for var, values in var_records.items()}

    return {
        "maximum_g_factor": max_g_factor,
        "optimal_parameters": optimal_params
    }

def polynomial_fitting(var_records, g_factor_results, degree):
    """
    Fit a polynomial curve to the experimental data for each variable.

    Args:
        var_records (dict): Dictionary of variable values for each variable.
        g_factor_results (list): Corresponding g-factor results.
        degree (int): Degree of the polynomial.

    Returns:
        dict: Polynomial coefficients and fitted values for each variable.
    """
    polynomial_fit = {}

    for var, values in var_records.items():
        coefficients = np.polyfit(values, g_factor_results, degree)
        polynomial = np.poly1d(coefficients)
        fitted_values = polynomial(values)

        polynomial_fit[var] = {
            "polynomial_coefficients": coefficients.tolist(),
            "fitted_values": fitted_values.tolist()
        }

    return polynomial_fit

def data_analysis_tool(var_records, g_factor_results, degree):
    """
    This tool integrates the data analysis results from `analyze_data_distribution`, `correlation_analysis`,
    `polynomial_fitting`, and `identify_critical_values`, which will be processed by AnalysisAgent.

    Args:
        var_records (dict): Dictionary of variable values for each variable.
        g_factor_results (list): Corresponding g-factor results.
        degree (int): Degree of the polynomial fitting.

    Returns:
        dict: The analysis result, which is OpenAI API-readable.
    """
    distribution_stats = analyze_data_distribution(var_records, g_factor_results)
    correlation_stats = correlation_analysis(var_records, g_factor_results)
    critical_values = identify_critical_values(var_records, g_factor_results)
    polynomial_fit = polynomial_fitting(var_records, g_factor_results, degree)

    result = {
        "distribution_statistics": distribution_stats,
        "correlation_analysis": correlation_stats,
        "critical_values": critical_values,
        "polynomial_fitting": polynomial_fit
    }

    return result