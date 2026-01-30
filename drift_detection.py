import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)
    
    if len(np.unique(expected)) < 2:
        return 0
    
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] -= 1e-5
    breakpoints[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)

    

    return np.sum(psi)


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.2):
    """
    Calculates PSI for each feature
    Returns: dict {feature: psi_value}
    """
    drift_results = {}

    for col in reference_df.columns:
        if col in current_df.columns:
            try:
                psi_value = calculate_psi(reference_df[col], current_df[col])
                drift_results[col] = psi_value
            except Exception:
                # Skip non-numeric columns safely
                continue

    return drift_results
