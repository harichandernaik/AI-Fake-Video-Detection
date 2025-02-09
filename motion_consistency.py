import numpy as np

def check_motion_consistency(motion_data):
    """Analyzes motion consistency and returns relevant statistics."""

    mean_magnitude = np.mean(motion_data)  # Compute mean
    median_magnitude = np.median(motion_data)  # Compute median
    motion_variance = np.var(motion_data)  # Compute variance
    motion_score = mean_magnitude * (motion_variance + 1)  # Custom score formula

    return mean_magnitude, median_magnitude, motion_variance, motion_score
