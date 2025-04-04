from typing import Tuple, List

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error.
    
    Args:
        y_true: Binary true labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for computing ECE
        
    Returns:
        ece: Expected Calibration Error
        bin_confs: Mean predicted probability for each bin
        bin_accs: Mean true label for each bin
        bin_sizes: Number of samples in each bin
    """
    # Input validation
    if not np.all((0 <= y_prob) & (y_prob <= 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    
    if y_true.shape != y_prob.shape:
        raise ValueError("Shape mismatch between labels and probabilities")
    
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    
    # Compute ECE
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)
    
    for bin_idx in range(n_bins):
        mask = binids == bin_idx
        if np.any(mask):
            bin_confs[bin_idx] = y_prob[mask].mean()
            bin_accs[bin_idx] = y_true[mask].mean()
            bin_sizes[bin_idx] = mask.sum()
    
    # Calculate ECE as maximum absolute difference for bins with samples
    valid_bins = bin_sizes > 0
    if np.any(valid_bins):
        ece = np.max(np.abs(bin_accs[valid_bins] - bin_confs[valid_bins]))
    else:
        ece = 0.0
    
    return ece, bin_confs, bin_accs, bin_sizes


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability curve coordinates.
    
    Args:
        y_true: Binary true labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        mean_predicted_value: Mean predicted probability for each bin
        fraction_of_positives: Fraction of positive samples for each bin
    """
    _, mean_predicted_value, fraction_of_positives, _ = expected_calibration_error(
        y_true, y_prob, n_bins
    )
    return mean_predicted_value, fraction_of_positives


def calibration_drift(
    reference_scores: np.ndarray,
    reference_labels: np.ndarray,
    current_scores: np.ndarray,
    current_labels: np.ndarray,
    metric: str = 'ece',
    n_bins: int = 10
) -> float:
    """Compute calibration drift between reference and current data.
    
    Args:
        reference_scores: Predicted probabilities from reference period
        reference_labels: True labels from reference period
        current_scores: Predicted probabilities from current period
        current_labels: True labels from current period
        metric: Metric to use ('ece' or 'brier')
        n_bins: Number of bins for ECE computation
        
    Returns:
        drift: Absolute difference in calibration metric
    """
    if metric == 'ece':
        ref_metric, _, _, _ = expected_calibration_error(
            reference_labels, reference_scores, n_bins
        )
        curr_metric, _, _, _ = expected_calibration_error(
            current_labels, current_scores, n_bins
        )
    elif metric == 'brier':
        ref_metric = brier_score_loss(reference_labels, reference_scores)
        curr_metric = brier_score_loss(current_labels, current_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return abs(curr_metric - ref_metric)
