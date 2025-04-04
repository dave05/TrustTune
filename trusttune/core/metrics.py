"""Metrics for calibration evaluation."""
import numpy as np
from typing import Tuple, List

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
        
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if y_true.shape != y_prob.shape:
        raise ValueError("Shape mismatch between labels and predictions")
    
    if not (np.all(y_prob >= 0) and np.all(y_prob <= 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    
    nonzero = bin_total > 0
    bin_acc[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    bin_conf[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    weights = bin_total[nonzero] / np.sum(bin_total)
    ece = float(np.sum(np.abs(bin_acc[nonzero] - bin_conf[nonzero]) * weights))
    
    return ece, bin_conf, bin_acc, bin_total

def calibration_drift(old_scores: np.ndarray, new_scores: np.ndarray, metric: str = "js") -> float:
    """Compute calibration drift between two sets of predictions."""
    if metric not in ["js", "kl", "tv"]:
        raise ValueError("Unknown metric. Supported metrics: js, kl, tv")
        
    old_scores = np.asarray(old_scores)
    new_scores = np.asarray(new_scores)
    
    if not (np.all(old_scores >= 0) and np.all(old_scores <= 1) and 
            np.all(new_scores >= 0) and np.all(new_scores <= 1)):
        raise ValueError("Scores must be between 0 and 1")
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    bins = np.linspace(0, 1, 11)  # 10 bins
    old_hist, _ = np.histogram(old_scores, bins=bins, density=True)
    new_hist, _ = np.histogram(new_scores, bins=bins, density=True)
    
    # Normalize and add epsilon
    old_hist = old_hist / np.sum(old_hist) + eps
    new_hist = new_hist / np.sum(new_hist) + eps
    
    if metric == "js":
        m = 0.5 * (old_hist + new_hist)
        return float(0.5 * (
            np.sum(old_hist * np.log(old_hist / m)) +
            np.sum(new_hist * np.log(new_hist / m))
        ))
    elif metric == "kl":
        return float(np.sum(old_hist * np.log(old_hist / new_hist)))
    else:  # tv
        return float(0.5 * np.sum(np.abs(old_hist - new_hist)))

def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability curve.
    
    Returns:
        Tuple of (mean predicted probability, true fraction) for each bin
    """
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return prob_pred, prob_true

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score."""
    return np.mean((y_true - y_prob) ** 2) 