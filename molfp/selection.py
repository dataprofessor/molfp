"""
Standalone feature selection utilities for any fingerprint matrix.

Works with output from any molfp function: fingerprints, graph_vectors, etc.
"""

from __future__ import annotations

import numpy as np


def remove_low_variance(X, threshold=0.0):
    """Remove features with variance at or below threshold.

    Args:
        X: numpy array (n_samples, n_features).
        threshold: variance cutoff. 0.0 removes constant features,
            0.01 removes near-constant, etc.

    Returns:
        dict with:
            X: filtered array (n_samples, n_remaining)
            mask: boolean mask of kept features
            n_removed: number of features removed
            n_remaining: number of features kept
    """
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return {
        "X": X[:, mask],
        "mask": mask,
        "n_removed": int(np.sum(~mask)),
        "n_remaining": int(np.sum(mask)),
    }


def select_features(X, y, k=500, threshold=None):
    """Select top-k features by mutual information with optional variance pre-filter.

    Args:
        X: numpy array (n_samples, n_features).
        y: labels array (n_samples,) — class labels for mutual information.
        k: number of top features to keep.
        threshold: if set, remove features with variance <= threshold before
            mutual information ranking. Reduces noise and speeds up MI computation.

    Returns:
        dict with:
            X: selected array (n_samples, k)
            indices: original column indices of selected features
            scores: mutual information scores for selected features
            n_input: original number of features
            n_after_variance: features after variance filter (if threshold set)
            n_selected: final number of selected features
    """
    from sklearn.feature_selection import mutual_info_classif

    n_input = X.shape[1]
    original_indices = np.arange(n_input)

    # Optional variance pre-filter
    if threshold is not None:
        variances = np.var(X, axis=0)
        var_mask = variances > threshold
        X = X[:, var_mask]
        original_indices = original_indices[var_mask]
    n_after_variance = X.shape[1]

    # Mutual information ranking
    k = min(k, X.shape[1])
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_k_idx = np.argsort(mi_scores)[::-1][:k]
    top_k_idx_sorted = np.sort(top_k_idx)  # preserve column order

    return {
        "X": X[:, top_k_idx_sorted],
        "indices": original_indices[top_k_idx_sorted],
        "scores": mi_scores[top_k_idx_sorted],
        "n_input": n_input,
        "n_after_variance": n_after_variance,
        "n_selected": k,
    }
