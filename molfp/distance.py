"""
Distance and similarity functions for molecular fingerprints.

Provides vectorized Tanimoto, Dice, Cosine, and Sokal-Michener metrics
for both binary and count fingerprints, operating on single pairs or
full pairwise matrices.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.sparse import issparse


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two fingerprint vectors.

    Works for both binary (Jaccard) and count vectors.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.dot(a, b)
    denom = np.dot(a, a) + np.dot(b, b) - dot
    return float(dot / denom) if denom > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity between two fingerprint vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.dot(a, b)
    denom = np.dot(a, a) + np.dot(b, b)
    return float(2 * dot / denom) if denom > 0 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two fingerprint vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.dot(a, b)
    norm = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(dot / norm) if norm > 0 else 0.0


def sokal_michener(a: np.ndarray, b: np.ndarray) -> float:
    """Sokal-Michener (simple matching) similarity for binary vectors."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    matches = np.sum(a == b)
    return float(matches / len(a)) if len(a) > 0 else 0.0


def russel_rao(a: np.ndarray, b: np.ndarray) -> float:
    """Russel-Rao similarity for binary vectors (shared ON bits / total bits)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    both_on = np.sum((a == 1) & (b == 1))
    return float(both_on / len(a)) if len(a) > 0 else 0.0


# ── Pairwise matrix computation (vectorized) ─────────────────────────────────

_METRICS = {
    "tanimoto": tanimoto,
    "dice": dice,
    "cosine": cosine,
    "sokal_michener": sokal_michener,
    "russel_rao": russel_rao,
}


def pairwise(X: np.ndarray, metric: str = "tanimoto") -> np.ndarray:
    """Compute pairwise similarity matrix for a fingerprint matrix.

    Uses vectorized computation for Tanimoto; falls back to pairwise
    loop for other metrics.

    Args:
        X: (n_samples, n_bits) fingerprint matrix
        metric: "tanimoto", "dice", "cosine", "sokal_michener", "russel_rao"

    Returns:
        (n_samples, n_samples) symmetric similarity matrix
    """
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if metric == "tanimoto":
        return _pairwise_tanimoto_vectorized(X)
    elif metric == "dice":
        return _pairwise_dice_vectorized(X)
    elif metric == "cosine":
        return _pairwise_cosine_vectorized(X)
    else:
        fn = _METRICS.get(metric)
        if fn is None:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(_METRICS.keys())}")
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                s = fn(X[i], X[j])
                mat[i, j] = mat[j, i] = s
        return mat


def _pairwise_tanimoto_vectorized(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Tanimoto for dense fingerprint matrix."""
    dot = X @ X.T
    norms_sq = np.diag(dot).copy()
    denom = norms_sq[:, None] + norms_sq[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0, dot / denom, 0.0)
    return result.astype(np.float32)


def _pairwise_dice_vectorized(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Dice."""
    dot = X @ X.T
    norms_sq = np.diag(dot).copy()
    denom = norms_sq[:, None] + norms_sq[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0, 2 * dot / denom, 0.0)
    return result.astype(np.float32)


def _pairwise_cosine_vectorized(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Cosine."""
    norms = np.sqrt(np.sum(X * X, axis=1))
    norms[norms == 0] = 1.0
    X_normed = X / norms[:, None]
    result = X_normed @ X_normed.T
    return result.astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray, metric: str = "tanimoto") -> float:
    """Distance (1 - similarity) between two fingerprint vectors."""
    fn = _METRICS.get(metric)
    if fn is None:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(_METRICS.keys())}")
    return 1.0 - fn(a, b)


def pairwise_distance(X: np.ndarray, metric: str = "tanimoto") -> np.ndarray:
    """Compute pairwise distance matrix (1 - similarity)."""
    return 1.0 - pairwise(X, metric)


def available_metrics() -> list[str]:
    """Return list of available similarity/distance metric names."""
    return list(_METRICS.keys())
