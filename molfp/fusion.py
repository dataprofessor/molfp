"""
Consensus/fusion fingerprints: dataset-aware optimal bit selection across FP types.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .fingerprints import _batch, FINGERPRINTS


def fuse(smiles_list: list[str], activities: list[float] | np.ndarray, fp_types: list[str] | None = None, n_bits: int = 500,
         method: str = "mutual_info", task: str = "regression") -> dict[str, Any]:
    """Build an optimized fusion fingerprint from multiple FP types.

    Selects the top N most informative bits across all fingerprint types
    based on mutual information with the target activities.

    Args:
        smiles_list: list of SMILES
        activities: list/array of activity values
        fp_types: fingerprint types to combine (default: all binary FPs)
        n_bits: number of bits to select
        method: feature selection method ("mutual_info" or "variance")
        task: "regression" or "classification"

    Returns:
        dict with:
            fused_fps: numpy array (n_samples, n_bits)
            selected_bits: list of (fp_type, bit_idx, score) tuples
            source_map: {col_idx: (fp_type, bit_idx)}
    """
    if fp_types is None:
        fp_types = ["maccs", "ecfp4", "pubchem", "substructure", "klekota_roth",
                     "estate", "atompairs2d", "rdk"]

    activities = np.asarray(activities, dtype=float)

    # Compute all fingerprints and concatenate
    all_fps = []
    all_labels = []  # (fp_type, bit_idx) for each column

    for fp_type in fp_types:
        fn, fp_size, _ = FINGERPRINTS[fp_type]
        fps = _batch(fn, smiles_list)
        all_fps.append(fps)
        for i in range(fps.shape[1]):
            all_labels.append((fp_type, i))

    X = np.hstack(all_fps)

    # Score each bit
    if method == "mutual_info":
        if task == "classification":
            scores = mutual_info_classif(X, activities, random_state=42)
        else:
            scores = mutual_info_regression(X, activities, random_state=42)
    elif method == "variance":
        scores = np.var(X, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Select top N
    n_bits = min(n_bits, len(scores))
    top_idx = np.argsort(scores)[::-1][:n_bits]

    selected = [(all_labels[i][0], all_labels[i][1], float(scores[i])) for i in top_idx]
    source_map = {j: (all_labels[i][0], all_labels[i][1]) for j, i in enumerate(top_idx)}
    fused = X[:, top_idx]

    return {
        "fused_fps": fused,
        "selected_bits": selected,
        "source_map": source_map,
        "n_bits": n_bits,
        "method": method,
        "fp_types_used": fp_types,
    }


def complementarity(smiles_list: list[str], fp_types: list[str] | None = None, sample_size: int = 500) -> dict[str, Any]:
    """Measure redundancy between fingerprint types.

    Returns a dict with:
        correlation_matrix: pairwise Jaccard similarity of ON-bit patterns
        unique_info: fraction of bits unique to each FP type
    """
    if fp_types is None:
        fp_types = ["maccs", "ecfp4", "pubchem", "substructure", "rdk"]

    # Subsample for speed
    if len(smiles_list) > sample_size:
        idx = np.random.RandomState(42).choice(len(smiles_list), sample_size, replace=False)
        smiles_list = [smiles_list[i] for i in idx]

    fps_dict = {}
    for fp_type in fp_types:
        fn = FINGERPRINTS[fp_type][0]
        fps_dict[fp_type] = _batch(fn, smiles_list)

    n = len(fp_types)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            density_i = fps_dict[fp_types[i]].mean(axis=1)
            density_j = fps_dict[fp_types[j]].mean(axis=1)
            if np.std(density_i) > 0 and np.std(density_j) > 0:
                corr[i, j] = float(np.corrcoef(density_i, density_j)[0, 1])
            else:
                corr[i, j] = 0.0

    return {
        "fp_types": fp_types,
        "correlation_matrix": corr.tolist(),
    }
