"""
Similarity diagnostics: characterize how fingerprints distribute similarity scores.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from .fingerprints import _batch, FINGERPRINTS
from .distance import pairwise as pairwise_sim


def similarity_matrix(smiles_list: list[str], fp_type: str = "ecfp4",
                      metric: str = "tanimoto") -> np.ndarray:
    """Compute pairwise similarity matrix for a list of molecules.

    Uses vectorized computation from the distance module.

    Returns (N, N) numpy array of float32.
    """
    fn, n_bits, _ = FINGERPRINTS[fp_type]
    fps = _batch(fn, smiles_list)
    return pairwise_sim(fps, metric=metric)


def similarity_profile(smiles_list: list[str], fp_type: str = "ecfp4",
                        metric: str = "tanimoto") -> dict:
    """Characterize the similarity distribution for a fingerprint on a dataset.

    Returns dict with statistical summary of pairwise similarities.
    """
    mat = similarity_matrix(smiles_list, fp_type, metric)
    n = len(smiles_list)

    if n < 2:
        return {
            "fp_type": fp_type,
            "n_molecules": n,
            "n_pairs": 0,
            "mean": 0.0, "std": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0,
            "histogram": [0] * 20,
            "bin_edges": np.linspace(0, 1, 21).tolist(),
            "pathological": True,
        }

    triu_idx = np.triu_indices(n, k=1)
    pw = mat[triu_idx]

    hist, bin_edges = np.histogram(pw, bins=20, range=(0, 1))

    return {
        "fp_type": fp_type,
        "n_molecules": n,
        "n_pairs": len(pw),
        "mean": float(np.mean(pw)),
        "std": float(np.std(pw)),
        "median": float(np.median(pw)),
        "min": float(np.min(pw)),
        "max": float(np.max(pw)),
        "q25": float(np.percentile(pw, 25)),
        "q75": float(np.percentile(pw, 75)),
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "pathological": float(np.std(pw)) < 0.05,
    }


def compare_fps(smiles_list: list[str], fp_types: list[str] | None = None,
                metric: str = "tanimoto") -> list[dict]:
    """Compare similarity distributions across multiple fingerprint types.

    Returns list of profile dicts, sorted by mean similarity.
    """
    if fp_types is None:
        fp_types = ["maccs", "ecfp4", "pubchem", "substructure", "klekota_roth",
                     "estate", "atompairs2d", "rdk"]

    profiles = []
    for fp_type in fp_types:
        try:
            prof = similarity_profile(smiles_list, fp_type, metric)
            profiles.append(prof)
        except Exception as e:
            profiles.append({"fp_type": fp_type, "error": str(e)})

    return sorted(profiles, key=lambda p: p.get("mean", 0), reverse=True)
