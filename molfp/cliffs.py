"""
Activity cliff detection: find molecule pairs with high structural similarity
but large activity differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fingerprints import _batch, FINGERPRINTS
from .similarity import similarity_matrix


def activity_cliffs(smiles_list: list[str], activities: list[float] | np.ndarray, fp_type: str = "ecfp4",
                    sim_threshold: float = 0.8, activity_threshold: float = 2.0, metric: str = "tanimoto") -> list[dict[str, Any]]:
    """Find activity cliff pairs in a dataset.

    An activity cliff is a pair of molecules with high structural similarity
    (Tanimoto >= sim_threshold) but large activity difference
    (|activity_a - activity_b| >= activity_threshold).

    Args:
        smiles_list: list of SMILES strings
        activities: list/array of activity values (e.g., pIC50)
        fp_type: fingerprint to use for similarity
        sim_threshold: minimum similarity to be considered structurally similar
        activity_threshold: minimum activity difference for a cliff
        metric: similarity metric ("tanimoto" or "dice")

    Returns:
        list of dicts, each with:
            idx_a, idx_b, smiles_a, smiles_b,
            similarity, activity_a, activity_b, delta_activity
        Sorted by delta_activity descending.
    """
    activities = np.asarray(activities, dtype=float)
    n = len(smiles_list)

    if len(activities) != n:
        raise ValueError(f"Length mismatch: {n} SMILES vs {len(activities)} activities")

    mat = similarity_matrix(smiles_list, fp_type, metric)

    cliffs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = mat[i, j]
            if sim >= sim_threshold:
                delta = abs(activities[i] - activities[j])
                if delta >= activity_threshold:
                    cliffs.append({
                        "idx_a": i,
                        "idx_b": j,
                        "smiles_a": smiles_list[i],
                        "smiles_b": smiles_list[j],
                        "similarity": float(sim),
                        "activity_a": float(activities[i]),
                        "activity_b": float(activities[j]),
                        "delta_activity": float(delta),
                    })

    return sorted(cliffs, key=lambda c: c["delta_activity"], reverse=True)


def cliff_summary(smiles_list: list[str], activities: list[float] | np.ndarray, fp_types: list[str] | None = None,
                   sim_threshold: float = 0.8, activity_threshold: float = 2.0) -> list[dict[str, Any]]:
    """Compare activity cliff counts across fingerprint types.

    Returns list of {fp_type, n_cliffs, max_delta, mean_delta}.
    """
    if fp_types is None:
        fp_types = ["maccs", "ecfp4", "pubchem", "substructure", "rdk"]

    results = []
    for fp_type in fp_types:
        try:
            cliffs = activity_cliffs(smiles_list, activities, fp_type,
                                     sim_threshold, activity_threshold)
            deltas = [c["delta_activity"] for c in cliffs]
            results.append({
                "fp_type": fp_type,
                "n_cliffs": len(cliffs),
                "max_delta": max(deltas) if deltas else 0,
                "mean_delta": float(np.mean(deltas)) if deltas else 0,
            })
        except Exception as e:
            results.append({"fp_type": fp_type, "error": str(e)})

    return sorted(results, key=lambda r: r.get("n_cliffs", 0), reverse=True)
