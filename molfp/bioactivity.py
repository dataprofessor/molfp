"""
Bioactivity class profiling: analyze fingerprint patterns across activity classes.

Given molecules with activity values (IC50, pIC50, pChEMBL, Ki, pKi),
bin them into classes and analyze intra-class similarity vs inter-class
separation, plus identify bits enriched in each class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from collections import defaultdict

from .fingerprints import _batch, FINGERPRINTS
from .similarity import similarity_matrix


# ── Activity binning ──────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "pchembl": {"active": 6.0, "inactive": 5.0},
    "pic50": {"active": 6.0, "inactive": 5.0},
    "pki": {"active": 6.0, "inactive": 5.0},
    "ic50_nm": {"active": 1000, "inactive": 10000},
    "ki_nm": {"active": 1000, "inactive": 10000},
}


def classify(activities: list[float] | np.ndarray, metric: str = "pchembl", thresholds: dict[str, float] | None = None) -> list[str]:
    """Bin activity values into active/intermediate/inactive classes.

    Args:
        activities: list/array of activity values
        metric: one of "pchembl", "pic50", "pki" (higher = more active)
                or "ic50_nm", "ki_nm" (lower = more active)
        thresholds: optional dict with "active" and "inactive" cutoffs
                    (overrides metric defaults)

    Returns:
        list of class labels ("active", "intermediate", "inactive")
    """
    activities = np.asarray(activities, dtype=float)
    t = thresholds or DEFAULT_THRESHOLDS.get(metric)
    if t is None:
        raise ValueError(f"Unknown metric: {metric}. Use one of {list(DEFAULT_THRESHOLDS.keys())} or pass thresholds=")

    inverted = metric in ("ic50_nm", "ki_nm")
    labels = []
    for val in activities:
        if np.isnan(val):
            labels.append("unassigned")
        elif inverted:
            if val <= t["active"]:
                labels.append("active")
            elif val >= t["inactive"]:
                labels.append("inactive")
            else:
                labels.append("intermediate")
        else:
            if val > t["active"]:
                labels.append("active")
            elif val < t["inactive"]:
                labels.append("inactive")
            else:
                labels.append("intermediate")
    return labels


# ── Class-level similarity analysis ──────────────────────────────────────────

def class_similarity(smiles_list: list[str], labels: list[str], fp_type: str = "ecfp4") -> dict[str, Any]:
    """Compute intra-class and inter-class similarity statistics.

    Args:
        smiles_list: list of SMILES
        labels: list of class labels (same length)
        fp_type: fingerprint type

    Returns:
        dict with:
            intra_class: {class: {mean, std, min, max, n_pairs}}
            inter_class: {(class_a, class_b): {mean, std, min, max, n_pairs}}
            separation: {(class_a, class_b): inter_mean - avg(intra_a, intra_b)}
            class_counts: {class: count}
    """
    n = len(smiles_list)
    if len(labels) != n:
        raise ValueError(f"Length mismatch: {n} SMILES vs {len(labels)} labels")

    mat = similarity_matrix(smiles_list, fp_type)

    # Group indices by class
    class_idx = defaultdict(list)
    for i, label in enumerate(labels):
        class_idx[label].append(i)

    classes = sorted(class_idx.keys())

    # Intra-class
    intra = {}
    for cls in classes:
        idx = class_idx[cls]
        if len(idx) < 2:
            intra[cls] = {"mean": 0, "std": 0, "min": 0, "max": 0, "n_pairs": 0}
            continue
        pairs = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairs.append(mat[idx[i], idx[j]])
        pairs = np.array(pairs)
        intra[cls] = {
            "mean": float(np.mean(pairs)),
            "std": float(np.std(pairs)),
            "min": float(np.min(pairs)),
            "max": float(np.max(pairs)),
            "n_pairs": len(pairs),
        }

    # Inter-class
    inter = {}
    for ci, cls_a in enumerate(classes):
        for cls_b in classes[ci + 1:]:
            pairs = []
            for i in class_idx[cls_a]:
                for j in class_idx[cls_b]:
                    pairs.append(mat[i, j])
            pairs = np.array(pairs)
            inter[(cls_a, cls_b)] = {
                "mean": float(np.mean(pairs)),
                "std": float(np.std(pairs)),
                "min": float(np.min(pairs)),
                "max": float(np.max(pairs)),
                "n_pairs": len(pairs),
            }

    # Separation score: how much lower is inter-class similarity vs intra-class
    separation = {}
    for (cls_a, cls_b), inter_stats in inter.items():
        avg_intra = (intra[cls_a]["mean"] + intra[cls_b]["mean"]) / 2
        separation[(cls_a, cls_b)] = float(avg_intra - inter_stats["mean"])

    return {
        "intra_class": intra,
        "inter_class": inter,
        "separation": separation,
        "class_counts": {cls: len(idx) for cls, idx in class_idx.items()},
    }


# ── Bit enrichment ───────────────────────────────────────────────────────────

def enriched_bits(smiles_list: list[str], labels: list[str], fp_type: str = "ecfp4", min_enrichment: float = 1.5) -> dict[str, list[tuple[int, float, float, float]]]:
    """Find fingerprint bits enriched in each activity class.

    For each class, computes bit frequency and compares to the overall frequency.
    Enrichment ratio = class_freq / overall_freq.

    Args:
        smiles_list: list of SMILES
        labels: list of class labels
        fp_type: fingerprint type
        min_enrichment: minimum enrichment ratio to report

    Returns:
        dict mapping class → list of (bit_idx, class_freq, overall_freq, enrichment)
        sorted by enrichment descending.
    """
    fn, n_bits, _ = FINGERPRINTS[fp_type]
    fps = _batch(fn, smiles_list)
    labels = list(labels)
    n = len(smiles_list)

    # Overall bit frequency
    overall_freq = fps.mean(axis=0)

    # Per-class frequency
    class_idx = defaultdict(list)
    for i, label in enumerate(labels):
        class_idx[label].append(i)

    result = {}
    for cls, idx in sorted(class_idx.items()):
        class_fps = fps[idx]
        class_freq = class_fps.mean(axis=0)

        enriched = []
        for bit in range(n_bits):
            if overall_freq[bit] > 0:
                ratio = class_freq[bit] / overall_freq[bit]
                if ratio >= min_enrichment and class_freq[bit] > 0:
                    enriched.append((
                        bit,
                        float(class_freq[bit]),
                        float(overall_freq[bit]),
                        float(ratio),
                    ))

        result[cls] = sorted(enriched, key=lambda x: -x[3])

    return result


def depleted_bits(smiles_list: list[str], labels: list[str], fp_type: str = "ecfp4", max_depletion: float = 0.5) -> dict[str, list[tuple[int, float, float, float]]]:
    """Find fingerprint bits depleted (underrepresented) in each activity class.

    Depletion ratio = class_freq / overall_freq. Values < max_depletion are reported.

    Returns dict mapping class → list of (bit_idx, class_freq, overall_freq, depletion_ratio).
    """
    fn, n_bits, _ = FINGERPRINTS[fp_type]
    fps = _batch(fn, smiles_list)

    overall_freq = fps.mean(axis=0)

    class_idx = defaultdict(list)
    for i, label in enumerate(labels):
        class_idx[label].append(i)

    result = {}
    for cls, idx in sorted(class_idx.items()):
        class_fps = fps[idx]
        class_freq = class_fps.mean(axis=0)

        depleted = []
        for bit in range(n_bits):
            if overall_freq[bit] > 0.05:
                ratio = class_freq[bit] / overall_freq[bit]
                if ratio <= max_depletion:
                    depleted.append((
                        bit,
                        float(class_freq[bit]),
                        float(overall_freq[bit]),
                        float(ratio),
                    ))

        result[cls] = sorted(depleted, key=lambda x: x[3])

    return result


# ── Multi-FP class profiling ─────────────────────────────────────────────────

def profile(smiles_list: list[str], activities: list[float] | np.ndarray, metric: str = "pchembl", thresholds: dict[str, float] | None = None,
            fp_types: list[str] | None = None) -> dict[str, Any]:
    """Full bioactivity class profiling across multiple fingerprint types.

    Args:
        smiles_list: list of SMILES
        activities: list of activity values
        metric: activity metric type
        thresholds: optional custom thresholds
        fp_types: fingerprint types to compare

    Returns:
        dict with:
            labels: class labels for each molecule
            class_counts: {class: count}
            fp_results: {fp_type: class_similarity result}
            best_separator: fp_type with highest mean separation
    """
    if fp_types is None:
        fp_types = ["maccs", "ecfp4", "pubchem", "substructure", "rdk"]

    labels = classify(activities, metric, thresholds)

    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1

    fp_results = {}
    for fp_type in fp_types:
        try:
            fp_results[fp_type] = class_similarity(smiles_list, labels, fp_type)
        except Exception as e:
            fp_results[fp_type] = {"error": str(e)}

    # Find best separator
    best_fp = None
    best_sep = -1
    for fp_type, res in fp_results.items():
        if "separation" in res:
            mean_sep = np.mean(list(res["separation"].values())) if res["separation"] else 0
            if mean_sep > best_sep:
                best_sep = mean_sep
                best_fp = fp_type

    return {
        "labels": labels,
        "class_counts": class_counts,
        "fp_results": fp_results,
        "best_separator": best_fp,
        "best_separation_score": float(best_sep) if best_sep > -1 else None,
    }
