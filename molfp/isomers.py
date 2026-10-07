"""
Structural isomer awareness: detect when different molecules produce identical fingerprints.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from collections import defaultdict

from .fingerprints import _batch, FINGERPRINTS


def find_identical_fps(smiles_list: list[Union[str, Chem.rdchem.Mol]], fp_type: str = "ecfp4") -> list[list[tuple[int, str]]]:
    """Find groups of distinct molecules that produce identical fingerprints.

    Returns list of groups, where each group is a list of (index, smiles).
    Only groups with 2+ members are returned.
    """
    fn = FINGERPRINTS[fp_type][0]
    fp_to_mols = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        try:
            fp = tuple(fn(smi))
            fp_to_mols[fp].append((i, smi))
        except (ValueError, Exception):
            continue

    return [group for group in fp_to_mols.values() if len(group) > 1]


def discriminative_power(smiles_list: list[Union[str, Chem.rdchem.Mol]], fp_types: list[str] | None = None) -> list[dict[str, Any]]:
    """Measure how well each fingerprint type distinguishes molecules.

    Returns list of dicts with:
        fp_type, total_molecules, unique_fps, discrimination_ratio
    Sorted by discrimination_ratio descending.
    """
    if fp_types is None:
        fp_types = list(FINGERPRINTS.keys())

    results = []
    for fp_type in fp_types:
        fn = FINGERPRINTS[fp_type][0]
        seen = set()
        total = 0

        for smi in smiles_list:
            try:
                fp = tuple(fn(smi))
                seen.add(fp)
                total += 1
            except (ValueError, Exception):
                continue

        ratio = len(seen) / total if total > 0 else 0
        results.append({
            "fp_type": fp_type,
            "total_molecules": total,
            "unique_fps": len(seen),
            "duplicates": total - len(seen),
            "discrimination_ratio": float(ratio),
        })

    return sorted(results, key=lambda r: r["discrimination_ratio"], reverse=True)


def isomer_sensitivity(smiles_pairs: list[tuple[str, str]], fp_types: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Test fingerprint sensitivity on known isomer pairs.

    Args:
        smiles_pairs: list of (smiles_a, smiles_b) tuples of known isomers

    Returns dict mapping fp_type → fraction of pairs distinguished (different FPs).
    """
    if fp_types is None:
        fp_types = list(FINGERPRINTS.keys())

    results = {}
    for fp_type in fp_types:
        fn = FINGERPRINTS[fp_type][0]
        distinguished = 0
        total = 0

        for smi_a, smi_b in smiles_pairs:
            try:
                fp_a = tuple(fn(smi_a))
                fp_b = tuple(fn(smi_b))
                total += 1
                if fp_a != fp_b:
                    distinguished += 1
            except (ValueError, Exception):
                continue

        results[fp_type] = {
            "total_pairs": total,
            "distinguished": distinguished,
            "sensitivity": distinguished / total if total > 0 else 0,
        }

    return results
