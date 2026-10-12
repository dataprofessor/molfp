"""
Scaffold extraction, fingerprinting, and scaffold-aware data splitting.

Murcko scaffolds identify the core ring system of a molecule after stripping
side chains — critical for scaffold hopping, series analysis, diversity
assessment, and preventing scaffold leakage in QSAR train/test splits.
"""

from __future__ import annotations

from typing import Union
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .fingerprints import _ensure_mol, _extract_smiles


def scaffold(input_data, *, column=None):
    """Extract Murcko and generic scaffolds from molecule(s).

    Args:
        input_data: SMILES string, list, Series, or DataFrame.
        column: column name for DataFrame input.

    Returns:
        Single molecule: dict with murcko, generic, smiles.
        Multiple molecules: list of dicts.
    """
    data, is_batch = _extract_smiles(input_data, column=column)

    if not is_batch:
        return _scaffold_single(data)

    return [_scaffold_single(s) for s in data]


def _scaffold_single(smiles_or_mol):
    """Extract scaffolds for a single molecule."""
    mol = _ensure_mol(smiles_or_mol)
    smiles = Chem.MolToSmiles(mol)

    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        murcko = Chem.MolToSmiles(core)
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(core)
        generic = Chem.MolToSmiles(generic_mol)
    except Exception:
        murcko = ""
        generic = ""

    return {
        "smiles": smiles,
        "murcko": murcko,
        "generic": generic,
    }


def scaffold_fingerprint(input_data, *, column=None, scaffold_type="murcko"):
    """Create a binary scaffold membership fingerprint.

    Each column represents a unique scaffold found in the dataset.
    A molecule gets a 1 in the column matching its scaffold.

    Args:
        input_data: list of SMILES, Series, or DataFrame.
        column: column name for DataFrame input.
        scaffold_type: "murcko" or "generic".

    Returns:
        dict with:
            X: numpy array (n_samples, n_scaffolds) — binary membership
            scaffolds: list of scaffold SMILES (column order)
            assignments: list of scaffold SMILES per molecule
    """
    data, is_batch = _extract_smiles(input_data, column=column)
    if not is_batch:
        data = [data]

    # Extract scaffolds
    assignments = []
    for s in data:
        info = _scaffold_single(s)
        assignments.append(info[scaffold_type])

    # Build vocabulary
    unique_scaffolds = sorted(set(assignments))
    scaffold_to_idx = {s: i for i, s in enumerate(unique_scaffolds)}

    # Binary matrix
    X = np.zeros((len(data), len(unique_scaffolds)), dtype=np.int32)
    for i, scaf in enumerate(assignments):
        if scaf in scaffold_to_idx:
            X[i, scaffold_to_idx[scaf]] = 1

    return {
        "X": X,
        "scaffolds": unique_scaffolds,
        "assignments": assignments,
        "n_scaffolds": len(unique_scaffolds),
    }


def scaffold_split(input_data, *, column=None, test_size=0.2, scaffold_type="murcko"):
    """Scaffold-aware train/test split — prevents scaffold leakage.

    Groups molecules by scaffold, then assigns entire groups to train or test.
    Ensures no scaffold appears in both sets — critical for fair QSAR evaluation.

    Args:
        input_data: list of SMILES, Series, or DataFrame.
        column: column name for DataFrame input.
        test_size: fraction of molecules for test set (approximate).
        scaffold_type: "murcko" or "generic".

    Returns:
        dict with:
            train_idx: numpy array of training indices
            test_idx: numpy array of test indices
            n_train: number of training molecules
            n_test: number of test molecules
            n_scaffolds_train: number of scaffolds in train
            n_scaffolds_test: number of scaffolds in test
            scaffold_groups: dict mapping scaffold SMILES → list of molecule indices
    """
    data, is_batch = _extract_smiles(input_data, column=column)
    if not is_batch:
        data = [data]

    n = len(data)

    # Group molecules by scaffold
    scaffold_groups = defaultdict(list)
    for i, s in enumerate(data):
        info = _scaffold_single(s)
        scaf = info[scaffold_type]
        scaffold_groups[scaf].append(i)

    # Sort groups by size (largest first) for deterministic splitting
    sorted_groups = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Assign groups to test until test_size is reached
    test_idx = []
    train_idx = []
    target_test = int(n * test_size)

    for scaf, indices in sorted_groups:
        if len(test_idx) + len(indices) <= target_test:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    # If test is empty (all groups too large), put the smallest group in test
    if not test_idx and sorted_groups:
        scaf, indices = sorted_groups[-1]
        test_idx = indices
        train_idx = [i for i in range(n) if i not in set(test_idx)]

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    test_idx = np.array(sorted(test_idx), dtype=np.int64)

    train_scaffolds = set()
    test_scaffolds = set()
    for scaf, indices in sorted_groups:
        if indices[0] in set(train_idx):
            train_scaffolds.add(scaf)
        else:
            test_scaffolds.add(scaf)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_scaffolds_train": len(train_scaffolds),
        "n_scaffolds_test": len(test_scaffolds),
        "scaffold_groups": dict(scaffold_groups),
    }
