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


# ── Named scaffold dictionary ────────────────────────────────────────────────
# Maps canonical SMILES of common ring systems to their chemical names.
# Covers the most frequent scaffolds in drug discovery (~150 entries).

_NAMED_SCAFFOLDS = {
    # ── 5-membered rings ─────────────────────────────────────────────
    "C1=CC=CC1": "Cyclopentadiene",
    "C1CCCC1": "Cyclopentane",
    "C1=CCC=C1": "Cyclopentadiene",
    "c1cc[nH]c1": "Pyrrole",
    "c1ccoc1": "Furan",
    "c1ccsc1": "Thiophene",
    "c1cn[nH]c1": "Pyrazole",
    "c1ccnn1": "Pyrazole",
    "c1cnc[nH]1": "Imidazole",
    "c1c[nH]cn1": "Imidazole",
    "c1cnco1": "Oxazole",
    "c1cocn1": "Oxazole",
    "c1cncs1": "Thiazole",
    "c1cscn1": "Thiazole",
    "c1ccno1": "Isoxazole",
    "c1ccon1": "Isoxazole",
    "c1ccns1": "Isothiazole",
    "c1csnn1": "Isothiazole",
    "c1nncn1": "1,2,4-Triazole",
    "c1nn[nH]n1": "Tetrazole",
    "c1nnn[nH]1": "Tetrazole",
    "c1cnnn1": "1,2,3-Triazole",
    "c1cnn[nH]1": "1,2,3-Triazole",
    "c1ncon1": "1,2,4-Oxadiazole",
    "c1nocn1": "1,2,4-Oxadiazole",
    "c1ncsn1": "1,2,4-Thiadiazole",
    "c1nscn1": "1,2,4-Thiadiazole",
    "C1CCNC1": "Pyrrolidine",
    "C1CCOC1": "Tetrahydrofuran",
    "C1CCSC1": "Tetrahydrothiophene",
    "C1CNNC1": "Pyrazolidine",
    "C1CNCC1": "Pyrrolidine",  # alternate canonical form

    # ── 6-membered rings ─────────────────────────────────────────────
    "c1ccccc1": "Benzene",
    "C1CCCCC1": "Cyclohexane",
    "C1=CCCCC1": "Cyclohexene",
    "c1ccncc1": "Pyridine",
    "c1ccccn1": "Pyridine",
    "c1ccnnc1": "Pyridazine",
    "c1cnccn1": "Pyrimidine",
    "c1ccncn1": "Pyrimidine",
    "c1cnncn1": "1,2,4-Triazine",
    "c1nccnc1": "Pyrazine",
    "c1cnccn1": "Pyrimidine",
    "C1CCNCC1": "Piperidine",
    "C1CCNC1": "Pyrrolidine",
    "C1CCOCC1": "Tetrahydropyran",
    "C1COCCC1": "Tetrahydropyran",
    "C1CNCCN1": "Piperazine",
    "C1CCNCCN1": "Piperazine",  # alternate
    "C1CCSCC1": "Thiane",
    "C1COCCO1": "1,4-Dioxane",
    "C1CCNCC1": "Piperidine",
    "C1=CC=NC=C1": "Pyridine",  # Kekulé form
    "C1CCCCCC1": "Cycloheptane",

    # ── Fused bicyclics (6+5) ────────────────────────────────────────
    "c1ccc2[nH]ccc2c1": "Indole",
    "c1ccc2c(c1)cc[nH]2": "Indole",
    "c1ccc2occc2c1": "Benzofuran",
    "c1ccc2c(c1)cco2": "Benzofuran",
    "c1ccc2sccc2c1": "Benzothiophene",
    "c1ccc2c(c1)ccs2": "Benzothiophene",
    "c1ccc2[nH]ncc2c1": "Indazole",
    "c1ccc2c(c1)cn[nH]2": "Indazole",
    "c1ccc2[nH]cnc2c1": "Benzimidazole",
    "c1ccc2c(c1)nc[nH]2": "Benzimidazole",
    "c1ccc2ocnc2c1": "Benzoxazole",
    "c1ccc2c(c1)nco2": "Benzoxazole",
    "c1ccc2scnc2c1": "Benzothiazole",
    "c1ccc2c(c1)ncs2": "Benzothiazole",
    "c1cnc2ccccc2n1": "Quinazoline",
    "c1ncc2ccccc2n1": "Quinazoline",
    "c1cc2ccccc2o1": "Benzofuran",
    "c1cc2ccccc2s1": "Benzothiophene",
    "c1cc2ccccc2[nH]1": "Indole",

    # ── Fused bicyclics (6+6) ────────────────────────────────────────
    "c1ccc2ccccc2c1": "Naphthalene",
    "c1ccc2ncccc2c1": "Quinoline",
    "c1ccnc2ccccc2c1": "Quinoline",
    "c1cnc2ccccc2c1": "Quinoline",
    "c1ccc2nccnc2c1": "Quinoxaline",
    "c1cnc2ccccc2n1": "Quinazoline",
    "c1ccc2[nH]c(=O)ccc2c1": "2-Quinolinone",
    "c1ccc2c(c1)cccc2": "Naphthalene",  # alternate
    "C1CCC2CCCCC2C1": "Decalin",
    "C1CC2CCCCC2CC1": "Decalin",

    # ── Fused tricyclics ─────────────────────────────────────────────
    "c1ccc2c(c1)ccc1ccccc12": "Anthracene",
    "c1ccc2cc3ccccc3cc2c1": "Anthracene",
    "c1ccc2c(c1)cc1ccccc1c2": "Phenanthrene",
    "c1ccc2c(c1)-c1ccccc1-2": "Fluorene",
    "c1ccc2c(c1)c1ccccc1n2": "Acridine",
    "c1ccc2c(c1)c1ccncc1n2": "1,8-Naphthyridine",
    "c1ccc2[nH]c3ccccc3c2c1": "Carbazole",

    # ── Common drug scaffolds ────────────────────────────────────────
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1": "Xanthine",
    "c1nc2ccccc2[nH]1": "Benzimidazole",
    "c1nsc2ccccc21": "Benzothiazole",
    "c1occ2ccccc21": "Benzofuran",
    "O=c1cc[nH]c(=O)[nH]1": "Uracil",
    "O=c1ccn([nH]1)": "Pyridazinone",
    "c1cn2ccnc2cn1": "Purine",
    "c1ncc2[nH]cnc2n1": "Purine",
    "c1nc2c([nH]1)ncn2": "Purine",

    # ── Saturated heterocycles ───────────────────────────────────────
    "C1CCC2(CC1)CCCC2": "Spirodecane",
    "C1CC2CCC1CC2": "Norbornane",
    "C1CC1": "Cyclopropane",
    "C1CCC1": "Cyclobutane",
    "C1CCCC1": "Cyclopentane",
    "C1CC2CCCC(C1)C2": "Adamantane-partial",
    "C1CCCCCC1": "Cycloheptane",
    "C12CC3CC(CC(C3)C1)C2": "Adamantane",

    # ── 7-membered rings ─────────────────────────────────────────────
    "c1ccc2c(c1)CCC=N2": "Benzodiazepine-partial",
    "C1CCNCCC1": "Azepane",
    "C1CCOCCC1": "Oxepane",
}

# Build a canonical lookup: re-canonicalize all keys for consistent matching
_SCAFFOLD_NAME_LOOKUP = {}
for smi, name in _NAMED_SCAFFOLDS.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        can = Chem.MolToSmiles(mol)
        _SCAFFOLD_NAME_LOOKUP[can] = name


def _name_scaffold(scaffold_smiles):
    """Look up the chemical name for a scaffold SMILES.

    For multi-ring scaffolds not in the dictionary, identifies individual
    named ring systems within the scaffold.
    """
    if not scaffold_smiles:
        return "No ring system"

    # Direct lookup
    if scaffold_smiles in _SCAFFOLD_NAME_LOOKUP:
        return _SCAFFOLD_NAME_LOOKUP[scaffold_smiles]

    # Try to identify sub-rings within a fused/complex scaffold
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return "Unknown scaffold"

    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return "Acyclic scaffold"

    # Decompose: find individual ring SMILES and name them
    ring_names = []
    seen = set()
    for ring_atoms in ring_info.AtomRings():
        # Extract ring as SMILES
        ring_mol = Chem.RWMol(mol)
        atoms_to_remove = [a.GetIdx() for a in ring_mol.GetAtoms() if a.GetIdx() not in ring_atoms]
        for idx in sorted(atoms_to_remove, reverse=True):
            ring_mol.RemoveAtom(idx)
        try:
            ring_smi = Chem.MolToSmiles(ring_mol)
            if ring_smi not in seen:
                seen.add(ring_smi)
                if ring_smi in _SCAFFOLD_NAME_LOOKUP:
                    ring_names.append(_SCAFFOLD_NAME_LOOKUP[ring_smi])
        except Exception:
            continue

    if ring_names:
        return " + ".join(sorted(set(ring_names)))

    n_rings = ring_info.NumRings()
    return f"{n_rings}-ring system"


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
        "name": _name_scaffold(murcko),
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
