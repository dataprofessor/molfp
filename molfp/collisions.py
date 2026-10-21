"""
Hash collision detection for folded/hashed molecular fingerprints.

Detects when multiple distinct substructures hash to the same bit position,
which degrades ML model performance and corrupts SHAP attributions
(see: Collision-Free Morgan Fingerprints, J Cheminform 2026).
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdFingerprintGenerator

from .fingerprints import _ensure_mol, _morgan_gen, _atompair_gen


def detect_collisions(smiles_or_mol: Union[str, Chem.rdchem.Mol], fp_type: str = "ecfp4", n_bits: int = 1024) -> dict[int, list[int]]:
    """Detect bit collisions in a hashed fingerprint for a single molecule.

    Returns dict mapping bit_idx → list of distinct hash values that folded
    to that bit. Bits with len > 1 have collisions.

    Only works for hashed FPs: ecfp4, atompairs2d, rdk, rdk_ext, rdk_graphonly.
    """
    mol = _ensure_mol(smiles_or_mol)

    if fp_type == "ecfp4":
        return _collisions_morgan(mol, n_bits)
    elif fp_type == "atompairs2d":
        return _collisions_atompair(mol, n_bits)
    elif fp_type in ("rdk", "rdk_ext", "rdk_graphonly"):
        return _collisions_rdk(mol, fp_type, n_bits)
    else:
        raise ValueError(f"Collision detection only for hashed FPs. Got: {fp_type}")


def _collisions_morgan(mol, n_bits):
    # Get unfolded (sparse) fingerprint with all unique hashes
    sparse_fp = _morgan_gen(2).GetSparseCountFingerprint(mol)
    hashes = list(sparse_fp.GetNonzeroElements().keys())

    # Map each hash to its folded bit position
    bit_to_hashes = {}
    for h in hashes:
        bit = h % n_bits
        bit_to_hashes.setdefault(bit, []).append(h)

    return {b: hs for b, hs in bit_to_hashes.items() if len(hs) > 1}


def _collisions_atompair(mol, n_bits):
    sparse_fp = _atompair_gen().GetSparseCountFingerprint(mol)
    hashes = list(sparse_fp.GetNonzeroElements().keys())

    bit_to_hashes = {}
    for h in hashes:
        bit = h % n_bits
        bit_to_hashes.setdefault(bit, []).append(h)

    return {b: hs for b, hs in bit_to_hashes.items() if len(hs) > 1}


def _collisions_rdk(mol, fp_type, n_bits):
    # RDKit fingerprint doesn't expose raw hashes easily
    # We compare bitInfo entries: if a bit has paths of different lengths, it's a collision
    bi = {}
    if fp_type == "rdk_graphonly":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=n_bits, maxPath=7, minPath=1, useBondOrder=False,
        )
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitPaths()
        gen.GetFingerprint(mol, additionalOutput=ao)
        bi = ao.GetBitPaths()
    else:
        branched = fp_type == "rdk_ext"
        from rdkit.Chem import RDKFingerprint as RDKFp
        RDKFp(mol, fpSize=n_bits, maxPath=7, minPath=1, branchedPaths=branched, bitInfo=bi)

    # A bit with paths of different lengths = collision (different substructures)
    collisions = {}
    for bit, paths in bi.items():
        if len(paths) > 1:
            path_lengths = set(len(p) for p in paths)
            if len(path_lengths) > 1:
                collisions[bit] = [list(p) for p in paths]
    return collisions


def collision_rate(smiles_list: list[Union[str, Chem.rdchem.Mol]], fp_type: str = "ecfp4", n_bits: int = 1024) -> dict[str, Any]:
    """Compute collision statistics across a dataset.

    Returns dict with:
        molecules: number of molecules analyzed
        total_unique_hashes: total distinct hashes across all molecules
        total_bits_set: total bits set (after folding)
        collided_bits: bits with >1 hash mapped to them
        collision_rate: fraction of set bits that have collisions
        avg_collisions_per_mol: mean collisions per molecule
    """
    total_unique = 0
    total_set = 0
    total_collided = 0
    n_mols = 0
    per_mol_collisions = []

    for smi in smiles_list:
        try:
            mol = _ensure_mol(smi)
            collisions = detect_collisions(mol, fp_type, n_bits)
            n_collided = len(collisions)

            if fp_type == "ecfp4":
                sparse = _morgan_gen(2).GetSparseCountFingerprint(mol)
                n_unique = len(sparse.GetNonzeroElements())
                fp = _morgan_gen(2, n_bits).GetFingerprint(mol)
                n_set = fp.GetNumOnBits()
            elif fp_type == "atompairs2d":
                sparse = _atompair_gen().GetSparseCountFingerprint(mol)
                n_unique = len(sparse.GetNonzeroElements())
                fp = _atompair_gen(n_bits, count_simulation=True).GetFingerprint(mol)
                n_set = fp.GetNumOnBits()
            else:
                n_unique = 0
                n_set = 0

            total_unique += n_unique
            total_set += n_set
            total_collided += n_collided
            per_mol_collisions.append(n_collided)
            n_mols += 1
        except (ValueError, Exception):
            continue

    return {
        "molecules": n_mols,
        "total_unique_hashes": total_unique,
        "total_bits_set": total_set,
        "collided_bits": total_collided,
        "collision_rate": total_collided / total_set if total_set else 0,
        "avg_collisions_per_mol": np.mean(per_mol_collisions) if per_mol_collisions else 0,
    }


def collision_pairs(smiles_list: list[Union[str, Chem.rdchem.Mol]], fp_type: str = "ecfp4", n_bits: int = 1024) -> list[tuple[str, str, int, int]]:
    """Find pairs of distinct molecules with identical fingerprints.

    Returns list of (smiles_a, smiles_b, idx_a, idx_b) tuples.
    """
    from .fingerprints import FINGERPRINTS

    fn = FINGERPRINTS[fp_type][0]
    default_bits = FINGERPRINTS[fp_type][1]
    # Pass n_bits if the function accepts it and it differs from default
    import inspect
    sig = inspect.signature(fn)
    use_nbits = "n_bits" in sig.parameters

    fps = {}
    for i, smi in enumerate(smiles_list):
        try:
            fp = tuple(fn(smi, n_bits=n_bits)) if use_nbits else tuple(fn(smi))
            fps.setdefault(fp, []).append((i, smi))
        except (ValueError, Exception):
            continue

    pairs = []
    for fp_key, members in fps.items():
        if len(members) > 1:
            for j in range(len(members)):
                for k in range(j + 1, len(members)):
                    pairs.append((members[j][1], members[k][1], members[j][0], members[k][0]))

    return pairs


def remove_collisions(X, smiles_list, fp_type="ecfp4", n_bits=1024):
    """Remove columns (bits) that have hash collisions in the dataset.

    Scans all molecules for colliding bits, builds a collision-free mask,
    and returns the filtered feature matrix.

    Args:
        X: numpy array (n_samples, n_bits) — fingerprint matrix.
        smiles_list: list of SMILES (same order as X rows).
        fp_type: hashed fingerprint type ("ecfp4", "atompairs2d", "rdk", etc.).
        n_bits: bit size used to compute X.

    Returns:
        dict with:
            X: filtered array (n_samples, n_clean_bits)
            mask: boolean mask of collision-free bits
            n_removed: number of colliding bits removed
            n_remaining: number of clean bits kept
            colliding_bits: set of bit indices that had collisions
    """
    colliding_bits = set()
    for smi in smiles_list:
        try:
            coll = detect_collisions(smi, fp_type, n_bits)
            colliding_bits.update(coll.keys())
        except (ValueError, Exception):
            continue

    mask = np.ones(X.shape[1], dtype=bool)
    for bit in colliding_bits:
        if bit < X.shape[1]:
            mask[bit] = False

    return {
        "X": X[:, mask],
        "mask": mask,
        "n_removed": int(np.sum(~mask)),
        "n_remaining": int(np.sum(mask)),
        "colliding_bits": colliding_bits,
    }
