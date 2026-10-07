"""
Bit-level explainability: map each ON bit to the atoms/bonds that triggered it.

For any molecule + fingerprint type, returns a dict mapping bit indices to
the structural features responsible for setting that bit. This is the core
novel feature of molfp — no other fingerprint library provides unified
bit-level explanations across all fingerprint types.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

from . import _smarts
from .fingerprints import _ensure_mol, _get_substructure_patterns, _get_kr_patterns, _get_pubchem_computer


def explain(smiles_or_mol: Union[str, Chem.rdchem.Mol], fp_type: str) -> dict[int, Any]:
    """Explain which atoms/bonds set each ON bit in a fingerprint.

    Args:
        smiles_or_mol: SMILES string or RDKit Mol
        fp_type: fingerprint name (e.g. "ecfp4", "maccs", "substructure")

    Returns:
        dict mapping bit_index → list of atom/bond explanations.
        Format varies by FP type:
          - ecfp4/atompairs2d: {bit: [(center_atom_idx, radius), ...]}
          - rdk/rdk_ext/rdk_graphonly: {bit: [[bond_indices], ...]}
          - substructure/klekota_roth: {bit: [atom_indices_tuple, ...]}
          - pubchem: {bit: {"section": str, "detail": str, "atoms": [...]}}
          - maccs: {bit: [atom_indices_tuple, ...]}
          - estate: {bit: [atom_idx, ...]}
    """
    mol = _ensure_mol(smiles_or_mol)
    dispatch = {
        "ecfp4": _explain_ecfp4,
        "maccs": _explain_maccs,
        "pubchem": _explain_pubchem,
        "substructure": _explain_substructure,
        "substructure_count": _explain_substructure,
        "klekota_roth": _explain_klekota_roth,
        "klekota_roth_count": _explain_klekota_roth,
        "estate": _explain_estate,
        "atompairs2d": _explain_atompairs2d,
        "atompairs2d_count": _explain_atompairs2d,
        "rdk": _explain_rdk,
        "rdk_ext": _explain_rdk_ext,
        "rdk_graphonly": _explain_rdk_graphonly,
    }
    if fp_type not in dispatch:
        raise KeyError(f"Unknown fp_type: {fp_type}. Available: {list(dispatch.keys())}")
    return dispatch[fp_type](mol)


def _explain_ecfp4(mol, n_bits=1024):
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, bitInfo=bi)
    return {bit: list(envs) for bit, envs in bi.items()}


def _explain_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    result = {}
    maccs_smarts = MACCSkeys.smartsPatts
    for bit_idx in range(1, 167):
        if fp[bit_idx]:
            if bit_idx in maccs_smarts:
                sma = maccs_smarts[bit_idx][0]
                pat = Chem.MolFromSmarts(sma)
                if pat:
                    matches = mol.GetSubstructMatches(pat)
                    result[bit_idx - 1] = [list(m) for m in matches]
                else:
                    result[bit_idx - 1] = []
            else:
                result[bit_idx - 1] = []
    return result


def _explain_substructure(mol):
    patterns = _get_substructure_patterns()
    result = {}
    for i, (name, pat) in enumerate(patterns):
        if isinstance(pat, _smarts._SaltMatcher):
            if _smarts._check_salt(mol):
                neg = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0]
                pos = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
                result[i] = [neg + pos]
        else:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                result[i] = [list(m) for m in matches]
    return result


def _explain_klekota_roth(mol):
    patterns = _get_kr_patterns()
    result = {}
    for i, (name, pat) in enumerate(patterns):
        if isinstance(pat, _smarts._SaltMatcher):
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches:
            result[i] = [list(m) for m in matches]
    return result


def _explain_estate(mol):
    counts, _ = FingerprintMol(mol)
    from rdkit.Chem.EState import AtomTypes as AT
    AT.BuildPatts()
    patterns = AT.esPatterns
    result = {}
    for i in range(min(79, len(patterns))):
        if counts[i] > 0:
            _, pat = patterns[i]
            if pat is not None:
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    result[i] = [list(m) for m in matches]
    return result


def _explain_pubchem(mol):
    computer = _get_pubchem_computer()
    computer._ensure_loaded()
    bits = computer.compute(mol)
    result = {}

    # Section 1: element counts
    elem_counts = {}
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        elem_counts.setdefault(anum, []).append(atom.GetIdx())
    h_atoms = []
    for atom in mol.GetAtoms():
        h_atoms.extend([atom.GetIdx()] * atom.GetTotalNumHs())

    for bit, anum, threshold in computer._s1:
        if bits[bit]:
            if anum == 1:
                result[bit] = {"section": "element_count", "detail": f"H >= {threshold}", "atoms": h_atoms[:threshold]}
            else:
                atom_idxs = elem_counts.get(anum, [])
                sym = Chem.GetPeriodicTable().GetElementSymbol(anum)
                result[bit] = {"section": "element_count", "detail": f"{sym} >= {threshold}", "atoms": atom_idxs}

    # Section 2: ring counts
    ri = mol.GetRingInfo()
    for bit, ring_type, ring_size, threshold in computer._s2:
        if bits[bit]:
            ring_atoms = []
            for ring in ri.AtomRings():
                if ring_size == "any" or len(ring) == ring_size:
                    ring_atoms.extend(list(ring))
            result[bit] = {"section": "ring_count", "detail": f"{ring_type} size={ring_size} >= {threshold}", "atoms": list(set(ring_atoms))}

    # Sections 3-7: SMARTS
    for bit, pat in computer._smarts:
        if bits[bit]:
            matches = mol.GetSubstructMatches(pat)
            result[bit] = {"section": "smarts", "detail": "substructure match", "atoms": [list(m) for m in matches]}

    return result


def _explain_atompairs2d(mol, n_bits=780):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    bi = ao.GetBitInfoMap()
    return {bit: list(envs) for bit, envs in bi.items()}


def _explain_rdk(mol, n_bits=1024):
    bi = {}
    RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1, bitInfo=bi)
    return {bit: [list(path) for path in paths] for bit, paths in bi.items()}


def _explain_rdk_ext(mol, n_bits=1024):
    bi = {}
    RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1, branchedPaths=True, bitInfo=bi)
    return {bit: [list(path) for path in paths] for bit, paths in bi.items()}


def _explain_rdk_graphonly(mol, n_bits=1024):
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        fpSize=n_bits, maxPath=7, minPath=1, useBondOrder=False,
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitPaths()
    gen.GetFingerprint(mol, additionalOutput=ao)
    bi = ao.GetBitPaths()
    return {bit: [list(path) for path in paths] for bit, paths in bi.items()}
