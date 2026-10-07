"""
All 13 molecular fingerprint functions.

Each function accepts a SMILES string or an RDKit Mol object
and returns a list of ints (0/1 for binary, counts for count variants).
"""

from __future__ import annotations

from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

from . import _smarts
from ._pubchem import PubChemComputer

# ── Lazy-loaded pattern caches ────────────────────────────────────────────────

_cache = {}


def _ensure_mol(smiles_or_mol):
    """Convert SMILES to Mol if needed."""
    if isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles_or_mol}")
        return mol
    raise TypeError(f"Expected SMILES string or RDKit Mol, got {type(smiles_or_mol)}")


def _get_substructure_patterns():
    if "substructure" not in _cache:
        raw = _smarts.load_smarts_file("smarts_substructure.txt")
        compiled, _ = _smarts.compile_smarts(raw)
        _cache["substructure"] = compiled
    return _cache["substructure"]


def _get_kr_patterns():
    if "kr" not in _cache:
        raw = _smarts.load_plain_smarts("klekota_roth_smarts.txt")
        compiled, _ = _smarts.compile_smarts(raw)
        _cache["kr"] = compiled
    return _cache["kr"]


def _get_pubchem_computer():
    if "pubchem" not in _cache:
        _cache["pubchem"] = PubChemComputer()
    return _cache["pubchem"]


def _get_graphonly_gen():
    if "graphonly_gen" not in _cache:
        _cache["graphonly_gen"] = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=1024, maxPath=7, minPath=1, useBondOrder=False,
        )
    return _cache["graphonly_gen"]


# ── 13 Fingerprint Functions ─────────────────────────────────────────────────

def maccs(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """MACCS Keys fingerprint (166 bits)."""
    mol = _ensure_mol(smiles_or_mol)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return [int(fp[i]) for i in range(1, 167)]


def ecfp4(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 1024) -> list[int]:
    """ECFP4 / Morgan radius=2 fingerprint."""
    mol = _ensure_mol(smiles_or_mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return [int(fp[i]) for i in range(n_bits)]


def pubchem(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """PubChem/CACTVS fingerprint (881 bits)."""
    mol = _ensure_mol(smiles_or_mol)
    return _get_pubchem_computer().compute(mol)


def substructure(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """SubstructureFP (307 bits) — Inte:Ligand SMARTS patterns."""
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_binary(mol, _get_substructure_patterns())


def substructure_count(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """Substructure Count (307 features) — match counts."""
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_count(mol, _get_substructure_patterns())


def klekota_roth(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """Klekota-Roth fingerprint (4860 bits)."""
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_binary(mol, _get_kr_patterns())


def klekota_roth_count(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """Klekota-Roth Count (4860 features) — match counts."""
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_count(mol, _get_kr_patterns())


def estate(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> list[int]:
    """E-State fingerprint (79 bits)."""
    mol = _ensure_mol(smiles_or_mol)
    counts, _ = FingerprintMol(mol)
    return [1 if c > 0 else 0 for c in counts]


def atompairs2d(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 780) -> list[int]:
    """AtomPairs2D fingerprint (780 bits, hashed)."""
    mol = _ensure_mol(smiles_or_mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return [int(fp[i]) for i in range(n_bits)]


def atompairs2d_count(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 780) -> list[int]:
    """AtomPairs2D Count (780 features, hashed counts)."""
    mol = _ensure_mol(smiles_or_mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=n_bits)
    result = [0] * n_bits
    for idx, count in fp.GetNonzeroElements().items():
        if 0 <= idx < n_bits:
            result[idx] = count
    return result


def rdk(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 1024) -> list[int]:
    """RDKit path-based fingerprint (1024 bits) — CDK Fingerprinter equivalent."""
    mol = _ensure_mol(smiles_or_mol)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1)
    return [int(fp[i]) for i in range(n_bits)]


def rdk_ext(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 1024) -> list[int]:
    """RDKit Extended fingerprint (1024 bits) — CDK ExtendedFingerprinter equivalent."""
    mol = _ensure_mol(smiles_or_mol)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1, branchedPaths=True)
    return [int(fp[i]) for i in range(n_bits)]


def rdk_graphonly(smiles_or_mol: Union[str, Chem.rdchem.Mol], n_bits: int = 1024) -> list[int]:
    """RDKit GraphOnly fingerprint (1024 bits) — ignores bond types."""
    mol = _ensure_mol(smiles_or_mol)
    fp = _get_graphonly_gen().GetFingerprint(mol)
    return [int(fp[i]) for i in range(n_bits)]


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _compute_one(args):
    """Worker function for parallel computation. Must be top-level for pickling."""
    fn, smiles_or_mol = args
    return fn(smiles_or_mol)


def _batch(fn, smiles_list, n_jobs=1, sparse=False):
    """Compute fingerprint for a list of SMILES/Mols.

    Args:
        fn: fingerprint function
        smiles_list: list of SMILES strings or RDKit Mol objects
        n_jobs: number of parallel workers (1=sequential, -1=all CPUs)
        sparse: if True, return scipy.sparse.csr_matrix instead of dense array

    Returns:
        numpy ndarray (dense) or scipy.sparse.csr_matrix (sparse)
    """
    if n_jobs == 1:
        results = [fn(s) for s in smiles_list]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(fn)(s) for s in smiles_list
        )

    if sparse:
        from scipy.sparse import csr_matrix
        return csr_matrix(np.array(results, dtype=np.int32))
    return np.array(results, dtype=np.int32)


def compute_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], fp_type: str = "ecfp4", n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """Compute any fingerprint type in batch mode with optional parallelism and sparse output.

    Args:
        smiles_list: list of SMILES strings or RDKit Mol objects
        fp_type: fingerprint name (from molfp.available())
        n_jobs: number of parallel workers (1=sequential, -1=all CPUs)
        sparse: if True, return scipy.sparse.csr_matrix

    Returns:
        numpy ndarray (n_samples, n_bits) or scipy.sparse.csr_matrix
    """
    fn = FINGERPRINTS[fp_type][0]
    return _batch(fn, smiles_list, n_jobs=n_jobs, sparse=sparse)


def maccs_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(maccs, smiles_list, n_jobs=n_jobs, sparse=sparse)

def ecfp4_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 1024, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: ecfp4(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def pubchem_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(pubchem, smiles_list, n_jobs=n_jobs, sparse=sparse)

def substructure_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(substructure, smiles_list, n_jobs=n_jobs, sparse=sparse)

def substructure_count_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(substructure_count, smiles_list, n_jobs=n_jobs, sparse=sparse)

def klekota_roth_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(klekota_roth, smiles_list, n_jobs=n_jobs, sparse=sparse)

def klekota_roth_count_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(klekota_roth_count, smiles_list, n_jobs=n_jobs, sparse=sparse)

def estate_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(estate, smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs2d_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 780, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: atompairs2d(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs2d_count_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 780, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: atompairs2d_count(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 1024, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: rdk(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_ext_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 1024, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: rdk_ext(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_graphonly_batch(smiles_list: list[Union[str, Chem.rdchem.Mol]], n_bits: int = 1024, n_jobs: int = 1, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    return _batch(lambda s: rdk_graphonly(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)


# ── Registry ──────────────────────────────────────────────────────────────────

FINGERPRINTS = {
    "maccs": (maccs, 166, "MACCS Keys"),
    "ecfp4": (ecfp4, 1024, "ECFP4 (Morgan r=2)"),
    "pubchem": (pubchem, 881, "PubChem/CACTVS"),
    "substructure": (substructure, 307, "SubstructureFP (Inte:Ligand)"),
    "substructure_count": (substructure_count, 307, "Substructure Count"),
    "klekota_roth": (klekota_roth, 4860, "Klekota-Roth"),
    "klekota_roth_count": (klekota_roth_count, 4860, "Klekota-Roth Count"),
    "estate": (estate, 79, "E-State"),
    "atompairs2d": (atompairs2d, 780, "AtomPairs2D"),
    "atompairs2d_count": (atompairs2d_count, 780, "AtomPairs2D Count"),
    "rdk": (rdk, 1024, "RDKit path-based"),
    "rdk_ext": (rdk_ext, 1024, "RDKit Extended"),
    "rdk_graphonly": (rdk_graphonly, 1024, "RDKit GraphOnly"),
}


def available() -> list[str]:
    """Return list of available fingerprint names."""
    return list(FINGERPRINTS.keys())


def get(name: str) -> tuple:
    """Get fingerprint function by name. Returns (fn, n_bits, description)."""
    if name not in FINGERPRINTS:
        raise KeyError(f"Unknown fingerprint: {name}. Available: {available()}")
    return FINGERPRINTS[name]
