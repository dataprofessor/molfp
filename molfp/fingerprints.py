"""
All 13 molecular fingerprint functions.

Each function accepts:
  - A single SMILES string or RDKit Mol → returns list[int]
  - A list of SMILES/Mols → returns numpy ndarray (n_samples, n_bits)
  - A pandas Series → returns numpy ndarray (n_samples, n_bits)
  - A pandas DataFrame + column name → returns numpy ndarray (n_samples, n_bits)
"""

from __future__ import annotations

from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

from . import _smarts
from ._pubchem import PubChemComputer

# ── Input handling ────────────────────────────────────────────────────────────

MolInput = Union[str, Chem.rdchem.Mol]

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


def _extract_smiles(input_data, column=None):
    """Extract a list of SMILES from various input types.

    Returns (smiles_list, is_batch) where is_batch=False means single molecule.
    """
    # Single string
    if isinstance(input_data, str):
        return input_data, False

    # RDKit Mol
    if isinstance(input_data, Chem.rdchem.Mol):
        return input_data, False

    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(input_data, pd.DataFrame):
            if column is not None:
                return input_data[column].tolist(), True
            # Auto-detect SMILES column
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles", "CANONICAL_SMILES",
                        "mol", "Mol", "MOL", "molecule", "smi"]:
                if col in input_data.columns:
                    return input_data[col].tolist(), True
            raise ValueError(
                "DataFrame has no recognized SMILES column. "
                "Pass column= or use a column named 'SMILES', 'smiles', or 'canonical_smiles'."
            )
        if isinstance(input_data, pd.Series):
            return input_data.tolist(), True
    except ImportError:
        pass

    # List / tuple / iterable
    if hasattr(input_data, '__iter__') and not isinstance(input_data, (str, bytes)):
        return list(input_data), True

    return input_data, False


def _versatile(single_fn, input_data, column=None, n_jobs=1, sparse=False, **kwargs):
    """Dispatch a single-molecule FP function to handle any input type.

    Returns:
        list[int] for single molecule
        numpy ndarray or scipy.sparse.csr_matrix for batch
    """
    data, is_batch = _extract_smiles(input_data, column=column)

    if not is_batch:
        return single_fn(data, **kwargs) if kwargs else single_fn(data)

    fn = (lambda s: single_fn(s, **kwargs)) if kwargs else single_fn
    return _batch(fn, data, n_jobs=n_jobs, sparse=sparse)


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


# ── Single-molecule core functions (private) ──────────────────────────────────

def _maccs_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return [int(fp[i]) for i in range(1, 167)]

def _ecfp4_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return [int(fp[i]) for i in range(n_bits)]

def _pubchem_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    return _get_pubchem_computer().compute(mol)

def _substructure_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_binary(mol, _get_substructure_patterns())

def _substructure_count_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_count(mol, _get_substructure_patterns())

def _klekota_roth_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_binary(mol, _get_kr_patterns())

def _klekota_roth_count_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    return _smarts.match_count(mol, _get_kr_patterns())

def _estate_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    counts, _ = FingerprintMol(mol)
    return [1 if c > 0 else 0 for c in counts]

def _atompairs2d_single(smiles_or_mol, n_bits=780):
    mol = _ensure_mol(smiles_or_mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs2d_count_single(smiles_or_mol, n_bits=780):
    mol = _ensure_mol(smiles_or_mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=n_bits)
    result = [0] * n_bits
    for idx, count in fp.GetNonzeroElements().items():
        if 0 <= idx < n_bits:
            result[idx] = count
    return result


def _estimate_max_iters(mol):
    """Estimate MMFF iterations needed based on molecular descriptors.

    MLR model trained on 1,420 FDA-approved drugs (R²=0.785, MAE=43 iters):
      iters = -10 + 9.9×heavy_atoms + 6.8×rot_bonds − 12.4×rings + 38.3×fsp3

    Validated with 80/20 splits on full FDA drug set:
      Kennard-Stone: R² train=0.782, test=0.757 (hard diverse test set).
      Random: R² train=0.772, test=0.833.
      Both: 97.9% first-pass convergence.

    Key insights from standardized coefficients:
      - heavy_atoms is the dominant predictor
      - rings REDUCE iterations (rigid = fewer DOF)
      - rot_bonds and fsp3 INCREASE iterations (flexibility)

    Applied with 1.5× safety margin, clamped to [200, 2000].
    """
    n_atoms = mol.GetNumHeavyAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    raw = -10 + 9.9 * n_atoms + 6.8 * n_rot - 12.4 * n_rings + 38.3 * fsp3
    return max(200, min(2000, int(1.5 * raw)))


# Transition metals, lanthanides, actinides, alkali/alkaline earth metals
_METAL_ATOMIC_NUMS = (
    set(range(3, 5))      # Li, Be
    | set(range(11, 14))  # Na, Mg, Al
    | set(range(19, 31))  # K-Zn
    | set(range(37, 49))  # Rb-Cd
    | set(range(55, 81))  # Cs-Hg
    | set(range(87, 104))  # Fr-Lr
)

_LARGE_MOL_THRESHOLD = 80  # heavy atoms


def _classify_molecule(mol):
    """Classify molecule for conformer strategy routing.

    Returns one of: 'standard', 'metal', 'large', 'metal_large'
    """
    has_metal = any(a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in mol.GetAtoms())
    is_large = mol.GetNumHeavyAtoms() > _LARGE_MOL_THRESHOLD
    if has_metal and is_large:
        return "metal_large"
    if has_metal:
        return "metal"
    if is_large:
        return "large"
    return "standard"


def _ensure_conformer(mol):
    """Add an optimized 3D conformer if the molecule doesn't have one.

    Routes molecules to the best strategy based on their characteristics:
      standard  → ETKDG + MMFF (adaptive iterations)
      metal     → ETKDG + UFF (skip MMFF — no metal parameters)
      large     → ETKDGv3 (better macrocycle handling) + MMFF (high iterations)
      metal_large → ETKDGv3 + UFF

    Sets mol property '_conformer_status' to one of:
      'converged_mmff', 'converged_uff', 'not_converged',
      'embed_failed', 'preexisting'
    And '_mol_class' to the routing decision.
    """
    if mol.GetNumConformers() > 0:
        mol.SetProp("_conformer_status", "preexisting")
        return mol

    mol_class = _classify_molecule(mol)
    mol = Chem.AddHs(mol)

    # Embedding strategy
    if mol_class in ("large", "metal_large"):
        # Large molecules: use ETKDGv3 with better macrocycle/ring handling
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status == -1:
            # Fallback: relax distance bounds
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)
    else:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    if status == -1:
        mol.SetProp("_conformer_status", "embed_failed")
        mol.SetProp("_mol_class", mol_class)
        return mol

    estimated = _estimate_max_iters(mol)

    # Optimization strategy: skip MMFF for metals (no parameters)
    if mol_class in ("metal", "metal_large"):
        try:
            for multiplier in [1, 2, 4]:
                max_iters = min(estimated * multiplier, 4000)
                result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
                if result == 0:
                    mol.SetProp("_conformer_status", "converged_uff")
                    mol.SetProp("_mol_class", mol_class)
                    return mol
            mol.SetProp("_conformer_status", "not_converged")
        except Exception:
            mol.SetProp("_conformer_status", "not_converged")
        mol.SetProp("_mol_class", mol_class)
        return mol

    # Standard / large: try MMFF first, fall back to UFF
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is not None:
        for multiplier in [1, 2, 4]:
            max_iters = min(estimated * multiplier, 4000)
            result = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
            if result == 0:
                mol.SetProp("_conformer_status", "converged_mmff")
                mol.SetProp("_mol_class", mol_class)
                return mol
        mol.SetProp("_conformer_status", "not_converged")
        mol.SetProp("_mol_class", mol_class)
        return mol

    # MMFF failed to set up → UFF fallback
    try:
        for multiplier in [1, 2, 4]:
            max_iters = min(estimated * multiplier, 4000)
            result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            if result == 0:
                mol.SetProp("_conformer_status", "converged_uff")
                mol.SetProp("_mol_class", mol_class)
                return mol
        mol.SetProp("_conformer_status", "not_converged")
    except Exception:
        mol.SetProp("_conformer_status", "not_converged")

    mol.SetProp("_mol_class", mol_class)
    return mol


def check_conformer(smiles_or_mol):
    """Check conformer generation status for a molecule.

    Returns dict with: status, converged, force_field, mol_class, n_atoms, n_conformers.
    """
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    status = mol.GetProp("_conformer_status") if mol.HasProp("_conformer_status") else "unknown"
    mol_class = mol.GetProp("_mol_class") if mol.HasProp("_mol_class") else "unknown"
    return {
        "status": status,
        "converged": status in ("converged_mmff", "converged_uff", "preexisting"),
        "force_field": "MMFF" if "mmff" in status else ("UFF" if "uff" in status else "none"),
        "mol_class": mol_class,
        "n_atoms": mol.GetNumHeavyAtoms(),
        "n_conformers": mol.GetNumConformers(),
    }


def _atompairs3d_single(smiles_or_mol, n_bits=780):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol, nBits=n_bits, use2D=False
    )
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs3d_count_single(smiles_or_mol, n_bits=780):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprint(
        mol, nBits=n_bits, use2D=False
    )
    result = [0] * n_bits
    for idx, count in fp.GetNonzeroElements().items():
        if 0 <= idx < n_bits:
            result[idx] = count
    return result


def _usr_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.GetUSR(mol)]

def _usrcat_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.GetUSRCAT(mol)]

def _autocorr3d_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.CalcAUTOCORR3D(mol)]

def _rdf_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.CalcRDF(mol)]

def _morse_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.CalcMORSE(mol)]

def _whim_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.CalcWHIM(mol)]

def _getaway_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    return [float(x) for x in rdMolDescriptors.CalcGETAWAY(mol)]

def _rdk_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1)
    return [int(fp[i]) for i in range(n_bits)]

def _rdk_ext_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1, branchedPaths=True)
    return [int(fp[i]) for i in range(n_bits)]

def _rdk_graphonly_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    fp = _get_graphonly_gen().GetFingerprint(mol)
    return [int(fp[i]) for i in range(n_bits)]


# ── 13 Versatile Fingerprint Functions ────────────────────────────────────────
#
# Each accepts: str, Mol, list[str], pd.Series, or pd.DataFrame
#   Single molecule → list[int]
#   Multiple molecules → numpy ndarray (or sparse csr_matrix)

def maccs(input_data, *, column=None, n_jobs=1, sparse=False):
    """MACCS Keys fingerprint (166 bits)."""
    return _versatile(_maccs_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def ecfp4(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """ECFP4 / Morgan radius=2 fingerprint."""
    return _versatile(_ecfp4_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def pubchem(input_data, *, column=None, n_jobs=1, sparse=False):
    """PubChem/CACTVS fingerprint (881 bits)."""
    return _versatile(_pubchem_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def substructure(input_data, *, column=None, n_jobs=1, sparse=False):
    """SubstructureFP (307 bits) — Inte:Ligand SMARTS patterns."""
    return _versatile(_substructure_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def substructure_count(input_data, *, column=None, n_jobs=1, sparse=False):
    """Substructure Count (307 features) — match counts."""
    return _versatile(_substructure_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def klekota_roth(input_data, *, column=None, n_jobs=1, sparse=False):
    """Klekota-Roth fingerprint (4860 bits)."""
    return _versatile(_klekota_roth_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def klekota_roth_count(input_data, *, column=None, n_jobs=1, sparse=False):
    """Klekota-Roth Count (4860 features) — match counts."""
    return _versatile(_klekota_roth_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def estate(input_data, *, column=None, n_jobs=1, sparse=False):
    """E-State fingerprint (79 bits)."""
    return _versatile(_estate_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def atompairs2d(input_data, *, n_bits=780, column=None, n_jobs=1, sparse=False):
    """AtomPairs2D fingerprint (780 bits, hashed)."""
    return _versatile(_atompairs2d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def atompairs2d_count(input_data, *, n_bits=780, column=None, n_jobs=1, sparse=False):
    """AtomPairs2D Count (780 features, hashed counts)."""
    return _versatile(_atompairs2d_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def rdk(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """RDKit path-based fingerprint (1024 bits)."""
    return _versatile(_rdk_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def rdk_ext(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """RDKit Extended fingerprint (1024 bits)."""
    return _versatile(_rdk_ext_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def rdk_graphonly(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """RDKit GraphOnly fingerprint (1024 bits) — ignores bond types."""
    return _versatile(_rdk_graphonly_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def atompairs3d(input_data, *, n_bits=780, column=None, n_jobs=1, sparse=False):
    """3D Atom Pairs fingerprint (780 bits) — uses Euclidean distances from 3D conformer."""
    return _versatile(_atompairs3d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def atompairs3d_count(input_data, *, n_bits=780, column=None, n_jobs=1, sparse=False):
    """3D Atom Pairs Count (780 features) — counts from 3D conformer distances."""
    return _versatile(_atompairs3d_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def usr(input_data, *, column=None, n_jobs=1, sparse=False):
    """Ultrafast Shape Recognition (12 floats) — molecular shape descriptor."""
    return _versatile(_usr_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def usrcat(input_data, *, column=None, n_jobs=1, sparse=False):
    """USRCAT (60 floats) — USR + pharmacophoric atom categories."""
    return _versatile(_usrcat_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def autocorr3d(input_data, *, column=None, n_jobs=1, sparse=False):
    """3D Autocorrelation (80 floats) — spatial correlation of atomic properties."""
    return _versatile(_autocorr3d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def rdf(input_data, *, column=None, n_jobs=1, sparse=False):
    """Radial Distribution Function (210 floats) — atom-pair distance histogram."""
    return _versatile(_rdf_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def morse(input_data, *, column=None, n_jobs=1, sparse=False):
    """MORSE (224 floats) — molecule representation based on electron diffraction."""
    return _versatile(_morse_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def whim(input_data, *, column=None, n_jobs=1, sparse=False):
    """WHIM (114 floats) — weighted holistic invariant molecular descriptors."""
    return _versatile(_whim_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def getaway(input_data, *, column=None, n_jobs=1, sparse=False):
    """GETAWAY (273 floats) — geometry, topology, and atom-weights assembly."""
    return _versatile(_getaway_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)


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
        arr = np.array(results, dtype=np.float64)
        np.nan_to_num(arr, copy=False, nan=0.0)
        return csr_matrix(arr)

    arr = np.array(results)
    if arr.dtype.kind == 'f':
        np.nan_to_num(arr, copy=False, nan=0.0)
        return arr.astype(np.float32)
    return arr.astype(np.int32)


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


def maccs_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_maccs_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def ecfp4_batch(smiles_list, n_bits=1024, n_jobs=1, sparse=False):
    return _batch(lambda s: _ecfp4_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def pubchem_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_pubchem_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def substructure_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_substructure_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def substructure_count_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_substructure_count_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def klekota_roth_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_klekota_roth_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def klekota_roth_count_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_klekota_roth_count_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def estate_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_estate_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs2d_batch(smiles_list, n_bits=780, n_jobs=1, sparse=False):
    return _batch(lambda s: _atompairs2d_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs2d_count_batch(smiles_list, n_bits=780, n_jobs=1, sparse=False):
    return _batch(lambda s: _atompairs2d_count_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_batch(smiles_list, n_bits=1024, n_jobs=1, sparse=False):
    return _batch(lambda s: _rdk_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_ext_batch(smiles_list, n_bits=1024, n_jobs=1, sparse=False):
    return _batch(lambda s: _rdk_ext_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdk_graphonly_batch(smiles_list, n_bits=1024, n_jobs=1, sparse=False):
    return _batch(lambda s: _rdk_graphonly_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs3d_batch(smiles_list, n_bits=780, n_jobs=1, sparse=False):
    return _batch(lambda s: _atompairs3d_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def atompairs3d_count_batch(smiles_list, n_bits=780, n_jobs=1, sparse=False):
    return _batch(lambda s: _atompairs3d_count_single(s, n_bits), smiles_list, n_jobs=n_jobs, sparse=sparse)

def usr_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_usr_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def usrcat_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_usrcat_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def autocorr3d_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_autocorr3d_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def rdf_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_rdf_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def morse_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_morse_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def whim_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_whim_single, smiles_list, n_jobs=n_jobs, sparse=sparse)

def getaway_batch(smiles_list, n_jobs=1, sparse=False):
    return _batch(_getaway_single, smiles_list, n_jobs=n_jobs, sparse=sparse)


# ── Registry ──────────────────────────────────────────────────────────────────

FINGERPRINTS = {
    "maccs": (_maccs_single, 166, "MACCS Keys"),
    "ecfp4": (_ecfp4_single, 1024, "ECFP4 (Morgan r=2)"),
    "pubchem": (_pubchem_single, 881, "PubChem/CACTVS"),
    "substructure": (_substructure_single, 307, "SubstructureFP (Inte:Ligand)"),
    "substructure_count": (_substructure_count_single, 307, "Substructure Count"),
    "klekota_roth": (_klekota_roth_single, 4860, "Klekota-Roth"),
    "klekota_roth_count": (_klekota_roth_count_single, 4860, "Klekota-Roth Count"),
    "estate": (_estate_single, 79, "E-State"),
    "atompairs2d": (_atompairs2d_single, 780, "AtomPairs2D"),
    "atompairs2d_count": (_atompairs2d_count_single, 780, "AtomPairs2D Count"),
    "rdk": (_rdk_single, 1024, "RDKit path-based"),
    "rdk_ext": (_rdk_ext_single, 1024, "RDKit Extended"),
    "rdk_graphonly": (_rdk_graphonly_single, 1024, "RDKit GraphOnly"),
    "atompairs3d": (_atompairs3d_single, 780, "3D Atom Pairs"),
    "atompairs3d_count": (_atompairs3d_count_single, 780, "3D Atom Pairs Count"),
    "usr": (_usr_single, 12, "Ultrafast Shape Recognition"),
    "usrcat": (_usrcat_single, 60, "USRCAT (shape + pharmacophore)"),
    "autocorr3d": (_autocorr3d_single, 80, "3D Autocorrelation"),
    "rdf": (_rdf_single, 210, "Radial Distribution Function"),
    "morse": (_morse_single, 224, "MORSE"),
    "whim": (_whim_single, 114, "WHIM"),
    "getaway": (_getaway_single, 273, "GETAWAY"),
}


def available() -> list[str]:
    """Return list of available fingerprint names."""
    return list(FINGERPRINTS.keys())


def get(name: str) -> tuple:
    """Get fingerprint function by name. Returns (fn, n_bits, description)."""
    if name not in FINGERPRINTS:
        raise KeyError(f"Unknown fingerprint: {name}. Available: {available()}")
    return FINGERPRINTS[name]
