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
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon import pyAvalonTools

from . import _smarts
from ._pubchem import PubChemComputer

# ── Input handling ────────────────────────────────────────────────────────────

MolInput = Union[str, Chem.rdchem.Mol]

_cache = {}
_conformer_cache = {}  # SMILES → Mol with conformer


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


def _get_graphonly_gen(n_bits=1024):
    key = f"graphonly_gen_{n_bits}"
    if key not in _cache:
        _cache[key] = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=n_bits, maxPath=7, minPath=1, useBondOrder=False,
        )
    return _cache[key]


# ── Single-molecule core functions (private) ──────────────────────────────────

def _maccs_single(smiles_or_mol):
    mol = _ensure_mol(smiles_or_mol)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(166, dtype=np.int32)
    for i in fp.GetOnBits():
        if 1 <= i <= 166:
            arr[i - 1] = 1
    return arr.tolist()

def _ecfp4_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold(AllChem.GetMorganFingerprint(mol, radius=2), n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _fcfp4_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold(AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True), n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useFeatures=True)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _lingo_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    smi = Chem.MolToSmiles(mol)
    q = 4
    arr = np.zeros(n_bits, dtype=np.int32)
    if len(smi) >= q:
        for i in range(len(smi) - q + 1):
            h = hash(smi[i:i + q]) % n_bits
            arr[h] = 1
    return arr.tolist()

def _shortest_path_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    dm = Chem.GetDistanceMatrix(mol)
    n = mol.GetNumAtoms()
    arr = np.zeros(n_bits, dtype=np.int32)
    for i in range(n):
        ai = mol.GetAtomWithIdx(i).GetAtomicNum()
        for j in range(i + 1, n):
            aj = mol.GetAtomWithIdx(j).GetAtomicNum()
            d = int(dm[i][j])
            h = hash((ai, aj, d)) % n_bits
            arr[h] = 1
    return arr.tolist()

def _hybridization_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    invariants = [int(mol.GetAtomWithIdx(i).GetHybridization()) for i in range(mol.GetNumAtoms())]
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(mol, customAtomInvariants=invariants)
    return fp.tolist()

def _signature_single(smiles_or_mol, n_bits=1024, max_radius=3):
    mol = _ensure_mol(smiles_or_mol)
    arr = np.zeros(n_bits, dtype=np.int32)
    for atom_idx in range(mol.GetNumAtoms()):
        for r in range(max_radius + 1):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            if env:
                submol = Chem.PathToSubmol(mol, env)
                sig = Chem.MolToSmiles(submol)
            else:
                sig = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            h = hash((sig, r)) % n_bits
            arr[h] = 1
    return arr.tolist()

def _topological_torsion_single(smiles_or_mol, n_bits=2048):
    mol = _ensure_mol(smiles_or_mol)
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _avalon_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _mhfp_single(smiles_or_mol, n_bits=1024, radius=3):
    mol = _ensure_mol(smiles_or_mol)
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    shingles = set()
    for bit, envs in info.items():
        for atom_idx, r in envs:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            if env:
                submol = Chem.PathToSubmol(mol, env)
                shingles.add(Chem.MolToSmiles(submol))
            else:
                shingles.add(mol.GetAtomWithIdx(atom_idx).GetSymbol())
    arr = np.zeros(n_bits, dtype=np.int32)
    for s in shingles:
        arr[hash(s) % n_bits] = 1
    return arr.tolist()

def _map4_single(smiles_or_mol, n_bits=1024, radius=2):
    mol = _ensure_mol(smiles_or_mol)
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    atom_envs = {}
    for bit, envs in info.items():
        for atom_idx, r in envs:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            if env:
                submol = Chem.PathToSubmol(mol, env)
                smi = Chem.MolToSmiles(submol)
            else:
                smi = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            if atom_idx not in atom_envs or r > atom_envs[atom_idx][1]:
                atom_envs[atom_idx] = (smi, r)
    dm = Chem.GetDistanceMatrix(mol)
    n_atoms = mol.GetNumAtoms()
    arr = np.zeros(n_bits, dtype=np.int32)
    atom_list = list(atom_envs.keys())
    for i in range(len(atom_list)):
        for j in range(i + 1, len(atom_list)):
            a1, a2 = atom_list[i], atom_list[j]
            d = int(dm[a1][a2])
            s1, s2 = atom_envs[a1][0], atom_envs[a2][0]
            pair = f"{min(s1,s2)}|{d}|{max(s1,s2)}"
            arr[hash(pair) % n_bits] = 1
    return arr.tolist()

def _map4c_single(smiles_or_mol, n_bits=1024):
    mol = _ensure_mol(smiles_or_mol)
    try:
        from mapchiral.mapchiral import encode
    except ImportError:
        raise ImportError(
            "mapchiral is required for MAP4C fingerprint. "
            "Install it with: pip install mapchiral"
        )
    hashes = encode(mol, max_radius=2, n_permutations=n_bits, mapping=False)
    arr = np.zeros(n_bits, dtype=np.int32)
    for h in hashes:
        arr[int(h) % n_bits] = 1
    return arr.tolist()

def _multihash_fold(sparse_fp, n_bits, n_hashes=2):
    """Fold a sparse fingerprint using k independent hashes (Bloom filter approach).

    Each substructure hash is mapped to k bit positions using different seeds.
    Collision probability drops from 1/n_bits to (1/n_bits)^k.
    """
    # Handle both UIntSparseIntVect (Morgan/AP) and SparseBitVect (RDK)
    if hasattr(sparse_fp, 'GetNonzeroElements'):
        raw_hashes = list(sparse_fp.GetNonzeroElements().keys())
    else:
        raw_hashes = list(sparse_fp.GetOnBits())

    arr = np.zeros(n_bits, dtype=np.int32)
    seeds = [1, 31, 997, 7919, 104729][:n_hashes]
    for h in raw_hashes:
        for seed in seeds:
            bit = ((h * seed) ^ (seed * 0x5BD1E995)) % n_bits
            arr[bit] = 1
    return arr.tolist()

def _multihash_fold_count(sparse_fp, n_bits, n_hashes=2):
    """Multi-hash fold for count fingerprints."""
    elements = sparse_fp.GetNonzeroElements()
    arr = np.zeros(n_bits, dtype=np.int32)
    seeds = [1, 31, 997, 7919, 104729][:n_hashes]
    for h, count in elements.items():
        for seed in seeds:
            bit = ((h * seed) ^ (seed * 0x5BD1E995)) % n_bits
            arr[bit] += count
    return arr.tolist()

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

def _atompairs2d_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold(rdMolDescriptors.GetAtomPairFingerprint(mol), n_bits)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs2d_count_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold_count(rdMolDescriptors.GetAtomPairFingerprint(mol), n_bits)
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


def _ensure_conformer_impl(mol):
    """Core conformer generation logic (called by _ensure_conformer with caching)."""
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


def _ensure_conformer(mol):
    """Add optimized 3D conformer with caching by canonical SMILES."""
    # Check cache
    try:
        smi = Chem.MolToSmiles(mol)
        if smi in _conformer_cache:
            return _conformer_cache[smi]
    except Exception:
        smi = None

    result = _ensure_conformer_impl(mol)

    # Cache result
    if smi and result.GetNumConformers() > 0 and len(_conformer_cache) < 10000:
        _conformer_cache[smi] = result
    return result


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


def _atompairs3d_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    if collision_free:
        return _multihash_fold(rdMolDescriptors.GetAtomPairFingerprint(mol, use2D=False), n_bits)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol, nBits=n_bits, use2D=False
    )
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs3d_count_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    if collision_free:
        return _multihash_fold_count(rdMolDescriptors.GetAtomPairFingerprint(mol, use2D=False), n_bits)
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

def _rdk_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2**31-1, maxPath=7, minPath=1)
        sparse = gen.GetSparseFingerprint(mol)
        return _multihash_fold(sparse, n_bits)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _rdk_ext_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2**31-1, maxPath=7, minPath=1, useBondOrder=True)
        sparse = gen.GetSparseFingerprint(mol)
        return _multihash_fold(sparse, n_bits)
    fp = RDKFingerprint(mol, fpSize=n_bits, maxPath=7, minPath=1, branchedPaths=True)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _rdk_graphonly_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2**31-1, maxPath=7, minPath=1, useBondOrder=False)
        sparse = gen.GetSparseFingerprint(mol)
        return _multihash_fold(sparse, n_bits)
    fp = _get_graphonly_gen(n_bits).GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()


# ── 13 Versatile Fingerprint Functions ────────────────────────────────────────
#
# Each accepts: str, Mol, list[str], pd.Series, or pd.DataFrame
#   Single molecule → list[int]
#   Multiple molecules → numpy ndarray (or sparse csr_matrix)

def maccs(input_data, *, column=None, n_jobs=1, sparse=False):
    """MACCS Keys fingerprint (166 bits)."""
    return _versatile(_maccs_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def ecfp4(input_data, *, n_bits=1024, collision_free=False, column=None, n_jobs=1, sparse=False):
    """ECFP4 / Morgan radius=2 fingerprint. Use collision_free=True for multi-hash version."""
    return _versatile(_ecfp4_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def fcfp4(input_data, *, n_bits=1024, collision_free=False, column=None, n_jobs=1, sparse=False):
    """FCFP4 / Feature Morgan radius=2 fingerprint. Uses pharmacophoric atom types instead of atomic invariants."""
    return _versatile(_fcfp4_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def ecfp4_multihash(input_data, *, n_bits=1024, n_hashes=2, column=None, n_jobs=1, sparse=False):
    """Multi-hash ECFP4 (deprecated — use ecfp4(collision_free=True) instead)."""
    return _versatile(_ecfp4_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=True)

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

def atompairs2d(input_data, *, n_bits=780, collision_free=False, column=None, n_jobs=1, sparse=False):
    """AtomPairs2D fingerprint (780 bits, hashed). Use collision_free=True for multi-hash."""
    return _versatile(_atompairs2d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def atompairs2d_count(input_data, *, n_bits=780, collision_free=False, column=None, n_jobs=1, sparse=False):
    """AtomPairs2D Count (780 features, hashed counts). Use collision_free=True for multi-hash."""
    return _versatile(_atompairs2d_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def rdk(input_data, *, n_bits=1024, collision_free=False, column=None, n_jobs=1, sparse=False):
    """RDKit path-based fingerprint (1024 bits). Use collision_free=True for multi-hash."""
    return _versatile(_rdk_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def rdk_ext(input_data, *, n_bits=1024, collision_free=False, column=None, n_jobs=1, sparse=False):
    """RDKit Extended fingerprint (1024 bits). Use collision_free=True for multi-hash."""
    return _versatile(_rdk_ext_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def rdk_graphonly(input_data, *, n_bits=1024, collision_free=False, column=None, n_jobs=1, sparse=False):
    """RDKit GraphOnly fingerprint (1024 bits). Use collision_free=True for multi-hash."""
    return _versatile(_rdk_graphonly_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def lingo(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """Lingo fingerprint (hashed SMILES 4-grams)."""
    return _versatile(_lingo_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def shortest_path(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """Shortest Path fingerprint — encodes atom-pair features along shortest paths."""
    return _versatile(_shortest_path_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def hybridization(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """Hybridization fingerprint — path-based with atom hybridization invariants."""
    return _versatile(_hybridization_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def signature(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """Signature fingerprint — atom-centered subgraph descriptors (Faulon signatures)."""
    return _versatile(_signature_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def topological_torsion(input_data, *, n_bits=2048, column=None, n_jobs=1, sparse=False):
    """Topological Torsion fingerprint (2048 bits) — encodes sequences of 4 bonded atoms."""
    return _versatile(_topological_torsion_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def avalon(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """Avalon fingerprint (1024 bits) — path-based count fingerprint."""
    return _versatile(_avalon_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def mhfp(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """MHFP (MinHashed Fingerprint) — hashed circular substructure shingles."""
    return _versatile(_mhfp_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def map4(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """MAP4 (MinHashed Atom Pair) — hashed atom-pair substructures at radius 2."""
    return _versatile(_map4_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def map4c(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """MAP4C (MinHashed Atom Pair Chiral) — stereochemistry-aware MAP4. Requires: pip install mapchiral"""
    return _versatile(_map4c_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def atompairs3d(input_data, *, n_bits=780, collision_free=False, column=None, n_jobs=1, sparse=False):
    """3D Atom Pairs fingerprint (780 bits). Use collision_free=True for multi-hash."""
    return _versatile(_atompairs3d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

def atompairs3d_count(input_data, *, n_bits=780, collision_free=False, column=None, n_jobs=1, sparse=False):
    """3D Atom Pairs Count (780 features). Use collision_free=True for multi-hash."""
    return _versatile(_atompairs3d_count_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits, collision_free=collision_free)

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
    "fcfp4": (_fcfp4_single, 1024, "FCFP4 (Feature Morgan r=2)"),
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
    "lingo": (_lingo_single, 1024, "Lingo (SMILES 4-grams)"),
    "shortest_path": (_shortest_path_single, 1024, "Shortest Path"),
    "hybridization": (_hybridization_single, 1024, "Hybridization"),
    "signature": (_signature_single, 1024, "Signature (atom-centered subgraphs)"),
    "topological_torsion": (_topological_torsion_single, 2048, "Topological Torsion"),
    "avalon": (_avalon_single, 1024, "Avalon"),
    "mhfp": (_mhfp_single, 1024, "MHFP (MinHashed Fingerprint)"),
    "map4": (_map4_single, 1024, "MAP4 (MinHashed Atom Pair r=2)"),
    "map4c": (_map4c_single, 1024, "MAP4C (MinHashed Atom Pair Chiral)"),
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

# Graph fingerprints are registered after graph module loads (see __init__.py)
# to avoid circular imports. They are added via _register_graph_fingerprints().

_GRAPH_FP_REGISTRY = {}


def _register_graph_fingerprints():
    """Register graph vector fingerprints. Called from __init__.py after graph module loads."""
    from .graph import graph_to_vector, _mol_graph_single, _distance_graph_single, _line_graph_single, _fragment_graph_single

    def _graph_standard_single(smiles_or_mol):
        g = _mol_graph_single(smiles_or_mol)
        return graph_to_vector(g).tolist()

    def _graph_distance_single(smiles_or_mol):
        g = _distance_graph_single(smiles_or_mol)
        return graph_to_vector(g).tolist()

    def _graph_line_single(smiles_or_mol):
        g = _line_graph_single(smiles_or_mol)
        return graph_to_vector(g).tolist()

    def _graph_fragment_single(smiles_or_mol):
        g = _fragment_graph_single(smiles_or_mol)
        return graph_to_vector(g).tolist()

    graph_fps = {
        "graph_standard": (_graph_standard_single, 298, "Graph Standard (atoms=nodes, bonds=edges)"),
        "graph_distance": (_graph_distance_single, 286, "Graph Distance (complete, 3D)"),
        "graph_line": (_graph_line_single, 714, "Graph Line (bonds=nodes)"),
        "graph_fragment": (_graph_fragment_single, 510, "Graph Fragment (functional groups=nodes)"),
    }

    # Enriched versions: Morgan hashes + functional groups + neighbor counts per atom
    def _graph_standard_enriched_single(smiles_or_mol):
        g = _mol_graph_single(smiles_or_mol, enriched=True)
        return graph_to_vector(g).tolist()

    def _graph_distance_enriched_single(smiles_or_mol):
        g = _distance_graph_single(smiles_or_mol, enriched=True)
        return graph_to_vector(g).tolist()

    def _graph_line_enriched_single(smiles_or_mol):
        g = _line_graph_single(smiles_or_mol, enriched=True)
        return graph_to_vector(g).tolist()

    def _graph_fragment_enriched_single(smiles_or_mol):
        g = _fragment_graph_single(smiles_or_mol, enriched=True)
        return graph_to_vector(g).tolist()

    graph_fps.update({
        "graph_standard_enriched": (_graph_standard_enriched_single, 5242, "Graph Standard Enriched (Morgan + FG + neighbors)"),
        "graph_distance_enriched": (_graph_distance_enriched_single, 5230, "Graph Distance Enriched (Morgan + FG + neighbors)"),
        "graph_line_enriched": (_graph_line_enriched_single, 12250, "Graph Line Enriched (Morgan + FG + neighbors)"),
        "graph_fragment_enriched": (_graph_fragment_enriched_single, 5454, "Graph Fragment Enriched (Morgan + FG + neighbors)"),
    })

    FINGERPRINTS.update(graph_fps)
    _GRAPH_FP_REGISTRY.update(graph_fps)


def available() -> list[str]:
    """Return list of available fingerprint names."""
    return list(FINGERPRINTS.keys())


def get(name: str) -> tuple:
    """Get fingerprint function by name. Returns (fn, n_bits, description)."""
    if name not in FINGERPRINTS:
        raise KeyError(f"Unknown fingerprint: {name}. Available: {available()}")
    return FINGERPRINTS[name]
