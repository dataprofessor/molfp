"""
All 13 molecular fingerprint functions.

Each function accepts:
  - A single SMILES string or RDKit Mol → returns list[int]
  - A list of SMILES/Mols → returns numpy ndarray (n_samples, n_bits)
  - A pandas Series → returns numpy ndarray (n_samples, n_bits)
  - A pandas DataFrame + column name → returns numpy ndarray (n_samples, n_bits)
"""

from __future__ import annotations

import hashlib
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


# rdFingerprintGenerator replacements for the deprecated AllChem/rdMolDescriptors
# fingerprint functions. Output is bit-identical; n_bits=None means unfolded (sparse).
def _morgan_gen(radius=2, n_bits=None, features=False):
    key = f"morgan_gen_{radius}_{n_bits}_{features}"
    if key not in _cache:
        kwargs = {"radius": radius}
        if n_bits is not None:
            kwargs["fpSize"] = n_bits
        if features:
            kwargs["atomInvariantsGenerator"] = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        _cache[key] = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
    return _cache[key]


def _atompair_gen(n_bits=None, use_2d=True, count_simulation=False):
    # count_simulation=True reproduces the old hashed *bit-vector* atom pairs
    key = f"atompair_gen_{n_bits}_{use_2d}_{count_simulation}"
    if key not in _cache:
        kwargs = {"use2D": use_2d, "countSimulation": count_simulation}
        if n_bits is not None:
            kwargs["fpSize"] = n_bits
        _cache[key] = rdFingerprintGenerator.GetAtomPairGenerator(**kwargs)
    return _cache[key]


def _torsion_gen(n_bits):
    key = f"torsion_gen_{n_bits}"
    if key not in _cache:
        _cache[key] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits, countSimulation=True)
    return _cache[key]


def _morgan_bit_info(mol, radius, n_bits=None):
    """Morgan bit -> ((atom_idx, radius), ...), like the old bitInfo argument."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen = _morgan_gen(radius, n_bits)
    if n_bits is None:
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
    else:
        gen.GetFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def _get_descriptor_funcs():
    """RDKit descriptor functions in molfp's pinned order; None if this RDKit lacks one."""
    if "descriptor_funcs" not in _cache:
        from rdkit.Chem import Descriptors as _Desc
        from ._rdkit_descriptors import DESCRIPTOR_NAMES
        available = dict(_Desc.descList)
        _cache["descriptor_funcs"] = [available.get(name) for name in DESCRIPTOR_NAMES]
    return _cache["descriptor_funcs"]


def _stable_hash(value):
    """Hash that is identical across Python sessions and worker processes.

    Built-in hash() is salted per process for str (PYTHONHASHSEED), so it gives
    different bits each session and inconsistent rows under n_jobs > 1.
    """
    return int.from_bytes(hashlib.blake2b(repr(value).encode(), digest_size=8).digest(), "little")


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
        return _multihash_fold(_morgan_gen(2).GetSparseCountFingerprint(mol), n_bits)
    fp = _morgan_gen(2, n_bits).GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _fcfp4_single(smiles_or_mol, n_bits=1024, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold(_morgan_gen(2, features=True).GetSparseCountFingerprint(mol), n_bits)
    fp = _morgan_gen(2, n_bits, features=True).GetFingerprint(mol)
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
            h = _stable_hash(smi[i:i + q]) % n_bits
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
            h = _stable_hash((sig, r)) % n_bits
            arr[h] = 1
    return arr.tolist()

def _topological_torsion_single(smiles_or_mol, n_bits=2048):
    mol = _ensure_mol(smiles_or_mol)
    fp = _torsion_gen(n_bits).GetFingerprint(mol)
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
    info = _morgan_bit_info(mol, radius)
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
        arr[_stable_hash(s) % n_bits] = 1
    return arr.tolist()

def _map4_single(smiles_or_mol, n_bits=1024, radius=2):
    mol = _ensure_mol(smiles_or_mol)
    info = _morgan_bit_info(mol, radius)
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
            arr[_stable_hash(pair) % n_bits] = 1
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

def _erg_single(smiles_or_mol):
    """ErG (Extended Reduced Graph) — pharmacophore-pair fingerprint with fuzzy binning."""
    mol = _ensure_mol(smiles_or_mol)
    from rdkit.Chem import rdReducedGraphs
    fp = rdReducedGraphs.GetErGFingerprint(mol)
    return fp.tolist()

def _get_pharm2d_factory():
    if "pharm2d_factory" not in _cache:
        import os
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        from rdkit.Chem.Pharm2D.SigFactory import SigFactory
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
        sig_factory = SigFactory(feat_factory, minPointCount=2, maxPointCount=3)
        sig_factory.SetBins([(0, 2), (2, 5), (5, 8)])
        sig_factory.Init()
        _cache["pharm2d_factory"] = sig_factory
    return _cache["pharm2d_factory"]

def _pharmacophore_single(smiles_or_mol):
    """2D Pharmacophore fingerprint — 2- and 3-point pharmacophore tuples."""
    mol = _ensure_mol(smiles_or_mol)
    from rdkit.Chem.Pharm2D import Generate
    factory = _get_pharm2d_factory()
    fp = Generate.Gen2DFingerprint(mol, factory)
    n = fp.GetNumBits()
    return [int(fp[i]) for i in range(n)]

def _layered_single(smiles_or_mol, n_bits=2048):
    """Layered fingerprint — multiple property layers (atom types, bond types, rings, etc.)."""
    mol = _ensure_mol(smiles_or_mol)
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _pattern_single(smiles_or_mol, n_bits=2048):
    """Pattern fingerprint — optimized for substructure screening."""
    mol = _ensure_mol(smiles_or_mol)
    fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def _e3fp_single(smiles_or_mol, n_bits=1024):
    """E3FP (Extended 3D Fingerprint) — 3D circular substructure fingerprint."""
    mol = _ensure_mol(smiles_or_mol)
    smi = Chem.MolToSmiles(mol)
    try:
        from e3fp.pipeline import fprints_from_smiles
    except ImportError:
        raise ImportError(
            "e3fp is required for E3FP fingerprint. "
            "Install it with: pip install e3fp"
        )
    import logging
    logging.getLogger("e3fp").setLevel(logging.WARNING)
    fprint_params = {"bits": n_bits, "radius_multiplier": 1.5, "rdkit_invariants": True}
    fprints = fprints_from_smiles(smi, smi, fprint_params=fprint_params, confgen_params={"num_conf": 1})
    arr = np.zeros(n_bits, dtype=np.int32)
    if fprints:
        for bit in fprints[0].indices:
            if 0 <= bit < n_bits:
                arr[bit] = 1
    return arr.tolist()

def _mqn_single(smiles_or_mol):
    """MQN (Molecular Quantum Numbers) — 42 integer descriptors."""
    mol = _ensure_mol(smiles_or_mol)
    return list(rdMolDescriptors.MQNs_(mol))

def _ghose_crippen_single(smiles_or_mol):
    """Ghose-Crippen — SlogP_VSA + SMR_VSA + PEOE_VSA descriptors (36 floats)."""
    mol = _ensure_mol(smiles_or_mol)
    slogp = list(rdMolDescriptors.SlogP_VSA_(mol))
    smr = list(rdMolDescriptors.SMR_VSA_(mol))
    peoe = list(rdMolDescriptors.PEOE_VSA_(mol))
    return slogp + smr + peoe

def _bcut2d_single(smiles_or_mol):
    """BCUT2D — Burden eigenvalues weighted by mass, charge, logP, MR (8 floats)."""
    mol = _ensure_mol(smiles_or_mol)
    try:
        return list(rdMolDescriptors.BCUT2D(mol))
    except Exception:
        return [0.0] * 8

_SHED_MAP = {
    "Donor": "D", "Acceptor": "A",
    "PosIonizable": "P", "NegIonizable": "N",
    "Hydrophobe": "H", "LumpedHydrophobe": "H",
    "Aromatic": "R",
}
_SHED_TYPES = ["D", "A", "P", "N", "H", "R", "X"]
_SHED_PAIRS = []
for _i, _t1 in enumerate(_SHED_TYPES):
    for _t2 in _SHED_TYPES[_i:]:
        _SHED_PAIRS.append((_t1, _t2))

def _get_shed_factory():
    if "shed_factory" not in _cache:
        import os
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _cache["shed_factory"] = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _cache["shed_factory"]

def _shed_single(smiles_or_mol):
    """SHED (Shannon Entropy Descriptors) — 28 pharmacophore-pair distance entropy values."""
    mol = _ensure_mol(smiles_or_mol)
    import math
    from collections import defaultdict
    factory = _get_shed_factory()
    feats = factory.GetFeaturesForMol(mol)
    atom_types = {}
    for feat in feats:
        shed_type = _SHED_MAP.get(feat.GetFamily())
        if shed_type:
            for idx in feat.GetAtomIds():
                if idx not in atom_types:
                    atom_types[idx] = set()
                atom_types[idx].add(shed_type)
    n_atoms = mol.GetNumAtoms()
    for i in range(n_atoms):
        if i not in atom_types:
            atom_types[i] = {"X"}
    dm = Chem.GetDistanceMatrix(mol)
    pair_dists = defaultdict(list)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = int(dm[i][j])
            for t1 in atom_types[i]:
                for t2 in atom_types[j]:
                    pair_dists[tuple(sorted([t1, t2]))].append(d)
    result = []
    for pair in _SHED_PAIRS:
        dists = pair_dists.get(tuple(sorted(pair)), [])
        if not dists:
            result.append(0.0)
            continue
        counts = defaultdict(int)
        for d in dists:
            counts[d] += 1
        total = len(dists)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        result.append(entropy)
    return result

def _functional_groups_single(smiles_or_mol):
    """Functional Groups — 85 RDKit fragment descriptors as count vector."""
    mol = _ensure_mol(smiles_or_mol)
    from rdkit.Chem import Fragments
    fns = sorted([f for f in dir(Fragments) if f.startswith("fr_")])
    return [getattr(Fragments, f)(mol) for f in fns]

_FG_SIZE = 85  # len(sorted([f for f in dir(Fragments) if f.startswith("fr_")]))

def _physiochemical_single(smiles_or_mol):
    """Physiochemical Properties — key RDKit 2D molecular descriptors."""
    mol = _ensure_mol(smiles_or_mol)
    result = []
    for func in _get_descriptor_funcs():
        try:
            v = func(mol) if func is not None else 0.0
            result.append(float(v) if v is not None else 0.0)
        except Exception:
            result.append(0.0)
    return result

def _cats2d_single(smiles_or_mol):
    """CATS2D (Chemically Advanced Template Search) — 2-point pharmacophore correlation vector."""
    mol = _ensure_mol(smiles_or_mol)
    from rdkit.Chem.Pharm2D import Generate
    factory = _get_cats2d_factory()
    fp = Generate.Gen2DFingerprint(mol, factory)
    n = fp.GetNumBits()
    return [int(fp[i]) for i in range(n)]

def _get_cats2d_factory():
    if "cats2d_factory" not in _cache:
        import os
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        from rdkit.Chem.Pharm2D.SigFactory import SigFactory
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
        sig_factory = SigFactory(feat_factory, minPointCount=2, maxPointCount=2)
        sig_factory.SetBins([(i, i + 1) for i in range(10)])
        sig_factory.Init()
        _cache["cats2d_factory"] = sig_factory
    return _cache["cats2d_factory"]

def _mordred_single(smiles_or_mol):
    """Mordred — 1613 2D molecular descriptors."""
    mol = _ensure_mol(smiles_or_mol)
    try:
        from mordred import Calculator, descriptors as mordred_desc
    except ImportError:
        raise ImportError(
            "mordred is required for Mordred descriptors. "
            "Install it with: pip install mordredcommunity "
            "(the original 'mordred' package is unmaintained and fails on current numpy)"
        )
    calc = _get_mordred_calc()
    result = calc(mol)
    out = []
    for v in result:
        if v is None or hasattr(v, "error"):
            out.append(0.0)
        else:
            f = float(v)
            out.append(f if np.isfinite(f) else 0.0)
    return out

def _get_mordred_calc():
    if "mordred_calc" not in _cache:
        from mordred import Calculator, descriptors as mordred_desc
        _cache["mordred_calc"] = Calculator(mordred_desc, ignore_3D=True)
    return _cache["mordred_calc"]

def _rdkit2d_single(smiles_or_mol):
    """RDKit 2D Descriptors — 210 RDKit descriptors in a fixed order (see _rdkit_descriptors)."""
    mol = _ensure_mol(smiles_or_mol)
    result = []
    for func in _get_descriptor_funcs():
        try:
            v = func(mol) if func is not None else 0.0
            f = float(v) if v is not None else 0.0
            result.append(f if np.isfinite(f) else 0.0)
        except Exception:
            result.append(0.0)
    return result

def _secfp_single(smiles_or_mol, n_bits=2048, radius=3, min_radius=1):
    """SECFP (SMILES Extended Connectivity Fingerprint).

    Like ECFP but hashes canonical SMILES of circular substructures instead of
    Morgan invariants. Includes ring membership. Default radius=3 gives SECFP6.
    """
    mol = _ensure_mol(smiles_or_mol)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()
    arr = np.zeros(n_bits, dtype=np.int32)
    ring_info = mol.GetRingInfo()
    for bit, envs in info.items():
        for atom_idx, r in envs:
            if r < min_radius:
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            if env:
                submol = Chem.PathToSubmol(mol, env)
                smi = Chem.MolToSmiles(submol)
                center_in_ring = ring_info.NumAtomRings(atom_idx)
                key = f"{smi}|r{center_in_ring}"
            else:
                key = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            arr[_stable_hash(key) % n_bits] = 1
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
        return _multihash_fold(_atompair_gen().GetSparseCountFingerprint(mol), n_bits)
    fp = _atompair_gen(n_bits, count_simulation=True).GetFingerprint(mol)
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs2d_count_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    if collision_free:
        return _multihash_fold_count(_atompair_gen().GetSparseCountFingerprint(mol), n_bits)
    fp = _atompair_gen(n_bits).GetCountFingerprint(mol)
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
        return _multihash_fold(_atompair_gen(use_2d=False).GetSparseCountFingerprint(mol), n_bits)
    fp = _atompair_gen(n_bits, use_2d=False, count_simulation=True).GetFingerprint(mol)
    return [int(fp[i]) for i in range(n_bits)]

def _atompairs3d_count_single(smiles_or_mol, n_bits=780, collision_free=False):
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    if collision_free:
        return _multihash_fold_count(_atompair_gen(use_2d=False).GetSparseCountFingerprint(mol), n_bits)
    fp = _atompair_gen(n_bits, use_2d=False).GetCountFingerprint(mol)
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

def secfp(input_data, *, n_bits=2048, column=None, n_jobs=1, sparse=False):
    """SECFP (SMILES Extended Connectivity Fingerprint) — SMILES-based circular FP with ring info."""
    return _versatile(_secfp_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def erg(input_data, *, column=None, n_jobs=1, sparse=False):
    """ErG (Extended Reduced Graph) — pharmacophore-pair fingerprint (315 floats)."""
    return _versatile(_erg_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def pharmacophore(input_data, *, column=None, n_jobs=1, sparse=False):
    """2D Pharmacophore fingerprint (2988 bits) — 2- and 3-point pharmacophore tuples with topological distances."""
    return _versatile(_pharmacophore_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def layered(input_data, *, n_bits=2048, column=None, n_jobs=1, sparse=False):
    """Layered fingerprint (2048 bits) — encodes atom types, bond types, ring membership across multiple layers."""
    return _versatile(_layered_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def pattern(input_data, *, n_bits=2048, column=None, n_jobs=1, sparse=False):
    """Pattern fingerprint (2048 bits) — optimized for substructure screening."""
    return _versatile(_pattern_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def e3fp(input_data, *, n_bits=1024, column=None, n_jobs=1, sparse=False):
    """E3FP (Extended 3D Fingerprint) — 3D circular substructures. Requires: pip install e3fp"""
    return _versatile(_e3fp_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse, n_bits=n_bits)

def mqn(input_data, *, column=None, n_jobs=1, sparse=False):
    """MQN (Molecular Quantum Numbers) — 42 integer descriptors covering atom/bond/polarity counts."""
    return _versatile(_mqn_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def ghose_crippen(input_data, *, column=None, n_jobs=1, sparse=False):
    """Ghose-Crippen (36 floats) — SlogP_VSA + SMR_VSA + PEOE_VSA lipophilicity/refractivity descriptors."""
    return _versatile(_ghose_crippen_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def bcut2d(input_data, *, column=None, n_jobs=1, sparse=False):
    """BCUT2D (8 floats) — Burden eigenvalues weighted by atomic mass, charge, logP, and molar refractivity."""
    return _versatile(_bcut2d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def shed(input_data, *, column=None, n_jobs=1, sparse=False):
    """SHED (28 floats) — Shannon Entropy Descriptors of pharmacophore-pair topological distance distributions."""
    return _versatile(_shed_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def functional_groups(input_data, *, column=None, n_jobs=1, sparse=False):
    """Functional Groups (85 counts) — RDKit fragment descriptor counts for 85 functional group types."""
    return _versatile(_functional_groups_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def physiochemical(input_data, *, column=None, n_jobs=1, sparse=False):
    """Physiochemical Properties (210 floats) — all RDKit 2D physicochemical descriptors."""
    return _versatile(_physiochemical_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def cats2d(input_data, *, column=None, n_jobs=1, sparse=False):
    """CATS2D (360 bits) — Chemically Advanced Template Search 2-point pharmacophore correlation vector."""
    return _versatile(_cats2d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def mordred(input_data, *, column=None, n_jobs=1, sparse=False):
    """Mordred (1613 floats) — comprehensive 2D molecular descriptors. Requires: pip install mordredcommunity"""
    return _versatile(_mordred_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

def rdkit2d(input_data, *, column=None, n_jobs=1, sparse=False):
    """RDKit 2D Descriptors (210 floats) — all RDKit topological descriptors."""
    return _versatile(_rdkit2d_single, input_data, column=column, n_jobs=n_jobs, sparse=sparse)

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
    "secfp": (_secfp_single, 2048, "SECFP (SMILES Extended Connectivity)"),
    "erg": (_erg_single, 315, "ErG (Extended Reduced Graph)"),
    "pharmacophore": (_pharmacophore_single, 2988, "2D Pharmacophore (2+3-point)"),
    "layered": (_layered_single, 2048, "Layered (multi-property layers)"),
    "pattern": (_pattern_single, 2048, "Pattern (substructure screening)"),
    "e3fp": (_e3fp_single, 1024, "E3FP (Extended 3D Fingerprint)"),
    "mqn": (_mqn_single, 42, "MQN (Molecular Quantum Numbers)"),
    "ghose_crippen": (_ghose_crippen_single, 36, "Ghose-Crippen (SlogP/SMR/PEOE VSA)"),
    "bcut2d": (_bcut2d_single, 8, "BCUT2D (Burden eigenvalues)"),
    "shed": (_shed_single, 28, "SHED (Shannon Entropy Descriptors)"),
    "functional_groups": (_functional_groups_single, 85, "Functional Groups (85 RDKit fragments)"),
    "physiochemical": (_physiochemical_single, 210, "Physiochemical Properties (RDKit 2D)"),
    "cats2d": (_cats2d_single, 360, "CATS2D (pharmacophore correlation)"),
    "mordred": (_mordred_single, 1613, "Mordred (2D descriptors)"),
    "rdkit2d": (_rdkit2d_single, 210, "RDKit 2D Descriptors"),
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
