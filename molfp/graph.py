"""
Molecular graph representations for deep learning.

Converts molecules to atom/bond feature matrices and adjacency structures
suitable for graph neural networks (GCN, GAT, MPNN, SchNet, etc.).

Framework-agnostic core (NumPy only), with optional converters for
PyTorch Geometric and DGL.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, rdMolDescriptors

from .fingerprints import _ensure_mol, _extract_smiles, _ensure_conformer


# ── Feature registries ───────────────────────────────────────────────────────

# Atom types to one-hot encode (covers 99%+ of drug-like molecules)
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]

# Hybridization types
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

# Bond types
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

# Stereo types
STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
    rdchem.BondStereo.STEREOANY,
]

# Feature dimensions
ATOM_FEATURE_DIM = len(ATOM_TYPES) + 1 + 1 + 1 + len(HYBRIDIZATIONS) + 1 + 1  # 20
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 1 + len(STEREO_TYPES)  # 12


# ── Atom featurization ───────────────────────────────────────────────────────

def _one_hot(value, choices):
    """One-hot encode a value against a list of choices."""
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        if "other" in choices:
            vec[choices.index("other")] = 1
    return vec


def _atom_features(atom):
    """Compute feature vector for a single atom.

    Returns list of floats (length = ATOM_FEATURE_DIM = 21):
        - Atom type one-hot (10)
        - Degree (1)
        - Formal charge (1)
        - Num Hs (1)
        - Hybridization one-hot (5)
        - Is aromatic (1)
        - In ring (1)
    """
    symbol = atom.GetSymbol()
    atom_type = symbol if symbol in ATOM_TYPES else "other"

    features = []
    features.extend(_one_hot(atom_type, ATOM_TYPES))       # 10
    features.append(atom.GetTotalDegree())                  # 1
    features.append(atom.GetFormalCharge())                  # 1
    features.append(atom.GetTotalNumHs())                   # 1
    features.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATIONS))  # 5
    features.append(1 if atom.GetIsAromatic() else 0)       # 1
    features.append(1 if atom.IsInRing() else 0)            # 1
    return features  # total: 20


# ── Enriched atom features (substructure-aware) ──────────────────────────────

_MORGAN_HASH_BINS = 128

_FG_SMARTS = {
    "hydroxyl": "[OX2H]",
    "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "amine_primary": "[NX3;H2;!$(NC=O)]",
    "amine_secondary": "[NX3;H1;!$(NC=O)]",
    "amine_tertiary": "[NX3;H0;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=O)[OX2H0]",
    "ether": "[OD2]([#6])[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "phosphate": "[PX4](=O)([OX2])",
    "halide": "[CX4][F,Cl,Br,I]",
    "nitrile": "[CX2]#[NX1]",
    "aromatic_n": "[nR1]",
    "aromatic_o": "[oR1]",
    "aromatic_s": "[sR1]",
    "imine": "[CX3]=[NX2]",
    "thiol": "[SX2H]",
    "thioether": "[SX2]([#6])[#6]",
}

_fg_patterns = None


def _get_fg_patterns():
    global _fg_patterns
    if _fg_patterns is None:
        _fg_patterns = {}
        for name, smarts in _FG_SMARTS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                _fg_patterns[name] = pat
    return _fg_patterns


def _enriched_atom_features(mol):
    """Compute enriched per-atom features with substructure context.

    Adds to the base 20 features per atom:
        - Morgan environment hash at r=0,1,2 folded to 128-bit each (384)
        - Functional group membership (20 binary flags)
        - Neighbor type counts (8)
    Total: 20 + 384 + 20 + 8 = 432 per atom
    """
    n_atoms = mol.GetNumAtoms()

    base_feats = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Morgan environment hashes per atom at radii 0, 1, 2
    morgan_feats = np.zeros((n_atoms, 3 * _MORGAN_HASH_BINS), dtype=np.float32)
    for radius in range(3):
        bi = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=_MORGAN_HASH_BINS, bitInfo=bi)
        for bit, atom_list in bi.items():
            for center_atom, rad in atom_list:
                if rad == radius:
                    morgan_feats[center_atom, radius * _MORGAN_HASH_BINS + bit] = 1.0

    # Functional group membership per atom
    fg_patterns = _get_fg_patterns()
    fg_names = sorted(fg_patterns.keys())
    fg_feats = np.zeros((n_atoms, len(fg_names)), dtype=np.float32)
    for i, name in enumerate(fg_names):
        for match in mol.GetSubstructMatches(fg_patterns[name]):
            for atom_idx in match:
                fg_feats[atom_idx, i] = 1.0

    # Neighbor type counts per atom
    neighbor_feats = np.zeros((n_atoms, 8), dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        for nb in atom.GetNeighbors():
            sym = nb.GetSymbol()
            if sym == "C": neighbor_feats[idx, 0] += 1
            elif sym == "N": neighbor_feats[idx, 1] += 1
            elif sym == "O": neighbor_feats[idx, 2] += 1
            elif sym in ("F", "Cl", "Br", "I"): neighbor_feats[idx, 3] += 1
            if nb.GetIsAromatic(): neighbor_feats[idx, 4] += 1
            if nb.IsInRing(): neighbor_feats[idx, 5] += 1
        for bond in atom.GetBonds():
            if bond.GetBondType() == rdchem.BondType.DOUBLE: neighbor_feats[idx, 6] += 1
            elif bond.GetBondType() == rdchem.BondType.SINGLE: neighbor_feats[idx, 7] += 1

    return np.hstack([base_feats, morgan_feats, fg_feats, neighbor_feats])


ENRICHED_ATOM_FEATURE_DIM = ATOM_FEATURE_DIM + 3 * _MORGAN_HASH_BINS + len(_FG_SMARTS) + 8  # 432


# ── Edge featurization ───────────────────────────────────────────────────────

def _edge_features(bond):
    """Compute feature vector for a single bond.

    Returns list of floats (length = EDGE_FEATURE_DIM = 12):
        - Bond type one-hot (4)
        - Is conjugated (1)
        - In ring (1)
        - Stereo one-hot (6)
    """
    features = []
    features.extend(_one_hot(bond.GetBondType(), BOND_TYPES))    # 4
    features.append(1 if bond.GetIsConjugated() else 0)          # 1
    features.append(1 if bond.IsInRing() else 0)                 # 1
    features.extend(_one_hot(bond.GetStereo(), STEREO_TYPES))    # 6
    return features  # total: 12


# ── Core graph conversion ────────────────────────────────────────────────────

def _mol_graph_single(smiles_or_mol, enriched=False):
    """Convert a single molecule to a graph dict."""
    mol = _ensure_mol(smiles_or_mol)
    smiles = Chem.MolToSmiles(mol)

    # Atom features
    if enriched:
        atom_feats = _enriched_atom_features(mol)
    else:
        atom_feats = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Edge index and features (undirected: each bond stored in both directions)
    edge_src = []
    edge_dst = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _edge_features(bond)

        # Add both directions for undirected graph
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_feats.extend([feat, feat])

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_feats = np.array(edge_feats, dtype=np.float32)
    else:
        # Single atom, no bonds
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    return {
        "atom_features": atom_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "num_atoms": int(atom_feats.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "smiles": smiles,
    }


# ── Distance graph (complete graph with 3D distances) ────────────────────────

DISTANCE_EDGE_DIM = 9  # distance, 1/dist, is_bonded, bond_type(4), conjugated, in_ring

def _distance_graph_single(smiles_or_mol, enriched=False):
    """Complete graph where every atom pair is connected with 3D distance."""
    mol = _ensure_mol(smiles_or_mol)
    mol = _ensure_conformer(mol)
    smiles = Chem.MolToSmiles(mol)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Atom features (same as standard)
    if enriched:
        atom_feats = _enriched_atom_features(mol)
    else:
        atom_feats = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Build bond lookup for is_bonded + bond features
    bond_map = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _edge_features(bond)
        bond_map[(i, j)] = feat
        bond_map[(j, i)] = feat

    # All atom pairs
    edge_src, edge_dst, edge_feats = [], [], []
    for i in range(n_atoms):
        pi = conf.GetAtomPosition(i)
        for j in range(n_atoms):
            if i == j:
                continue
            pj = conf.GetAtomPosition(j)
            dist = pi.Distance(pj)
            inv_dist = 1.0 / max(dist, 1e-6)
            is_bonded = 1.0 if (i, j) in bond_map else 0.0

            if (i, j) in bond_map:
                bf = bond_map[(i, j)]
            else:
                bf = [0.0] * EDGE_FEATURE_DIM  # no bond

            # Edge features: dist, 1/dist, is_bonded, bond_type(4), conjugated, in_ring
            feat = [dist, inv_dist, is_bonded] + bf[:4] + [bf[4], bf[5]]
            edge_src.append(i)
            edge_dst.append(j)
            edge_feats.append(feat)

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_feats = np.array(edge_feats, dtype=np.float32)

    return {
        "atom_features": atom_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "num_atoms": n_atoms,
        "num_edges": int(edge_index.shape[1]),
        "smiles": smiles,
        "type": "distance",
    }


# ── Line graph (bonds as nodes) ──────────────────────────────────────────────

LINE_NODE_DIM = 2 * ATOM_FEATURE_DIM + EDGE_FEATURE_DIM  # 20+20+12 = 52
LINE_EDGE_DIM = ATOM_FEATURE_DIM  # shared atom features = 20

def _line_graph_single(smiles_or_mol, enriched=False):
    """Line graph where bonds become nodes, connected when sharing an atom."""
    mol = _ensure_mol(smiles_or_mol)
    smiles = Chem.MolToSmiles(mol)

    bonds = list(mol.GetBonds())
    n_bonds = len(bonds)

    # Determine atom feature width
    if enriched:
        enriched_feats = _enriched_atom_features(mol)
        af_dim = enriched_feats.shape[1]
    else:
        af_dim = ATOM_FEATURE_DIM

    node_dim = 2 * af_dim + EDGE_FEATURE_DIM
    edge_dim = af_dim

    if n_bonds == 0:
        return {
            "atom_features": np.zeros((0, node_dim), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_features": np.zeros((0, edge_dim), dtype=np.float32),
            "num_atoms": 0,
            "num_edges": 0,
            "smiles": smiles,
            "type": "line",
        }

    # Precompute atom features
    if enriched:
        atom_feat_map = {i: enriched_feats[i].tolist() for i in range(mol.GetNumAtoms())}
    else:
        atom_feat_map = {}
        for atom in mol.GetAtoms():
            atom_feat_map[atom.GetIdx()] = _atom_features(atom)

    # Bond-node features: [source_atom_feats, dest_atom_feats, bond_feats]
    node_feats = []
    bond_atoms = []  # (begin, end) for each bond
    for bond in bonds:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_atoms.append((i, j))
        feat = atom_feat_map[i] + atom_feat_map[j] + _edge_features(bond)
        node_feats.append(feat)
    node_feats = np.array(node_feats, dtype=np.float32)

    # Edge index: bond_a → bond_b when they share an atom
    edge_src, edge_dst, edge_feats = [], [], []
    for a in range(n_bonds):
        atoms_a = set(bond_atoms[a])
        for b in range(n_bonds):
            if a == b:
                continue
            atoms_b = set(bond_atoms[b])
            shared = atoms_a & atoms_b
            if shared:
                shared_atom = list(shared)[0]
                edge_src.append(a)
                edge_dst.append(b)
                edge_feats.append(atom_feat_map[shared_atom])

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_feats = np.array(edge_feats, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, LINE_EDGE_DIM), dtype=np.float32)

    return {
        "atom_features": node_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "num_atoms": n_bonds,
        "num_edges": int(edge_index.shape[1]),
        "smiles": smiles,
        "type": "line",
    }


# ── Fragment graph (functional groups as nodes) ──────────────────────────────

def _fragment_graph_single(smiles_or_mol, enriched=False):
    """Fragment graph where functional groups are nodes."""
    mol = _ensure_mol(smiles_or_mol)
    smiles = Chem.MolToSmiles(mol)

    fg_patterns = _get_fg_patterns()
    fg_names = sorted(fg_patterns.keys())
    n_fg_types = len(fg_names)

    # Determine atom feature width
    if enriched:
        enriched_feats = _enriched_atom_features(mol)
        af_dim = enriched_feats.shape[1]
    else:
        af_dim = ATOM_FEATURE_DIM

    node_dim = n_fg_types + 1 + af_dim  # one_hot + size + mean_atom_feats

    # Find all fragment instances
    fragments = []  # (fg_index, atom_set)
    for i, name in enumerate(fg_names):
        matches = mol.GetSubstructMatches(fg_patterns[name])
        for match in matches:
            fragments.append((i, set(match)))

    if not fragments:
        # No recognized functional groups — return single node for whole molecule
        if enriched:
            mean_feats = np.mean(enriched_feats, axis=0) if len(enriched_feats) > 0 else np.zeros(af_dim)
        else:
            atom_feats_all = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
            mean_feats = np.mean(atom_feats_all, axis=0) if len(atom_feats_all) > 0 else np.zeros(af_dim)
        node_feat = np.zeros(node_dim, dtype=np.float32)
        node_feat[n_fg_types] = mol.GetNumAtoms()
        node_feat[n_fg_types + 1:] = mean_feats
        return {
            "atom_features": node_feat.reshape(1, -1),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_features": np.zeros((0, 2), dtype=np.float32),
            "num_atoms": 1,
            "num_edges": 0,
            "smiles": smiles,
            "type": "fragment",
        }

    # Precompute atom features
    if enriched:
        atom_feat_map = {i: enriched_feats[i] for i in range(mol.GetNumAtoms())}
    else:
        atom_feat_map = {}
        for atom in mol.GetAtoms():
            atom_feat_map[atom.GetIdx()] = np.array(_atom_features(atom), dtype=np.float32)

    # Node features: [fg_type_one_hot, n_atoms, mean_atom_features]
    node_feats = []
    for fg_idx, atom_set in fragments:
        one_hot = [0.0] * n_fg_types
        one_hot[fg_idx] = 1.0
        n_atoms_frag = float(len(atom_set))
        mean_atom = np.mean([atom_feat_map[a] for a in atom_set], axis=0)
        node_feats.append(one_hot + [n_atoms_frag] + mean_atom.tolist())
    node_feats = np.array(node_feats, dtype=np.float32)

    # Edges: two fragments connected if they share atoms or are bonded
    edge_src, edge_dst, edge_feats = [], [], []
    for a in range(len(fragments)):
        atoms_a = fragments[a][1]
        for b in range(len(fragments)):
            if a == b:
                continue
            atoms_b = fragments[b][1]

            # Shared atoms
            shared = atoms_a & atoms_b
            if shared:
                edge_src.append(a)
                edge_dst.append(b)
                edge_feats.append([float(len(shared)), 0.0])
                continue

            # Check if any atom in a is bonded to any atom in b
            connected = False
            for ai in atoms_a:
                atom_obj = mol.GetAtomWithIdx(ai)
                for nb in atom_obj.GetNeighbors():
                    if nb.GetIdx() in atoms_b:
                        connected = True
                        break
                if connected:
                    break
            if connected:
                edge_src.append(a)
                edge_dst.append(b)
                edge_feats.append([0.0, 1.0])

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_feats = np.array(edge_feats, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, 2), dtype=np.float32)

    return {
        "atom_features": node_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "num_atoms": len(fragments),
        "num_edges": int(edge_index.shape[1]),
        "smiles": smiles,
        "type": "fragment",
    }


def mol_graph(input_data, *, column=None, enriched=False, type="standard"):
    """Convert molecule(s) to graph representation(s) for deep learning.

    Args:
        input_data: molecule input (str, Mol, list, Series, DataFrame).
        column: column name for DataFrame input.
        enriched: if True, add Morgan hashes, functional group membership,
            and neighbor counts per atom (432 dims vs 20 base dims).
            Only applies to type="standard" and type="distance".
        type: graph type:
            "standard" — atoms=nodes, bonds=edges (GCN, GAT, GIN)
            "distance" — atoms=nodes, all pairs=edges with 3D distance (SchNet, DimeNet)
            "line" — bonds=nodes, bond-bond connections (D-MPNN, Chemprop)
            "fragment" — functional groups=nodes (hierarchical GNNs)

    Returns:
        Single molecule: dict with atom_features, edge_index, edge_features.
        Multiple molecules: list of dicts.
    """
    _VALID_TYPES = ("standard", "distance", "line", "fragment")
    if type not in _VALID_TYPES:
        raise ValueError(f"Unknown type={type!r}. Available: {_VALID_TYPES}")

    # Select builder
    builders = {
        "standard": lambda s: _mol_graph_single(s, enriched=enriched),
        "distance": lambda s: _distance_graph_single(s, enriched=enriched),
        "line": lambda s: _line_graph_single(s, enriched=enriched),
        "fragment": lambda s: _fragment_graph_single(s, enriched=enriched),
    }
    build = builders[type]

    data, is_batch = _extract_smiles(input_data, column=column)

    if not is_batch:
        return build(data)

    return [build(s) for s in data]


# ── PyTorch Geometric converter ──────────────────────────────────────────────

def to_pyg(graph_dict, y=None):
    """Convert a graph dict to a PyTorch Geometric Data object.

    Requires: pip install torch torch_geometric

    Args:
        graph_dict: output of mol_graph() for a single molecule.
        y: optional label (int, float, or tensor).

    Returns:
        torch_geometric.data.Data
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required for to_pyg(). "
            "Install with: pip install torch torch_geometric"
        )

    data = Data(
        x=torch.from_numpy(graph_dict["atom_features"]),
        edge_index=torch.from_numpy(graph_dict["edge_index"]),
        edge_attr=torch.from_numpy(graph_dict["edge_features"]),
    )

    if y is not None:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor([y])
        data.y = y

    data.smiles = graph_dict["smiles"]
    return data


def graphs_to_pyg(graph_list, y=None, batch_size=32, shuffle=False):
    """Convert a list of graph dicts to a PyTorch Geometric DataLoader.

    Args:
        graph_list: list of graph dicts from mol_graph().
        y: optional list/array of labels (same length as graph_list).
        batch_size: batch size for DataLoader.
        shuffle: whether to shuffle the DataLoader.

    Returns:
        torch_geometric.loader.DataLoader
    """
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required for graphs_to_pyg(). "
            "Install with: pip install torch torch_geometric"
        )

    data_list = []
    for i, g in enumerate(graph_list):
        label = y[i] if y is not None else None
        data_list.append(to_pyg(g, y=label))

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


# ── DGL converter ────────────────────────────────────────────────────────────

def to_dgl(graph_dict):
    """Convert a graph dict to a DGL graph.

    Requires: pip install dgl torch

    Args:
        graph_dict: output of mol_graph() for a single molecule.

    Returns:
        dgl.DGLGraph with ndata['feat'] and edata['feat']
    """
    try:
        import torch
        import dgl
    except ImportError:
        raise ImportError(
            "DGL is required for to_dgl(). "
            "Install with: pip install dgl torch"
        )

    edge_index = graph_dict["edge_index"]
    if edge_index.shape[1] > 0:
        src = torch.from_numpy(edge_index[0])
        dst = torch.from_numpy(edge_index[1])
        g = dgl.graph((src, dst), num_nodes=graph_dict["num_atoms"])
    else:
        g = dgl.graph(([], []), num_nodes=graph_dict["num_atoms"])

    g.ndata["feat"] = torch.from_numpy(graph_dict["atom_features"])
    if graph_dict["edge_features"].shape[0] > 0:
        g.edata["feat"] = torch.from_numpy(graph_dict["edge_features"])

    return g


# ── Feature info ─────────────────────────────────────────────────────────────

def atom_feature_names():
    """Return ordered list of atom feature names."""
    names = []
    for t in ATOM_TYPES:
        names.append(f"atom_type_{t}")
    names.append("degree")
    names.append("formal_charge")
    names.append("num_hs")
    for h in ["sp", "sp2", "sp3", "sp3d", "sp3d2"]:
        names.append(f"hybridization_{h}")
    names.append("is_aromatic")
    names.append("in_ring")
    return names


def edge_feature_names():
    """Return ordered list of edge feature names."""
    names = []
    for b in ["single", "double", "triple", "aromatic"]:
        names.append(f"bond_type_{b}")
    names.append("is_conjugated")
    names.append("in_ring")
    for s in ["none", "Z", "E", "cis", "trans", "any"]:
        names.append(f"stereo_{s}")
    return names


# ── Chemistry-aware graph pooling for sklearn ────────────────────────────────

def graph_to_vector(graph_dict):
    """Convert a graph dict to a fixed-length feature vector.

    Uses chemistry-aware aggregation to preserve more structural information
    than simple mean/max pooling. Designed for sklearn classifiers.
    Adapts to both base (20-dim) and enriched (432-dim) atom features.
    """
    af = graph_dict["atom_features"]
    ef = graph_dict["edge_features"]
    n_atoms = af.shape[0]
    af_dim = af.shape[1]  # 20 (base) or 432 (enriched)

    # ── 1. Global atom aggregations (4 × af_dim) ────────────────────────
    atom_mean = np.mean(af, axis=0)
    atom_max = np.max(af, axis=0)
    atom_sum = np.sum(af, axis=0)
    atom_std = np.std(af, axis=0)

    # ── 2. Per-element aggregations (4 × af_dim) ────────────────────────
    # Columns 0-9 are atom type one-hot (same position in base and enriched)
    c_mask = af[:, 0] == 1
    n_mask = af[:, 1] == 1
    o_mask = af[:, 2] == 1
    other_mask = ~(c_mask | n_mask | o_mask)

    zero_atom = np.zeros(af_dim, dtype=np.float32)
    c_mean = np.mean(af[c_mask], axis=0) if c_mask.any() else zero_atom
    n_mean = np.mean(af[n_mask], axis=0) if n_mask.any() else zero_atom
    o_mean = np.mean(af[o_mask], axis=0) if o_mask.any() else zero_atom
    other_mean = np.mean(af[other_mask], axis=0) if other_mask.any() else zero_atom

    # ── 3. Ring vs non-ring (2 × af_dim) ────────────────────────────────
    # in_ring is always at base position 19 (last of base features)
    ring_col = ATOM_FEATURE_DIM - 1  # index 19
    ring_mask = af[:, ring_col] == 1
    nonring_mask = ~ring_mask
    ring_atom_mean = np.mean(af[ring_mask], axis=0) if ring_mask.any() else zero_atom
    nonring_atom_mean = np.mean(af[nonring_mask], axis=0) if nonring_mask.any() else zero_atom

    # ── 4. Aromatic vs non-aromatic (2 × af_dim) ────────────────────────
    arom_col = ATOM_FEATURE_DIM - 2  # index 18
    arom_mask = af[:, arom_col] == 1
    nonarom_mask = ~arom_mask
    arom_atom_mean = np.mean(af[arom_mask], axis=0) if arom_mask.any() else zero_atom
    nonarom_atom_mean = np.mean(af[nonarom_mask], axis=0) if nonarom_mask.any() else zero_atom

    # ── 5. Global edge aggregations (2 × ef_dim) ──────────────────────
    ef_dim = ef.shape[1]
    zero_edge = np.zeros(ef_dim, dtype=np.float32)
    if ef.shape[0] > 0:
        edge_mean = np.mean(ef, axis=0)
        edge_max = np.max(ef, axis=0)
    else:
        edge_mean = zero_edge
        edge_max = zero_edge

    # ── 6. Ring edge vs non-ring edge (2 × ef_dim) ──────────────────
    # Only split by ring if edge features have the ring column (standard/distance)
    edge_ring_col = 5
    if ef.shape[0] > 0 and ef.shape[1] > edge_ring_col:
        ering_mask = ef[:, edge_ring_col] == 1
        enonring_mask = ~ering_mask
        ring_edge_mean = np.mean(ef[ering_mask], axis=0) if ering_mask.any() else zero_edge
        nonring_edge_mean = np.mean(ef[enonring_mask], axis=0) if enonring_mask.any() else zero_edge
    else:
        ring_edge_mean = zero_edge
        nonring_edge_mean = zero_edge

    # ── 7. Graph topology (10) ──────────────────────────────────────────
    n_edges = graph_dict["num_edges"]
    max_edges = n_atoms * (n_atoms - 1) if n_atoms > 1 else 1
    sp3_col = 15
    has_sp3 = af_dim > sp3_col
    has_arom = af_dim > arom_col
    has_ring = af_dim > ring_col

    graph_topo = np.array([
        n_atoms,
        n_edges,
        n_edges / max(n_atoms, 1),
        n_edges / max(max_edges, 1),
        float(np.mean(af[:, arom_col])) if has_arom else 0.0,
        float(np.mean(af[:, ring_col])) if has_ring else 0.0,
        float(np.mean(af[:, sp3_col])) if has_sp3 else 0.0,
        float(1.0 - np.mean(af[:, 0])) if af_dim > 0 else 0.0,
        float(np.sum(af[:, ring_col])) if has_ring else 0.0,
        float(np.sum(ef[:, edge_ring_col])) / 2 if ef.shape[0] > 0 and ef.shape[1] > edge_ring_col else 0.0,
    ], dtype=np.float32)

    return np.concatenate([
        atom_mean, atom_max, atom_sum, atom_std,        # 80
        c_mean, n_mean, o_mean, other_mean,             # 80
        ring_atom_mean, nonring_atom_mean,              # 40
        arom_atom_mean, nonarom_atom_mean,              # 40
        edge_mean, edge_max,                            # 24
        ring_edge_mean, nonring_edge_mean,              # 24
        graph_topo,                                     # 10
    ])


GRAPH_VECTOR_DIM = 4 * ATOM_FEATURE_DIM + 4 * ATOM_FEATURE_DIM + 2 * ATOM_FEATURE_DIM + 2 * ATOM_FEATURE_DIM + 2 * EDGE_FEATURE_DIM + 2 * EDGE_FEATURE_DIM + 10  # 298


def graph_vectors(input_data, *, column=None, enriched=False, type="standard"):
    """Convert molecule(s) to fixed-length graph feature vectors for sklearn.

    Args:
        input_data: SMILES string, list, Series, or DataFrame.
        column: column name for DataFrame input.
        enriched: if True, use enriched atom features.
        type: graph type — "standard", "distance", "line", "fragment".

    Returns:
        Single molecule: numpy array
        Multiple molecules: numpy array (n_samples, n_features)
    """
    data, is_batch = _extract_smiles(input_data, column=column)

    if not is_batch:
        g = mol_graph(data, enriched=enriched, type=type)
        return graph_to_vector(g)

    graphs = mol_graph(data, enriched=enriched, type=type)
    return np.array([graph_to_vector(g) for g in graphs], dtype=np.float32)


# ── sklearn transformer ──────────────────────────────────────────────────────

class GraphTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for molecular graph features.

    Args:
        enriched: if True, use enriched atom features with Morgan hashes,
            functional group membership, and neighbor counts.
        type: graph type — "standard", "distance", "line", "fragment".

    For feature selection, use standalone functions after transforming:
        X = molfp.GraphTransformer(enriched=True).fit_transform(smiles)
        result = molfp.select_features(X, y, k=500, threshold=0.01)
    """

    def __init__(self, enriched=False, type="standard"):
        self.enriched = enriched
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return graph_vectors(X, enriched=self.enriched, type=self.type)
