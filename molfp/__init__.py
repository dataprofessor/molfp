"""
molfp — Interpretable molecular fingerprints with SHAP-driven chemical explanations.

55 fingerprint types (37 2D + 10 3D + 8 graph) plus analysis modules:
  - explain/annotate: bit-level explainability and chemical annotation
  - interpret: SHAP interpretation with chemical meaning
  - collisions: hash collision detection, removal, and collision-free mode
  - similarity: pairwise metrics and similarity profiling
  - cliffs: activity cliff detection
  - bioactivity: activity classification, enriched/depleted bits
  - fusion: consensus fingerprint construction
  - isomers: structural isomer discrimination
  - graph: molecular graph representations for GNNs (4 types x 2 variants)
  - selection: feature selection (variance filter, mutual information)
  - scaffold: scaffold extraction, fingerprinting, and scaffold-aware splitting
  - viz: use molviewer package for 2D drawing, 3D viewing, and attribution maps
  - standardize: desalt, neutralize, canonicalize, deduplicate
  - sklearn: FingerprintTransformer (universal) + GraphTransformer for pipelines

Usage:
    import molfp

    # Compute fingerprints — versatile input (str, list, Series, DataFrame)
    fp = molfp.maccs("CCO")
    fps = molfp.ecfp4(["CCO", "c1ccccc1"], collision_free=True)

    # Molecular graphs for GNNs or sklearn
    g = molfp.mol_graph("c1ccccc1", type="standard", enriched=True)
    X = molfp.graph_vectors(smiles, type="standard", enriched=True)

    # Scaffold analysis
    molfp.scaffold("CC(=O)Oc1ccccc1C(=O)O")  # {"murcko": ..., "name": "Benzene"}
    train_idx, test_idx = molfp.scaffold_split(smiles, test_size=0.2)

    # Interpret with collision awareness
    result = molfp.interpret(smiles, activities, fp_type="ecfp4", collision_aware=True)
"""

__version__ = "0.3.6"

# ── Core fingerprint functions ────────────────────────────────────────────────
from .fingerprints import (
    maccs,
    ecfp4,
    ecfp4_multihash,
    fcfp4,
    pubchem,
    substructure,
    substructure_count,
    klekota_roth,
    klekota_roth_count,
    estate,
    atompairs2d,
    atompairs2d_count,
    rdk,
    rdk_ext,
    rdk_graphonly,
    lingo,
    shortest_path,
    hybridization,
    signature,
    topological_torsion,
    avalon,
    mhfp,
    map4,
    map4c,
    secfp,
    erg,
    pharmacophore,
    layered,
    pattern,
    e3fp,
    mqn,
    ghose_crippen,
    bcut2d,
    shed,
    functional_groups,
    physiochemical,
    cats2d,
    mordred,
    rdkit2d,
    atompairs3d,
    atompairs3d_count,
    usr,
    usrcat,
    autocorr3d,
    rdf,
    morse,
    whim,
    getaway,
    check_conformer,
    maccs_batch,
    ecfp4_batch,
    pubchem_batch,
    substructure_batch,
    substructure_count_batch,
    klekota_roth_batch,
    klekota_roth_count_batch,
    estate_batch,
    atompairs2d_batch,
    atompairs2d_count_batch,
    rdk_batch,
    rdk_ext_batch,
    rdk_graphonly_batch,
    atompairs3d_batch,
    atompairs3d_count_batch,
    usr_batch,
    usrcat_batch,
    autocorr3d_batch,
    rdf_batch,
    morse_batch,
    whim_batch,
    getaway_batch,
    available,
    get,
    compute_batch,
    FINGERPRINTS,
)

# ── Novel analysis modules ────────────────────────────────────────────────────
from .explain import explain
from .annotate import annotate, get_annotations
from .collisions import detect_collisions, collision_rate, collision_pairs, remove_collisions
from .similarity import similarity_matrix, similarity_profile, compare_fps
from .cliffs import activity_cliffs, cliff_summary
from .fusion import fuse, complementarity
from .isomers import find_identical_fps, discriminative_power, isomer_sensitivity
from .bioactivity import (
    classify, class_similarity, enriched_bits, depleted_bits, profile as bioactivity_profile,
)
from .standardize import (
    standardize_mol, standardize_smiles, get_inchikey,
    desalt, neutralize, canonicalize_tautomer,
    standardize_batch, deduplicate, preprocess,
    smiles_to_mol, mol_to_smiles,
)
from .sklearn import FingerprintTransformer, get_transformer, TRANSFORMERS
from .interpret import interpret, interpret_molecule, compare_molecules
from .distance import (
    tanimoto, dice, cosine, sokal_michener, russel_rao,
    distance, pairwise, pairwise_distance, available_metrics,
)
from .graph import (
    mol_graph, to_pyg, to_dgl, graphs_to_pyg,
    atom_feature_names, edge_feature_names,
    graph_to_vector, graph_vectors, GraphTransformer,
    ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, GRAPH_VECTOR_DIM,
    ENRICHED_ATOM_FEATURE_DIM,
)
from .selection import select_features, remove_low_variance
from .scaffold import scaffold, scaffold_fingerprint, scaffold_split

# Register graph fingerprints after both modules are loaded
from .fingerprints import _register_graph_fingerprints
_register_graph_fingerprints()

# Rebuild TRANSFORMERS to include graph fingerprints
from .sklearn import _rebuild_transformers
_rebuild_transformers()
