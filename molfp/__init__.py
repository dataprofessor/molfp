"""
molfp — Native Python/RDKit molecular fingerprint library.

13 fingerprint types + 7 novel analysis modules:
  - explain: bit-level explainability (which atoms set which bits)
  - annotate: chemical names/descriptions for every bit
  - collisions: hash collision detection for hashed FPs
  - similarity: similarity distribution diagnostics
  - cliffs: activity cliff detection
  - fusion: consensus fingerprint construction
  - isomers: structural isomer awareness

Usage:
    import molfp

    # Compute fingerprints
    fp = molfp.maccs("CCO")
    fps = molfp.maccs_batch(["CCO", "c1ccccc1"])

    # Explain which atoms set which bits
    molfp.explain("CCO", "ecfp4")

    # Get chemical annotation for a bit
    molfp.annotate("substructure", 1)

    # Detect hash collisions
    molfp.detect_collisions("c1ccccc1", "ecfp4")

    # Find activity cliffs
    molfp.activity_cliffs(smiles, activities, "ecfp4")

    # Build optimal fusion fingerprint
    molfp.fuse(smiles, activities, ["ecfp4", "maccs", "pubchem"])

    # Check discriminative power
    molfp.discriminative_power(smiles)
"""

__version__ = "0.3.0"

# ── Core fingerprint functions ────────────────────────────────────────────────
from .fingerprints import (
    maccs,
    ecfp4,
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
from .collisions import detect_collisions, collision_rate, collision_pairs
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
from .sklearn import (
    MACCSTransformer, ECFPTransformer, PubChemTransformer,
    SubstructureTransformer, SubstructureCountTransformer,
    KlekotaRothTransformer, KlekotaRothCountTransformer,
    EStateTransformer, AtomPairs2DTransformer, AtomPairs2DCountTransformer,
    RDKTransformer, RDKExtTransformer, RDKGraphOnlyTransformer,
    AtomPairs3DTransformer, AtomPairs3DCountTransformer,
    get_transformer, TRANSFORMERS,
)
from .interpret import interpret, interpret_molecule, compare_molecules
from .distance import (
    tanimoto, dice, cosine, sokal_michener, russel_rao,
    distance, pairwise, pairwise_distance, available_metrics,
)
from .viz import draw, view3d
from .graph import (
    mol_to_graph, to_pyg, to_dgl, graphs_to_pyg,
    atom_feature_names, edge_feature_names,
    graph_to_vector, graph_vectors, GraphTransformer,
    ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, GRAPH_VECTOR_DIM,
    ENRICHED_ATOM_FEATURE_DIM,
)
