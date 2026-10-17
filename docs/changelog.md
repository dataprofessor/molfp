# Changelog

## v0.3.1 (2026-10-13)

- Fix PyPI page: images now display correctly (absolute URLs)
- Remove comparison table from README
- Sync `__init__.py` docstring, README, and notebook with current API
- Update logo to inline wordmark (fused-ring "m", benzene "o", teal "lfp")

## v0.3.0 (2026-10-12)

### New Fingerprint Types
- 9 3D fingerprints: USR, USRCAT, Autocorr3D, RDF, MORSE, WHIM, GETAWAY, PEOE_VSA3D, SMR_VSA3D
- 8 graph fingerprints: standard, distance, line, fragment (each with base and enriched variants)
- Total: 35 fingerprint types (18 2D + 9 3D + 8 graph)

### New Modules
- **scaffold**: `scaffold()`, `scaffold_fingerprint()`, `scaffold_split()` — Murcko/generic scaffold extraction with ~130 named ring systems, scaffold-aware train/test splitting
- **collisions**: `remove_collisions()`, `collision_free=True` on all 8 hashed FPs (multi-hash Bloom filter), `collision_aware=True` for SHAP interpretation
- **graph**: `mol_graph()` with 4 graph types, `graph_vectors()` for fixed-length sklearn features, PyG/DGL converters
- **selection**: `select_features()` and `remove_low_variance()` — standalone feature selection
- **viz**: `draw()` with grid/highlight modes, `view3d()` interactive 3D viewer
- **standardize**: `preprocess()`, `standardize_smiles()` — desalt, neutralize, canonicalize, deduplicate

### Improvements
- Versatile input: all fingerprint functions accept str, list, Series, or DataFrame
- Enriched atom features (432 dims) for graph representations
- Adaptive conformer optimization (MMFF94, MLR-based)
- 22 sklearn transformers + GraphTransformer

## v0.2.0 (2026-04-19)

Initial release:

- 13 2D fingerprint types (MACCS, ECFP4, PubChem, Substructure, KlekotaRoth, etc.)
- `explain()` — bit-level atom attribution
- `annotate()` — chemical names for fingerprint bits
- `detect_collisions()` — hash collision detection
- `activity_cliffs()` — activity cliff detection
- `classify()`, `enriched_bits()` — bioactivity profiling
- `interpret()` — SHAP interpretation with chemical meaning
- `fuse()` — consensus fingerprint construction
- `discriminative_power()` — isomer discrimination analysis
- scikit-learn compatible transformers
- Sparse output and parallel computation support
