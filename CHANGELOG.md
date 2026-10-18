# Changelog

## v0.3.5 (2026-10-18)

- Visualization split into standalone [molviewer](https://pypi.org/project/molviewer/) package
- Removed `draw()`, `view3d()`, `draw_attribution()` from molfp
- Removed py3Dmol and ipyspeck dependencies (lighter install)
- Use `pip install molviewer` for 2D/3D visualization

## v0.3.4 (2026-10-18)

- Speck defaults to ball_and_stick when no preset specified
- Graceful handling of unsupported ipyspeck presets
- Suppressed RDKit kekulization warnings on import

## v0.3.3 (2026-10-18)

- Added ball_and_stick, toon, and licorice Speck presets
- Exposed Speck parameters: atom_scale, bond_scale, ao, brightness, outline, bonds
- Speck style in view3d() for ambient-occlusion 3D rendering
- ipyspeck added as base dependency

## v0.3.2 (2026-10-17)

### New Fingerprint Types
- FCFP4: Feature Morgan (pharmacophoric atom types), collision_free supported
- Lingo: hashed SMILES 4-grams
- Shortest Path: atom-pair features along shortest paths
- Hybridization: RDKit path FP with hybridization invariants
- Signature: atom-centered subgraph descriptors (Faulon signatures)
- Total: 35 FP types (18 2D + 9 3D + 8 graph), 27 sklearn transformers

### New Features
- `draw_attribution()`: class-specific SHAP attribution maps (green=active, red=inactive)
- Two attribution styles: highlight (solid circles) and contour (gaussian map)
- CI test workflow (Python 3.9-3.12)
- MkDocs Material documentation site
- CITATION.cff for academic citations
- Colab-compatible getting started notebook
- 393 tests passing

## v0.3.1 (2026-10-13)

- Fix PyPI page: images now display correctly (absolute URLs)
- Remove comparison table from README
- Sync `__init__.py` docstring, README, and notebook with current API
- Update logo to inline wordmark (fused-ring "m", benzene "o", teal "lfp")

## v0.3.0 (2026-10-12)

### New Fingerprint Types
- 9 3D fingerprints: USR, USRCAT, Autocorr3D, RDF, MORSE, WHIM, GETAWAY, PEOE_VSA3D, SMR_VSA3D
- 8 graph fingerprints: standard, distance, line, fragment (each with base and enriched variants)
- Total: 30 fingerprint types (13 2D + 9 3D + 8 graph)

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
