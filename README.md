<p align="center">
<img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/logo.png" alt="molfp logo" width="300">
</p>

<p align="center">
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/v/molfp" alt="PyPI"></a>
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/pyversions/molfp" alt="Python"></a>
<a href="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml"><img src="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/dataprofessor/molfp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/molfp" alt="License"></a>
<a href="https://dataprofessor.github.io/molfp/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
<a href="https://colab.research.google.com/github/dataprofessor/molfp/blob/main/examples/getting_started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

# molfp

**Interpretable molecular fingerprints with SHAP-driven chemical explanations.**

A native Python/RDKit library that computes 35 molecular fingerprint types (18 2D + 9 3D + 8 graph-based) and provides built-in tools for interpretation, collision detection, activity cliff analysis, bioactivity profiling, and SHAP-based feature explanation.

For molecule visualization (2D drawing, 3D viewing, attribution maps), see [molviewer](https://github.com/dataprofessor/molviewer): `pip install molviewer`

## Features

### 35 Fingerprint Types

#### 2D Fingerprints (18 types)

| Type | Bits | Function |
|------|------|----------|
| MACCS Keys | 166 | `molfp.maccs()` |
| ECFP4 (Morgan r=2) | 1024 | `molfp.ecfp4()` |
| PubChem/CACTVS | 881 | `molfp.pubchem()` |
| SubstructureFP | 307 | `molfp.substructure()` |
| Substructure Count | 307 | `molfp.substructure_count()` |
| Klekota-Roth | 4860 | `molfp.klekota_roth()` |
| Klekota-Roth Count | 4860 | `molfp.klekota_roth_count()` |
| E-State | 79 | `molfp.estate()` |
| AtomPairs2D | 780 | `molfp.atompairs2d()` |
| AtomPairs2D Count | 780 | `molfp.atompairs2d_count()` |
| RDKit path-based | 1024 | `molfp.rdk()` |
| RDKit Extended | 1024 | `molfp.rdk_ext()` |
| RDKit GraphOnly | 1024 | `molfp.rdk_graphonly()` |
| FCFP4 (Feature Morgan) | 1024 | `molfp.fcfp4()` |
| Lingo | 1024 | `molfp.lingo()` |
| Shortest Path | 1024 | `molfp.shortest_path()` |
| Hybridization | 1024 | `molfp.hybridization()` |
| Signature | 1024 | `molfp.signature()` |

#### 3D Fingerprints (9 types)

| Type | Dims | Function |
|------|------|----------|
| 3D Atom Pairs | 780 | `molfp.atompairs3d()` |
| 3D Atom Pairs Count | 780 | `molfp.atompairs3d_count()` |
| USR | 12 | `molfp.usr()` |
| USRCAT | 60 | `molfp.usrcat()` |
| 3D Autocorrelation | 80 | `molfp.autocorr3d()` |
| RDF | 210 | `molfp.rdf()` |
| MORSE | 224 | `molfp.morse()` |
| WHIM | 114 | `molfp.whim()` |
| GETAWAY | 273 | `molfp.getaway()` |

3D fingerprints use adaptive conformer optimization with MMFF94, where iteration counts are estimated from molecular complexity via a regression model trained on 1,420 FDA-approved drugs.

#### Graph Fingerprints (8 types)

| Type | Dims | Function |
|------|------|----------|
| Graph Standard | 298 | `molfp.compute_batch(smiles, "graph_standard")` |
| Graph Distance (3D) | 286 | `molfp.compute_batch(smiles, "graph_distance")` |
| Graph Line | 714 | `molfp.compute_batch(smiles, "graph_line")` |
| Graph Fragment | 510 | `molfp.compute_batch(smiles, "graph_fragment")` |
| Graph Standard Enriched | 5,242 | `molfp.compute_batch(smiles, "graph_standard_enriched")` |
| Graph Distance Enriched | 5,230 | `molfp.compute_batch(smiles, "graph_distance_enriched")` |
| Graph Line Enriched | 12,250 | `molfp.compute_batch(smiles, "graph_line_enriched")` |
| Graph Fragment Enriched | 5,454 | `molfp.compute_batch(smiles, "graph_fragment_enriched")` |

Enriched variants add Morgan environment hashes, functional group membership, and neighbor type counts per atom. Base variants use atom-level properties only (type, degree, charge, hybridization, aromaticity, ring).

### Versatile Input Handling

Every fingerprint function accepts any of these input types:

```python
molfp.ecfp4("c1ccccc1")                  # single SMILES → list
molfp.ecfp4(["CCO", "c1ccccc1"])          # list → numpy array
molfp.ecfp4(pd.Series(["CCO", "c1c..."]))  # Series → numpy array
molfp.ecfp4(df, column="SMILES")           # DataFrame → numpy array (auto-detects column)
```

### Bit-Level Explainability

Map each ON bit to the atoms/bonds that triggered it — across all 13 2D FP types:

```python
molfp.explain("CC(=O)Oc1ccccc1C(=O)O", "ecfp4")
# {11: [(1, 1)], 33: [(0, 0)], 64: [(6, 1), (7, 1)], ...}
# bit 64 → center atoms 6 and 7 at radius 1
```

### Chemical Annotation

Human-readable names for every fingerprint bit:

```python
molfp.annotate("substructure", 83)
# {'name': 'Carboxylic_acid', 'smarts': '[CX3...](=[OX1])...', 'category': 'functional_group'}

molfp.annotate("pubchem", 9)
# {'description': 'Element count: C >= 2', 'category': 'element_count'}
```

### SHAP Interpretation with Chemical Meaning

One call: fingerprint → model → SHAP → chemical annotation → per-class SAR:

```python
result = molfp.interpret(smiles, activities, fp_type="maccs", metric="pchembl")

# Per-class drivers with present/absent effect:
# ACTIVE:
#   ↑ present  methyl group (-CH3)     SHAP +0.004  (freq 23% vs 19%)
#   ↓ present  oxygen atom             SHAP -0.003  (freq 13% vs 17%)
```

Additional functions for molecule-level analysis:

```python
# Explain a single molecule's prediction
molfp.interpret_molecule(smiles, interpretation_result)

# Compare two molecules' structural differences by importance
molfp.compare_molecules(smiles_a, smiles_b, interpretation_result)
```

### Hash Collision Detection

Detect when different substructures hash to the same bit:

```python
molfp.detect_collisions("c1ccccc1", "ecfp4", n_bits=256)
# {63: [2360741695, 3976623167]}  → 2 hashes collided at bit 63

molfp.collision_rate(smiles_list, "ecfp4", n_bits=1024)
# {'collision_rate': 0.027, 'collided_bits': 15, ...}
```

### Activity Cliff Detection

Find structurally similar molecules with large activity differences:

```python
cliffs = molfp.activity_cliffs(smiles, activities, "maccs",
                                sim_threshold=0.8, activity_threshold=2.0)
```

### Bioactivity Class Profiling

Classify molecules, analyze intra/inter-class similarity, find enriched bits:

```python
labels = molfp.classify(pchembl_values, metric="pchembl")
result = molfp.class_similarity(smiles, labels, "maccs")
enriched = molfp.enriched_bits(smiles, labels, "maccs")
```

### Structure Standardization

Desalt, neutralize, canonicalize tautomers, deduplicate by InChIKey:

```python
molfp.standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
# → "CC(=O)Oc1ccccc1C(=O)O"  (desalted + neutralized)

result = molfp.preprocess(smiles_list)
# n_input: 1000 → n_after_standardization: 985 → n_after_deduplication: 923
```

### Distance Functions

Vectorized Tanimoto, Dice, Cosine, Sokal-Michener, Russel-Rao:

```python
molfp.tanimoto(fp_a, fp_b)           # single pair → float
molfp.pairwise(X, metric="tanimoto") # (N, N_bits) → (N, N) matrix (vectorized)
```

### scikit-learn Compatible

All 35 FP types as sklearn transformers with Pipeline, GridSearchCV, and FeatureUnion support:

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(
    molfp.ECFPTransformer(n_bits=2048, n_jobs=-1),
    RandomForestClassifier()
)
pipe.fit(smiles_train, y_train)
```

### Sparse Matrix & Parallelism

```python
molfp.klekota_roth(smiles_list, sparse=True, n_jobs=-1)
# → scipy.sparse.csr_matrix, 97% memory savings for KR 4860-bit FPs
```

### Consensus Fingerprints

Build optimal fusion fingerprints from multiple types using mutual information:

```python
molfp.fuse(smiles, activities, ["ecfp4", "maccs", "pubchem"])
```

### Feature Selection

Standalone functions that work with any fingerprint or graph feature output:

```python
# Remove constant/near-constant features (no labels needed)
result = molfp.remove_low_variance(X, threshold=0.01)
X_clean = result["X"]  # filtered matrix

# Select top-k features by mutual information (requires labels)
result = molfp.select_features(X, y, k=500, threshold=0.01)
X_selected = result["X"]       # (n_samples, 500)
result["indices"]               # original column indices
result["scores"]                # mutual information scores
```

### Molecular Graph Representations for Deep Learning

Convert molecules to atom/bond feature matrices for graph neural networks (GCN, GAT, MPNN, etc.). Framework-agnostic core with optional PyTorch Geometric and DGL converters.

```python
# Convert to graph — returns NumPy arrays (no DL framework needed)
g = molfp.mol_graph("c1ccccc1")                      # type="standard" (default)
g = molfp.mol_graph("c1ccccc1", type="distance")     # complete graph with 3D distances
g = molfp.mol_graph("c1ccccc1", type="line")          # bonds as nodes
g = molfp.mol_graph("c1ccccc1", type="fragment")      # functional groups as nodes
g = molfp.mol_graph("c1ccccc1", enriched=True)        # enriched atom features (432 dims)

# g["atom_features"]  → numpy (n_nodes, n_features)
# g["edge_index"]     → numpy (2, n_edges)
# g["edge_features"]  → numpy (n_edges, n_edge_features)

# Batch conversion
graphs = molfp.mol_graph(["CCO", "c1ccccc1", "CC(=O)O"])

# PyTorch Geometric (requires: pip install torch torch_geometric)
data = molfp.to_pyg(g, y=1)                          # single graph → Data
loader = molfp.graphs_to_pyg(graphs, y=labels, batch_size=32)  # → DataLoader

# DGL (requires: pip install dgl torch)
dgl_graph = molfp.to_dgl(g)

# Feature inspection
molfp.atom_feature_names()  # ['atom_type_C', 'atom_type_N', ..., 'is_aromatic', 'in_ring']
molfp.edge_feature_names()  # ['bond_type_single', ..., 'is_conjugated', 'in_ring', ...]
```

**Atom features (20 dims):** atom type (one-hot, 10), degree, formal charge, num Hs, hybridization (one-hot, 5), is aromatic, in ring.

**Edge features (12 dims):** bond type (one-hot, 4), is conjugated, in ring, stereo (one-hot, 6).

### Graph Vectors for scikit-learn

```python
# Fixed-length vectors from any graph type
X = molfp.graph_vectors(smiles, type="standard", enriched=True)

# sklearn pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
pipe = make_pipeline(molfp.GraphTransformer(enriched=True, type="standard"), RandomForestClassifier())
pipe.fit(smiles_train, y_train)
```

### Collision Handling

Hashed fingerprints (ECFP4, RDK, AtomPairs) fold substructures into fixed-length vectors via modulo, causing hash collisions that corrupt SHAP attributions.

```python
# Detect collisions
molfp.detect_collisions("c1ccccc1", "ecfp4", n_bits=1024)

# Remove colliding bits from feature matrix
result = molfp.remove_collisions(X, smiles, "ecfp4", 1024)

# Collision-free mode (multi-hash Bloom filter approach)
fps = molfp.ecfp4(smiles, collision_free=True)  # works on all 8 hashed FPs

# Collision-aware SHAP interpretation
result = molfp.interpret(smiles, activities, fp_type="ecfp4", collision_aware=True)
```

### Scaffold Analysis

Extract Murcko/generic scaffolds with chemical names from ~130 common ring systems. Scaffold-aware train/test splitting prevents data leakage in QSAR.

```python
molfp.scaffold("CC(=O)Oc1ccccc1C(=O)O")
# {"murcko": "c1ccccc1", "generic": "C1CCCCC1", "name": "Benzene"}

# Scaffold fingerprint — binary membership matrix
result = molfp.scaffold_fingerprint(smiles)

# Scaffold-aware split (no scaffold leakage)
result = molfp.scaffold_split(smiles, test_size=0.2)
train_idx, test_idx = result["train_idx"], result["test_idx"]
```

## License

MIT
