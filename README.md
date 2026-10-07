# molfp

**Interpretable molecular fingerprints with SHAP-driven chemical explanations.**

A native Python/RDKit library that computes 13 molecular fingerprint types and provides built-in tools for interpretation, collision detection, activity cliff analysis, bioactivity profiling, and SHAP-based feature explanation — capabilities not found in any other fingerprint library.

## Install

```bash
pip install molfp            # core (RDKit + scikit-learn)
pip install molfp[shap]      # + SHAP interpretation
pip install molfp[all]       # everything
```

## Quick Start

```python
import molfp

# Compute fingerprints
fp = molfp.maccs("CCO")                        # → list of 166 ints
fp = molfp.ecfp4("c1ccccc1", n_bits=2048)      # → list of 2048 ints
fps = molfp.maccs_batch(["CCO", "c1ccccc1"])    # → numpy (2, 166)

# All 13 types
molfp.available()
# ['maccs', 'ecfp4', 'pubchem', 'substructure', 'substructure_count',
#  'klekota_roth', 'klekota_roth_count', 'estate', 'atompairs2d',
#  'atompairs2d_count', 'rdk', 'rdk_ext', 'rdk_graphonly']
```

## Features

### 13 Fingerprint Types

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

### Bit-Level Explainability

Map each ON bit to the atoms/bonds that triggered it — across all 13 FP types:

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

All 13 FP types as sklearn transformers with Pipeline, GridSearchCV, and FeatureUnion support:

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
molfp.klekota_roth_batch(smiles, sparse=True, n_jobs=-1)
# → scipy.sparse.csr_matrix, 97% memory savings for KR 4860-bit FPs
```

## Comparison with scikit-fingerprints (skfp)

| Feature | molfp | skfp |
|---------|-------|------|
| Fingerprint types | 13 | 35 |
| Bit-level explainability | All 13 FPs | None |
| Chemical annotation | Built-in | Feature names only |
| SHAP + chemical meaning | `interpret()` | None |
| Hash collision detection | `detect_collisions()` | None |
| Activity cliff detection | `activity_cliffs()` | None |
| Bioactivity profiling | `classify()`, `enriched_bits()` | None |
| Consensus fingerprints | `fuse()` | None |
| Structure standardization | Full pipeline | Basic |
| sklearn compatible | Yes | Yes |
| Sparse + parallel | Yes | Yes |

## License

MIT
