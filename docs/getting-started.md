# Getting Started

## Installation

```bash
pip install molfp
```

molfp requires **Python 3.9+** and **RDKit**. RDKit is installed automatically via the `rdkit` PyPI package. If you're using conda, you may already have it:

```bash
# Check if RDKit is available
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
```

**Optional dependencies** for graph neural networks:

```bash
# PyTorch Geometric
pip install torch torch_geometric

# DGL
pip install dgl torch
```

## Quick Start: End-to-End QSAR Workflow

Here's a complete workflow — from SMILES to a trained classifier with interpretable results:

```python
import molfp
import numpy as np

# 1. Define molecules and activities
smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "OC(=O)c1ccccc1O",  # Salicylic acid
    "c1ccccc1",  # Benzene
    "CCO",  # Ethanol
    "CCCCCC",  # Hexane
]
activities = [7.2, 6.8, 6.5, 4.0, 3.5, 3.0]  # pIC50 values

# 2. Standardize SMILES
clean = molfp.preprocess(smiles)

# 3. Compute fingerprints (collision-free for clean SHAP)
fps = molfp.ecfp4(clean["smiles"], collision_free=True)

# 4. Interpret with SHAP — get chemical meaning for top features
result = molfp.interpret(
    clean["smiles"], activities,
    fp_type="ecfp4", metric="pchembl", collision_aware=True
)
print(f"Top driving bits: {result['top_bits'][:5]}")

# 5. Scaffold-aware split (no data leakage)
split = molfp.scaffold_split(clean["smiles"], test_size=0.3)
print(f"Train: {len(split['train_idx'])}, Test: {len(split['test_idx'])}")

# 6. Build sklearn pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(molfp.ECFPTransformer(), RandomForestClassifier())
# pipe.fit(smiles_train, y_train)
```

## Computing Fingerprints

All fingerprint functions accept a single SMILES string, a list of SMILES, a pandas Series, or a DataFrame column.

```python
import molfp

# Single molecule
fp = molfp.maccs("CCO")

# Batch — returns numpy array
fps = molfp.ecfp4(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])

# All 30 available types
for name in molfp.available():
    fn, n_bits, desc = molfp.get(name)
    print(f"{name:>30s}  {n_bits:>5}  {desc}")
```

## Explainability

```python
# Which atoms set which bits?
exp = molfp.explain("CCO", "ecfp4")
# exp[bit_index] = {"atoms": [0, 1], "radius": 2, "smiles": "CCO"}

# Chemical annotation for a bit
ann = molfp.annotate("maccs", 162)
# {"name": "Aromatic ring", "smarts": "c1ccccc1", ...}
```

## SHAP Interpretation

```python
result = molfp.interpret(smiles, activities, fp_type="ecfp4", metric="pchembl")
# result["shap_values"], result["top_bits"], result["class_drivers"]
```

## Collision-Free Fingerprints

```python
# Detect collisions
molfp.detect_collisions("c1ccccc1", "ecfp4", n_bits=1024)

# Collision-free mode (multi-hash)
fps = molfp.ecfp4(smiles, collision_free=True)
```

## Scaffold Analysis

```python
molfp.scaffold("CC(=O)Oc1ccccc1C(=O)O")
# {"murcko": "c1ccccc1", "generic": "C1CCCCC1", "name": "Benzene"}

# Scaffold-aware train/test split
result = molfp.scaffold_split(smiles, test_size=0.2)
```

## Molecular Graphs

```python
# Graph representation for GNNs
g = molfp.mol_graph("c1ccccc1", type="standard", enriched=True)

# Fixed-length vectors for sklearn
X = molfp.graph_vectors(smiles, type="standard", enriched=True)
```

## sklearn Pipelines

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(molfp.ECFPTransformer(), RandomForestClassifier())
pipe.fit(smiles_train, y_train)
```
