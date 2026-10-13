# Getting Started

## Installation

```bash
pip install molfp
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
