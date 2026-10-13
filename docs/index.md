<p align="center">
<img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/logo.png" alt="molfp logo" width="300">
</p>

<p align="center">
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/v/molfp" alt="PyPI"></a>
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/pyversions/molfp" alt="Python"></a>
<a href="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml"><img src="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/dataprofessor/molfp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/molfp" alt="License"></a>
</p>

# molfp

**Interpretable molecular fingerprints with SHAP-driven chemical explanations.**

A native Python/RDKit library that computes 30 molecular fingerprint types (13 2D + 9 3D + 8 graph-based) and provides built-in tools for interpretation, visualization, collision detection, activity cliff analysis, bioactivity profiling, and SHAP-based feature explanation.

## Install

```bash
pip install molfp
```

## Quick Example

```python
import molfp

# Compute fingerprints
fp = molfp.maccs("CC(=O)Oc1ccccc1C(=O)O")
fps = molfp.ecfp4(["CCO", "c1ccccc1"], collision_free=True)

# Explain which atoms set which bits
molfp.explain("CCO", "ecfp4")

# SHAP interpretation with chemical meaning
result = molfp.interpret(smiles, activities, fp_type="ecfp4", collision_aware=True)

# Scaffold-aware train/test split
result = molfp.scaffold_split(smiles, test_size=0.2)
```

## Features

- **30 fingerprint types** — 13 2D, 9 3D, 8 graph-based
- **Bit-level explainability** — atom attribution for every ON bit
- **Chemical annotation** — human-readable names for fingerprint bits
- **SHAP interpretation** — collision-aware feature importance with chemical meaning
- **Collision handling** — detect, remove, or avoid hash collisions entirely
- **Scaffold analysis** — Murcko scaffolds with ~130 named ring systems, scaffold-aware splitting
- **Activity cliffs** — detect structurally similar molecules with different activities
- **Bioactivity profiling** — enriched/depleted bit analysis
- **Graph representations** — 4 graph types with PyG/DGL converters
- **sklearn compatible** — 22 transformers + GraphTransformer
- **Visualization** — 2D drawing with bit highlighting, 3D interactive viewer
