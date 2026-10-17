<p align="center">
<img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/logo.png" alt="molfp logo" width="300">
</p>

<p align="center">
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/v/molfp" alt="PyPI"></a>
<a href="https://pypi.org/project/molfp/"><img src="https://img.shields.io/pypi/pyversions/molfp" alt="Python"></a>
<a href="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml"><img src="https://github.com/dataprofessor/molfp/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/dataprofessor/molfp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/molfp" alt="License"></a>
<a href="https://colab.research.google.com/github/dataprofessor/molfp/blob/main/examples/getting_started_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

# molfp

**Interpretable molecular fingerprints with SHAP-driven chemical explanations.**

A native Python/RDKit library that computes 35 molecular fingerprint types (18 2D + 9 3D + 8 graph-based) and provides built-in tools for interpretation, visualization, collision detection, activity cliff analysis, bioactivity profiling, and SHAP-based feature explanation.

## Install

```bash
pip install molfp
```

Requires Python 3.9+. RDKit is installed automatically. See [Getting Started](getting-started.md) for optional dependencies and troubleshooting.

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

- **35 fingerprint types** — 18 2D, 9 3D, 8 graph-based
- **Bit-level explainability** — atom attribution for every ON bit
- **Chemical annotation** — human-readable names for fingerprint bits
- **SHAP interpretation** — collision-aware feature importance with chemical meaning
- **Collision handling** — detect, remove, or avoid hash collisions entirely
- **Scaffold analysis** — Murcko scaffolds with ~130 named ring systems, scaffold-aware splitting
- **Activity cliffs** — detect structurally similar molecules with different activities
- **Bioactivity profiling** — enriched/depleted bit analysis
- **Isomer discrimination** — test whether fingerprints distinguish structural isomers
- **Graph representations** — 4 graph types with PyG/DGL converters
- **sklearn compatible** — 22 transformers + GraphTransformer
- **Visualization** — 2D drawing with bit highlighting, 3D interactive viewer

## Visualization

<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/draw_2d_grid.png" alt="2D molecule grid" width="500"></td>
<td align="center"><img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/draw_2d_highlight.png" alt="2D with fingerprint bit highlighting" width="220"></td>
</tr>
<tr>
<td align="center"><em>Molecule grid with labels</em></td>
<td align="center"><em>Fingerprint bit highlighting</em></td>
</tr>
</table>

```python
molfp.draw(smiles, legends=["Aspirin", "Caffeine"])
molfp.draw("CC(=O)Oc1ccccc1C(=O)O", highlight_bits=[88, 143, 162], fp_type="maccs")
```

### 3D Interactive Viewer

```python
molfp.view3d("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", style="ball_and_stick", spin=True)
```

<p align="center">
<img src="https://raw.githubusercontent.com/dataprofessor/molfp/main/assets/view3d_notebook.png" alt="3D interactive viewer" width="600">
</p>

## 35 Fingerprint Types

### 2D Fingerprints (18 types)

| Type | Bits | Function |
|------|------|----------|
| MACCS Keys | 166 | `maccs()` |
| ECFP4 (Morgan) | 1024 | `ecfp4()` |
| PubChem/CACTVS | 881 | `pubchem()` |
| Substructure FP | 307 | `substructure()` |
| Substructure Count | 307 | `substructure_count()` |
| Klekota-Roth | 4860 | `klekota_roth()` |
| Klekota-Roth Count | 4860 | `klekota_roth_count()` |
| E-State | 79 | `estate()` |
| AtomPairs2D | 780 | `atompairs2d()` |
| AtomPairs2D Count | 780 | `atompairs2d_count()` |
| RDK | 1024 | `rdk()` |
| RDK Extended | 1024 | `rdk_ext()` |
| RDK GraphOnly | 1024 | `rdk_graphonly()` |
| FCFP4 (Feature Morgan) | 1024 | `fcfp4()` |
| Lingo | 1024 | `lingo()` |
| Shortest Path | 1024 | `shortest_path()` |
| Hybridization | 1024 | `hybridization()` |
| Signature | 1024 | `signature()` |

### 3D Fingerprints (9 types)

| Type | Dims | Function |
|------|------|----------|
| AtomPairs3D | 780 | `atompairs3d()` |
| AtomPairs3D Count | 780 | `atompairs3d_count()` |
| USR | 12 | `usr()` |
| USRCAT | 60 | `usrcat()` |
| Autocorr3D | 80 | `autocorr3d()` |
| RDF | 210 | `rdf()` |
| MORSE | 224 | `morse()` |
| WHIM | 114 | `whim()` |
| GETAWAY | 273 | `getaway()` |

### Graph Fingerprints (8 types)

| Type | Function |
|------|----------|
| Standard | `mol_graph(type="standard")` |
| Standard Enriched | `mol_graph(type="standard", enriched=True)` |
| Distance | `mol_graph(type="distance")` |
| Distance Enriched | `mol_graph(type="distance", enriched=True)` |
| Line | `mol_graph(type="line")` |
| Line Enriched | `mol_graph(type="line", enriched=True)` |
| Fragment | `mol_graph(type="fragment")` |
| Fragment Enriched | `mol_graph(type="fragment", enriched=True)` |
