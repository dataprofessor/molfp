# Conformer Convergence Analysis

## Overview

molfp's 3D fingerprints require optimized 3D conformers. This document details the empirical analysis
used to derive the adaptive iteration formula for MMFF94 force field optimization.

## Dataset

- **Source**: CureFFI.org FDA-approved drug list (1,497 drugs with SMILES)
- **Filtered**: removed metals, >80 heavy atoms, invalid SMILES
- **Final dataset**: 1,420 standard drug-like molecules
- **Iteration range**: 25–1000 (measured on coarse grid: 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000)
- **Median iterations**: 200

## Molecular Descriptors

| Descriptor | Description |
|---|---|
| `heavy_atoms` | Number of non-hydrogen atoms |
| `rot_bonds` | Number of rotatable bonds |
| `rings` | Total ring count |
| `arom_rings` | Aromatic ring count |
| `hba` | H-bond acceptors |
| `hbd` | H-bond donors |
| `mw` | Molecular weight |
| `tpsa` | Topological polar surface area |
| `bonds` | Total bond count |
| `fsp3` | Fraction of sp3-hybridized carbons |

## Full Model (10 features)

Standardized coefficients (relative importance):

| Descriptor | Standardized β | Interpretation |
|---|---|---|
| **heavy_atoms** | +233 | Dominant — more atoms = more iterations |
| bonds | -104 | Correlated with atoms (collinear) |
| hba | +81 | H-bond acceptors add conformational flexibility |
| tpsa | -64 | Polar area correlates with rigidity |
| rot_bonds | +44 | Each rotatable bond adds ~9 iterations |
| arom_rings | -32 | Aromatic rings are planar/rigid |
| mw | -27 | Collinear with atom count |
| rings | -12 | Ring constraints reduce DOF |
| fsp3 | -7 | sp3 fraction = tetrahedral flexibility |
| hbd | -0.4 | Negligible |

## Model Selection

| Variables | R² | MAE (iters) |
|---|---|---|
| atoms + rot | 0.770 | 53 |
| atoms + rot + rings | 0.778 | 51 |
| **atoms + rot + rings + fsp3** | **0.789** | **49** |
| atoms + rot + bonds | 0.778 | 51 |
| mw + rot + rings | 0.737 | 54 |

**Selected: 4-variable model** (atoms, rot_bonds, rings, fsp3) — best balance of accuracy and simplicity.

## Train/Test Validation (80/20 split)

### Measurement Grid
- **Coarse grid** (first round): [25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000] — uneven steps up to 250
- **Fine grid** (final): step=25 from 25 to 1000 — uniform resolution

The coarse grid overestimates convergence iterations by an average of **31 iters** due to
measurement noise (mean coarse=267 vs fine=236 on the same 296 molecules).

### Controlled Comparison (same 296 molecules, same 236/60 split)

| | Kennard-Stone | | Random | |
|---|---|---|---|---|
| | Coarse | Fine | Coarse | Fine |
| **R² train** | 0.777 | 0.806 | 0.750 | 0.776 |
| **R² test** | 0.446 | 0.447 | 0.798 | 0.820 |
| **MAE train** | 54 | **43** | 52 | **40** |
| **MAE test** | 49 | **35** | 53 | **42** |

Fine grid improved MAE by 20–28% on identical molecules and splits — purely by reducing measurement noise.

### Split Methods
- **Kennard-Stone**: selects test set to uniformly cover the descriptor space (PCA-based).
  Produces a harder test set — positive train-test gap is expected.
- **Random**: standard random 80/20 split (seed=42).
  Can be optimistic — negative gap means easy molecules land in test by chance.

### Scaled-up Validation (1,420 FDA drugs, coarse grid)

To confirm on a larger sample, the coarse grid was run on all 1,420 valid FDA drugs:

| Metric | Kennard-Stone | | | Random | | |
|---|---|---|---|---|---|---|
| | Train | Test | Gap | Train | Test | Gap |
| **R² (4 feat)** | 0.761 | 0.737 | +0.024 | 0.751 | 0.817 | -0.066 |
| **MAE** | 55 | 44 | | 54 | 50 | |
| **% within ±100** | 94% | | | 90% | | |

### Coefficients (standardized features)

| Feature | Kennard-Stone | Random |
|---|---|---|
| heavy_atoms | +130.5 | +125.5 |
| rot_bonds | +28.5 | +33.2 |
| rings | -26.6 | -21.8 |
| fsp3 | +9.3 | +10.1 |

### Test Set Accuracy

| Metric | Kennard-Stone | Random |
|---|---|---|
| % within ±100 iters | **94%** | 90% |

## Key Findings

1. **Kennard-Stone produces a harder test set** — the train-test R² gap is positive (+0.024),
   meaning the model performs slightly worse on the diverse test set. This is the expected behavior:
   KS ensures the test set covers the full chemical space, including edge cases.

2. **Random split can be optimistic** — the negative gap (-0.066) means the test R² is actually
   *higher* than train, which happens when easy molecules land in the test set by chance.

3. **Coefficients are stable across splits** — heavy_atoms dominates in both (125–131),
   rot_bonds (+29–33), rings (-22 to -27), fsp3 (+9–10). The model is robust.

4. **94% of Kennard-Stone test molecules predicted within ±100 iterations** — the 6% that
   fall outside are caught by the 2×/4× escalation in `_ensure_conformer`.

## Final Formula (Implemented)

```python
iters = max(200, min(2000, 1.5 × (-25 + 8.7×atoms + 10.4×rot - 8.1×rings + 39.5×fsp3)))
```

Trained on 296 FDA drugs with uniform step=25 grid. R²=0.793, MAE=41 iters.
97.0% first-pass convergence with 1.5× safety margin.

### Chemical Interpretation
- **Rings reduce iterations**: each ring constrains 3 DOF (rigid geometry)
- **Rotatable bonds increase iterations**: each adds a torsional DOF to optimize
- **fsp3 increases iterations**: sp3 carbons are tetrahedral with more conformational freedom than flat aromatic atoms
- **Heavy atom count dominates**: simply more 3D coordinates to minimize

### Fallback Strategy
```
Estimate iters → optimize → not converged? → 2× iters → not converged? → 4× iters
```
Caps at 4000 iterations total. Molecules that still don't converge are tagged `not_converged`.

## Molecule Routing

| Class | Detection | Embedding | Force Field |
|---|---|---|---|
| Standard | No metals, ≤80 atoms | ETKDG | MMFF94 (adaptive) |
| Metal | Contains transition/lanthanide metal | ETKDG | UFF (skip MMFF) |
| Large | >80 heavy atoms | ETKDGv3 + random coords | MMFF94 |
| Metal + Large | Both | ETKDGv3 + random coords | UFF |

## FDA Drug Benchmark (200 sampled)

| Status | Count | % |
|---|---|---|
| converged_mmff | 191 | 95.5% |
| converged_uff | 7 | 3.5% |
| embed_failed | 2 | 1.0% |
| **Overall convergence** | **198** | **99.0%** |

The 2 embed failures were cyanocobalamin (MW 1356, 93 atoms) and telavancin (MW 1792, 122 atoms) —
both exceed the 80-atom threshold and would be routed to the `large` strategy.
