# API Reference

All functions are importable directly from `molfp`:

```python
import molfp
molfp.maccs("CCO")
```

## Module Overview

The sidebar modules are ordered by a typical cheminformatics workflow — from computing fingerprints through analysis to model building:

```mermaid
graph TD
    FP["<b>Fingerprints</b><br/>Compute 30 FP types"]
    EA["<b>Explain &<br/>Annotate</b>"]
    CO["<b>Collisions</b>"]
    SC["<b>Scaffold</b>"]
    IN["<b>Interpret</b><br/>SHAP"]
    SD["<b>Similarity &<br/>Distance</b>"]
    BA["<b>Bioactivity</b>"]
    AC["<b>Activity<br/>Cliffs</b>"]
    FU["<b>Fusion</b>"]
    GR["<b>Graph</b>"]
    FS["<b>Feature<br/>Selection</b>"]
    SK["<b>sklearn</b><br/>Transformers"]
    MB["<b>Model Building</b>"]

    FP --> EA
    FP --> CO
    FP --> SC
    EA --> IN
    CO --> SD
    IN --> BA
    SD --> AC
    BA --> FU
    AC --> FU
    FU --> GR
    FU --> FS
    FU --> SK
    SC --> GR
    SC --> FS
    SC --> SK
    GR --> MB
    FS --> MB
    SK --> MB

    style FP fill:#2563eb,color:#fff,stroke:#1d4ed8
    style EA fill:#3b82f6,color:#fff,stroke:#2563eb
    style CO fill:#3b82f6,color:#fff,stroke:#2563eb
    style SC fill:#3b82f6,color:#fff,stroke:#2563eb
    style IN fill:#3b82f6,color:#fff,stroke:#2563eb
    style SD fill:#3b82f6,color:#fff,stroke:#2563eb
    style BA fill:#0d9488,color:#fff,stroke:#0f766e
    style AC fill:#0d9488,color:#fff,stroke:#0f766e
    style FU fill:#0d9488,color:#fff,stroke:#0f766e
    style GR fill:#0d9488,color:#fff,stroke:#0f766e
    style FS fill:#0d9488,color:#fff,stroke:#0f766e
    style SK fill:#0d9488,color:#fff,stroke:#0f766e
    style MB fill:#1e293b,color:#fff,stroke:#334155
```

*Utilities not shown: [Visualization](viz.md) (2D/3D rendering) · [Standardize](standardize.md) (SMILES cleaning)*

| Stage | Modules | Purpose |
|-------|---------|---------|
| **Compute** | Fingerprints | Generate 30 molecular fingerprint types |
| **Understand** | Explain & Annotate, Interpret | Bit-level attribution, SHAP with chemical meaning |
| **Validate** | Collisions, Similarity & Distance | Hash collision handling, pairwise metrics |
| **Analyze** | Bioactivity, Activity Cliffs | Enriched/depleted bits, structure-activity cliffs |
| **Combine** | Fusion, Scaffold | Consensus FPs, scaffold extraction and splitting |
| **Represent** | Graph, Feature Selection | GNN-ready graphs, variance/MI feature filtering |
| **Model** | sklearn | 22 pipeline-ready transformers + GraphTransformer |
| **Utilities** | Visualization, Standardize | 2D/3D rendering, SMILES cleaning and deduplication |
