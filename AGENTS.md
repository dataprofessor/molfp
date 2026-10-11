# molfp — Development Notes

## Git & Repository

- **Git identity**: `Chanin Nantasenamat <dataprofessorofficial@gmail.com>` — always use this for commits. Set via `git config` in this repo.
- **Do NOT** add `Co-authored-by: Snowflake CoCo` to commit messages.
- **GitHub repo**: `dataprofessor/molfp` (fresh repo, clean contributor list).
- **PyPI publishing**: GitHub Actions trusted publisher via `pypi` environment. Triggered on release.

## Future Work

### Additional Molecular Graph Types

`mol_graph()` and `graph_vectors()` use a `kind` parameter to select the graph type:

```python
molfp.mol_graph(smiles, type="standard")    # current — atoms=nodes, bonds=edges
molfp.mol_graph(smiles, type="distance")    # all atom pairs, 3D distance weights
molfp.mol_graph(smiles, type="line")        # bonds=nodes, bond-bond connections
molfp.mol_graph(smiles, type="fragment")    # functional groups=nodes
```

Currently implemented: all 4 types.

| Type | Parameter | Nodes | Edges | GNN Use |
|------|-----------|-------|-------|---------|
| `"standard"` | default | Atoms | Bonds | GCN, GAT, GIN |
| `"distance"` | 3D required | Atoms | All pairs (3D dist) | SchNet, DimeNet |
| `"line"` | | Bonds | Bond-bond connections | D-MPNN, Chemprop |
| `"fragment"` | | Functional groups | Group connections | Hierarchical GNNs |

1. **`type="distance"`** — complete graph where every atom pair is connected, edge weight = 3D Euclidean distance. Used by SchNet, DimeNet, SphereNet. We already have `_ensure_conformer()` for 3D coordinates.

2. **`type="line"`** — line graph where bonds become nodes, connected when they share an atom. Each node carries both bond features and source atom features. Used by D-MPNN (Chemprop).

3. **`type="fragment"`** — coarse-grained graph where functional groups are nodes, edges represent connections between groups. We already have 307 substructure SMARTS and 20 functional group SMARTS that could define the fragments.

Priority: distance and line graphs are most widely used in practice.

## Key Findings

### Hash Collisions in Fingerprints

**Collision-prone (hashed):** ECFP4, RDK, RDK Extended, RDK GraphOnly, AtomPairs2D — use modulo folding, collisions are inevitable.

**Collision-free (SMARTS-based):** MACCS (166), PubChem (881), Substructure (307), Klekota-Roth (4860), E-State (79) — each bit has a fixed chemical definition, no hashing.

**3D and Graph FPs:** Not applicable — they produce continuous descriptors, not hashed bits.

### Aromatase Benchmark (2,359 molecules, 5-fold CV, Random Forest)

| Method | Dims | MCC |
|--------|------|-----|
| ECFP4 standard | 1024 | 0.785 |
| ECFP4 collision-free (183 bits removed) | 841 | 0.766 |
| ECFP4 multi-hash k=2 | 1024 | 0.784 |
| ECFP4 multi-hash k=3 | 1024 | 0.777 |
| Graph enriched | 5242 | 0.775 |
| Graph base | 298 | 0.680 |

- 183/1024 ECFP4 bits (18%) have collisions in the aromatase dataset
- Removing collisions loses information (MCC drops 0.019)
- Multi-hash k=2 matches standard performance while reducing collision probability exponentially
- Higher k (3) causes bit saturation at 1024 bits — diminishing returns

### Collision Handling API

Three-function approach (detect → remove OR resolve):
- `detect_collisions()` — identify which bits have collisions per molecule
- `remove_collisions()` — mask out colliding bits from feature matrix
- `ecfp4_multihash()` — multi-hash FP using k independent hash functions

Design decision: `ecfp4_multihash` is registered as a FP type since it produces a fundamentally different fingerprint (different bits set). It's not a post-processing step — it's an alternative computation. Could be generalized to other hashed FPs (rdk_multihash, atompairs2d_multihash) if needed.
