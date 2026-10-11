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
