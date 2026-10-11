# molfp — Development Notes

## Future Work

### Additional Molecular Graph Types

`mol_to_graph()` and `graph_vectors()` use a `kind` parameter to select the graph type:

```python
molfp.mol_to_graph(smiles, kind="standard")    # current — atoms=nodes, bonds=edges
molfp.mol_to_graph(smiles, kind="distance")    # all atom pairs, 3D distance weights
molfp.mol_to_graph(smiles, kind="line")        # bonds=nodes, bond-bond connections
molfp.mol_to_graph(smiles, kind="fragment")    # functional groups=nodes
```

Currently implemented: `kind="standard"` (default).

Planned:

1. **`kind="distance"`** — complete graph where every atom pair is connected, edge weight = 3D Euclidean distance. Used by SchNet, DimeNet, SphereNet. We already have `_ensure_conformer()` for 3D coordinates.

2. **`kind="line"`** — line graph where bonds become nodes, connected when they share an atom. Each node carries both bond features and source atom features. Used by D-MPNN (Chemprop).

3. **`kind="fragment"`** — coarse-grained graph where functional groups are nodes, edges represent connections between groups. We already have 307 substructure SMARTS and 20 functional group SMARTS that could define the fragments.

Priority: distance and line graphs are most widely used in practice.
