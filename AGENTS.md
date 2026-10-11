# molfp — Development Notes

## Future Work

### Additional Molecular Graph Types

`mol_graph()` and `graph_vectors()` use a `kind` parameter to select the graph type:

```python
molfp.mol_graph(smiles, type="standard")    # current — atoms=nodes, bonds=edges
molfp.mol_graph(smiles, type="distance")    # all atom pairs, 3D distance weights
molfp.mol_graph(smiles, type="line")        # bonds=nodes, bond-bond connections
molfp.mol_graph(smiles, type="fragment")    # functional groups=nodes
```

Currently implemented: `type="standard"` (default).

Planned:

1. **`type="distance"`** — complete graph where every atom pair is connected, edge weight = 3D Euclidean distance. Used by SchNet, DimeNet, SphereNet. We already have `_ensure_conformer()` for 3D coordinates.

2. **`type="line"`** — line graph where bonds become nodes, connected when they share an atom. Each node carries both bond features and source atom features. Used by D-MPNN (Chemprop).

3. **`type="fragment"`** — coarse-grained graph where functional groups are nodes, edges represent connections between groups. We already have 307 substructure SMARTS and 20 functional group SMARTS that could define the fragments.

Priority: distance and line graphs are most widely used in practice.
