# molfp — Development Notes

## Future Work

### Additional Molecular Graph Types

Currently `mol_to_graph()` produces a standard molecular graph (atoms as nodes, bonds as edges). The following graph types are used by different GNN architectures and could be added:

1. **Complete/Distance graph** — every atom pair connected, edge weight = 3D distance. Used by SchNet, DimeNet, SphereNet. We already have `_ensure_conformer()` for 3D coordinates.

2. **Line graph** — bonds become nodes, connected when they share an atom. Used by D-MPNN (Chemprop). Each node carries both bond features and source atom features.

3. **Fragment graph** — functional groups as nodes, connections between groups as edges. We already have 307 substructure SMARTS and 20 functional group SMARTS that could define the fragments.

4. **Atom+Bond graph** — both atoms and bonds as nodes in a bipartite-like structure. Used by AttentiveFP.

Priority: complete/distance graph and line graph are most widely used in practice.
