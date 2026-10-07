"""
PubChem/CACTVS fingerprint (881 bits) — internal implementation.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from . import _smarts


_ELEMENT_TO_ANUM = {}
for _z in range(1, 119):
    _sym = Chem.GetPeriodicTable().GetElementSymbol(_z)
    _ELEMENT_TO_ANUM[_sym] = _z


def _classify_rings(mol):
    """Classify SSSR rings by size and type. Returns {(type, size): count}."""
    ri = mol.GetRingInfo()
    counts = {}

    for ring in ri.AtomRings():
        size = len(ring)
        atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]

        all_carbon = all(z == 6 for z in atomic_nums)
        has_nitrogen = any(z == 7 for z in atomic_nums)
        has_hetero = any(z not in (1, 6) for z in atomic_nums)

        bonds = []
        for i in range(size):
            b = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % size])
            if b:
                bonds.append(b)

        is_aromatic = all(b.GetIsAromatic() for b in bonds)
        is_saturated = all(b.GetBondType() == Chem.BondType.SINGLE for b in bonds)
        is_sat_or_arom = is_saturated or is_aromatic
        is_unsaturated = not is_saturated and not is_aromatic

        counts[("countAnyRing", size)] = counts.get(("countAnyRing", size), 0) + 1

        if is_sat_or_arom:
            if all_carbon:
                k = ("countSaturatedOrAromaticCarbonOnlyRing", size)
                counts[k] = counts.get(k, 0) + 1
            if has_nitrogen:
                k = ("countSaturatedOrAromaticNitrogenContainingRing", size)
                counts[k] = counts.get(k, 0) + 1
            if has_hetero:
                k = ("countSaturatedOrAromaticHeteroContainingRing", size)
                counts[k] = counts.get(k, 0) + 1

        if is_unsaturated:
            if all_carbon:
                k = ("countUnsaturatedCarbonOnlyRing", size)
                counts[k] = counts.get(k, 0) + 1
            if has_nitrogen:
                k = ("countUnsaturatedNitrogenContainingRing", size)
                counts[k] = counts.get(k, 0) + 1
            if has_hetero:
                k = ("countUnsaturatedHeteroContainingRing", size)
                counts[k] = counts.get(k, 0) + 1

        if is_aromatic:
            counts[("countAromaticRing", "any")] = counts.get(("countAromaticRing", "any"), 0) + 1
            if has_hetero:
                counts[("countHeteroAromaticRing", "any")] = counts.get(("countHeteroAromaticRing", "any"), 0) + 1

    return counts


def load_definitions() -> tuple[list[tuple[int, int, int]], list[tuple[int, str, int | str, int]], list[tuple[int, Chem.rdchem.Mol]]]:
    """Parse pubchem_fingerprint_definitions.txt.

    Returns (section1, section2, smarts_bits).
    """
    path = _smarts._data_path("pubchem_fingerprint_definitions.txt")
    section1, section2, smarts_bits = [], [], []
    current_section = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#") or line.startswith("====="):
                continue
            if "SECTION 1" in line:
                current_section = 1; continue
            elif "SECTION 2" in line:
                current_section = 2; continue
            elif any(f"SECTION {i}" in line for i in range(3, 8)):
                current_section = "smarts"; continue
            if line.startswith("Format:") or line.startswith("Ring type") or line.startswith("  "):
                continue

            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            bit = int(parts[0])

            if current_section == 1:
                anum = _ELEMENT_TO_ANUM.get(parts[1], 0)
                section1.append((bit, anum, int(parts[3])))
            elif current_section == 2:
                ring_size = parts[2] if parts[2] == "any" else int(parts[2])
                section2.append((bit, parts[1], ring_size, int(parts[4])))
            elif current_section == "smarts" and len(parts) > 1:
                pat = Chem.MolFromSmarts(parts[1])
                if pat is not None:
                    smarts_bits.append((bit, pat))

    return section1, section2, smarts_bits


class PubChemComputer:
    """Lazy-loaded PubChem FP computer."""

    def __init__(self):
        self._loaded = False
        self._s1 = self._s2 = self._smarts = None

    def _ensure_loaded(self):
        if not self._loaded:
            self._s1, self._s2, self._smarts = load_definitions()
            self._loaded = True

    def compute(self, mol: Chem.rdchem.Mol) -> list[int]:
        """Compute 881-bit PubChem FP for an RDKit Mol. Returns list of ints."""
        self._ensure_loaded()
        bits = [0] * 881

        # Section 1: element counts
        elem_counts = {}
        for atom in mol.GetAtoms():
            anum = atom.GetAtomicNum()
            elem_counts[anum] = elem_counts.get(anum, 0) + 1
        elem_counts[1] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())

        for bit, anum, threshold in self._s1:
            if elem_counts.get(anum, 0) >= threshold:
                bits[bit] = 1

        # Section 2: ring counts
        ring_counts = _classify_rings(mol)
        for bit, ring_type, ring_size, threshold in self._s2:
            if ring_counts.get((ring_type, ring_size), 0) >= threshold:
                bits[bit] = 1

        # Sections 3-7: SMARTS
        for bit, pat in self._smarts:
            if mol.HasSubstructMatch(pat):
                bits[bit] = 1

        return bits
