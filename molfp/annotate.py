"""
Chemical annotation: human-readable names and descriptions for fingerprint bits.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import MACCSkeys

from . import _smarts
from .fingerprints import _get_substructure_patterns, _get_kr_patterns, _get_pubchem_computer

_annotation_cache = {}


def annotate(fp_type: str, bit_idx: int) -> dict[str, Any]:
    """Get human-readable annotation for a specific fingerprint bit.

    Args:
        fp_type: fingerprint name (e.g. "substructure", "pubchem", "maccs")
        bit_idx: bit index (0-based)

    Returns:
        dict with keys: name, smarts (if applicable), description, category
    """
    annotations = get_annotations(fp_type)
    if bit_idx not in annotations:
        return {"name": f"bit_{bit_idx}", "description": "Unknown bit", "category": "unknown"}
    return annotations[bit_idx]


def get_annotations(fp_type: str) -> dict[int, dict[str, Any]]:
    """Get all annotations for a fingerprint type. Returns {bit_idx: annotation_dict}."""
    if fp_type in _annotation_cache:
        return _annotation_cache[fp_type]

    dispatch = {
        "maccs": _annotate_maccs,
        "substructure": _annotate_substructure,
        "substructure_count": _annotate_substructure,
        "klekota_roth": _annotate_kr,
        "klekota_roth_count": _annotate_kr,
        "pubchem": _annotate_pubchem,
        "estate": _annotate_estate,
        "ecfp4": _annotate_hashed,
        "atompairs2d": _annotate_hashed,
        "atompairs2d_count": _annotate_hashed,
        "rdk": _annotate_hashed,
        "rdk_ext": _annotate_hashed,
        "rdk_graphonly": _annotate_hashed,
    }
    if fp_type not in dispatch:
        raise KeyError(f"Unknown fp_type: {fp_type}")

    result = dispatch[fp_type](fp_type)
    _annotation_cache[fp_type] = result
    return result


def _annotate_maccs(fp_type):
    annotations = {}
    patts = MACCSkeys.smartsPatts
    for bit_idx in range(166):
        rdkit_bit = bit_idx + 1
        if rdkit_bit in patts:
            entry = patts[rdkit_bit]
            sma = entry[0] if len(entry) > 0 else ""
            desc = entry[1] if len(entry) > 1 else ""
            if not isinstance(desc, str):
                desc = f"MACCS key {rdkit_bit}"
            annotations[bit_idx] = {
                "name": f"MACCS_{rdkit_bit}",
                "smarts": sma,
                "description": desc if desc else f"MACCS key {rdkit_bit}",
                "category": "substructure",
            }
        else:
            annotations[bit_idx] = {
                "name": f"MACCS_{rdkit_bit}",
                "description": f"MACCS key {rdkit_bit} (no public definition)",
                "category": "substructure",
            }
    return annotations


def _annotate_substructure(fp_type):
    patterns = _get_substructure_patterns()
    annotations = {}
    for i, (name, pat) in enumerate(patterns):
        smarts = ""
        if isinstance(pat, _smarts._SaltMatcher):
            smarts = "([-charge]).([+charge])"
        else:
            smarts = Chem.MolToSmarts(pat) if pat else ""
        annotations[i] = {
            "name": name,
            "smarts": smarts,
            "description": f"Inte:Ligand functional group: {name}",
            "category": "functional_group",
        }
    return annotations


def _annotate_kr(fp_type):
    raw = _smarts.load_plain_smarts("klekota_roth_smarts.txt")
    annotations = {}
    for i, (name, smarts_str) in enumerate(raw):
        annotations[i] = {
            "name": f"KRFP{i + 1}",
            "smarts": smarts_str,
            "description": f"Klekota-Roth substructure {i + 1}",
            "category": "bioactivity_substructure",
        }
    return annotations


def _annotate_pubchem(fp_type):
    computer = _get_pubchem_computer()
    computer._ensure_loaded()
    annotations = {}

    for bit, anum, threshold in computer._s1:
        sym = Chem.GetPeriodicTable().GetElementSymbol(anum) if anum > 0 else "?"
        annotations[bit] = {
            "name": f"PubchemFP{bit}",
            "description": f"Element count: {sym} >= {threshold}",
            "category": "element_count",
        }

    for bit, ring_type, ring_size, threshold in computer._s2:
        annotations[bit] = {
            "name": f"PubchemFP{bit}",
            "description": f"Ring: {ring_type} size={ring_size} >= {threshold}",
            "category": "ring_count",
        }

    for bit, pat in computer._smarts:
        smarts = Chem.MolToSmarts(pat) if pat else ""
        section = "3-7"
        if 263 <= bit <= 326:
            section = "3 (atom pairs)"
        elif 327 <= bit <= 415:
            section = "4 (nearest neighbors)"
        elif 416 <= bit <= 459:
            section = "5 (detailed neighbors)"
        elif 460 <= bit <= 712:
            section = "6 (bond-specific)"
        elif 713 <= bit <= 880:
            section = "7 (ring substitution)"
        annotations[bit] = {
            "name": f"PubchemFP{bit}",
            "smarts": smarts,
            "description": f"Section {section} substructure",
            "category": "smarts",
        }

    return annotations


def _annotate_estate(fp_type):
    from rdkit.Chem.EState import AtomTypes as AT
    AT.BuildPatts()
    patterns = AT.esPatterns

    atom_type_names = [
        "sLi","ssBe","ssssBe","ssBH","sssB","ssssB",
        "sCH3","dCH2","ssCH2","tCH","dsCH","aaCH","sssCH","ddC","tsC",
        "dssC","aasC","aaaC","ssssC","sNH3","sNH2","ssNH2","dNH","ssNH",
        "aaNH","tN","sssNH","dsN","aaN","sssN","ddsN","aasN","ssssN",
        "sOH","dO","ssO","aaO","sF","sSiH3","ssSiH2","sssSiH","ssssSi",
        "sPH2","ssPH","sssP","dsssP","sssssP","sSH","dS","ssS","aaS",
        "dssS","ddssS","ssssssS","sCl","sGeH3","ssGeH2","sssGeH","ssssGe",
        "sAsH2","ssAsH","sssAs","sssdAs","sssssAs","sSeH","dSe","ssSe",
        "aaSe","dssSe","ddssSe","sBr","sSnH3","ssSnH2","sssSnH","ssssSn",
        "sI","sPbH3","ssPbH2","sssPbH","ssssPb",
    ]
    annotations = {}
    for i in range(79):
        name = atom_type_names[i] if i < len(atom_type_names) else f"EState_{i}"
        smarts = ""
        if patterns and i < len(patterns):
            _, pat = patterns[i]
            if pat is not None:
                smarts = Chem.MolToSmarts(pat)
        annotations[i] = {
            "name": name,
            "smarts": smarts,
            "description": f"E-State atom type: {name}",
            "category": "atom_type",
        }
    return annotations


def _annotate_hashed(fp_type):
    """For hashed fingerprints, bits don't have fixed meanings."""
    return {}  # empty — use explain() with a specific molecule instead
