"""
SMARTS pattern loading, compilation, and matching utilities.
"""

from __future__ import annotations

import importlib.resources
import os

from rdkit import Chem


def _data_path(filename):
    """Resolve path to a data file in molfp/data/."""
    return os.path.join(os.path.dirname(__file__), "data", filename)


def load_smarts_file(filename: str) -> list[tuple[str, str]]:
    """Load SMARTS patterns from a data file. Returns list of (name, smarts) tuples."""
    path = _data_path(filename)
    patterns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line and not line.startswith("["):
                name, smarts = line.split(":", 1)
                patterns.append((name.strip(), smarts.strip()))
            else:
                patterns.append((f"bit_{len(patterns)}", line))
    return patterns


def load_plain_smarts(filename: str) -> list[tuple[str, str]]:
    """Load a file where each non-empty, non-comment line is a SMARTS string."""
    path = _data_path(filename)
    patterns = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append((f"bit_{i}", line))
    return patterns


def compile_smarts(patterns: list[tuple[str, str]]) -> tuple[list[tuple[str, Chem.rdchem.Mol | _SaltMatcher]], list[tuple[str, str]]]:
    """Compile (name, smarts) list. Returns (compiled, failed).

    Handles special patterns like the Salt disconnected-charge pattern.
    """
    compiled = []
    failed = []
    for name, smarts in patterns:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled.append((name, pat))
        elif _is_salt_pattern(smarts):
            compiled.append((name, _SaltMatcher()))
        else:
            failed.append((name, smarts))
    return compiled, failed


def _is_salt_pattern(smarts):
    return "[-1,-2,-3,-4,-5,-6,-7" in smarts and "+1,+2,+3,+4,+5,+6,+7" in smarts


class _SaltMatcher:
    pass


def _check_salt(mol):
    has_neg = has_pos = False
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc < 0:
            has_neg = True
        if fc > 0:
            has_pos = True
        if has_neg and has_pos:
            return True
    return False


def match_binary(mol: Chem.rdchem.Mol, compiled_patterns: list[tuple[str, Chem.rdchem.Mol | _SaltMatcher]]) -> list[int]:
    """Match patterns, return list of 0/1 ints."""
    result = []
    for _, pat in compiled_patterns:
        if isinstance(pat, _SaltMatcher):
            result.append(1 if _check_salt(mol) else 0)
        else:
            result.append(1 if mol.HasSubstructMatch(pat) else 0)
    return result


def match_count(mol: Chem.rdchem.Mol, compiled_patterns: list[tuple[str, Chem.rdchem.Mol | _SaltMatcher]]) -> list[int]:
    """Count pattern matches, return list of ints."""
    result = []
    for _, pat in compiled_patterns:
        if isinstance(pat, _SaltMatcher):
            result.append(1 if _check_salt(mol) else 0)
        else:
            result.append(len(mol.GetSubstructMatches(pat)))
    return result
