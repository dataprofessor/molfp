"""
Structure normalization: standardize molecules before fingerprint computation.

Provides a pipeline for cleaning, desalting, normalizing, and deduplicating
molecules — critical preprocessing before fingerprint calculation to ensure
consistent and reproducible results.

Pipeline (following Greg Landrum's RSC OpenScience protocol):
  1. Parse SMILES → RDKit Mol
  2. Cleanup: remove Hs, disconnect metals, normalize, reionize
  3. Desalt: keep largest fragment (FragmentParent)
  4. Neutralize: remove charges where possible (Uncharger)
  5. Canonicalize tautomer: pick a single canonical tautomer form
  6. Canonical SMILES: generate standardized SMILES string
  7. InChIKey: compute for deduplication
"""

from __future__ import annotations

from typing import Any, Union

from rdkit import Chem
from rdkit.Chem import inchi as rdInchi
from rdkit.Chem.MolStandardize import rdMolStandardize

from .fingerprints import _ensure_mol


_uncharger = rdMolStandardize.Uncharger()
_te = rdMolStandardize.TautomerEnumerator()


def standardize_mol(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol | None:
    """Full standardization pipeline. Returns standardized RDKit Mol or None.

    Steps: parse → cleanup → desalt → neutralize → canonical tautomer.
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
    if mol is None:
        return None

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _uncharger.uncharge(mol)
        mol = _te.Canonicalize(mol)
    except Exception:
        return None

    return mol


# ── Mol ↔ SMILES conversion ──────────────────────────────────────────────────

def smiles_to_mol(input_data, *, column=None, standardize=False):
    """Convert SMILES to RDKit Mol objects.

    Args:
        input_data: SMILES string, list of SMILES, pd.Series, or pd.DataFrame
        column: DataFrame column name (auto-detected if None)
        standardize: if True, apply full standardization pipeline

    Returns:
        Single Mol (for string input) or list[Mol|None] (for batch input).
        Returns None for invalid SMILES.
    """
    data, is_batch = _extract_input(input_data, column)

    if not is_batch:
        if standardize:
            return standardize_mol(data)
        return Chem.MolFromSmiles(data) if isinstance(data, str) else data

    results = []
    for smi in data:
        if standardize:
            results.append(standardize_mol(smi))
        else:
            results.append(Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi)
    return results


def mol_to_smiles(input_data, *, column=None, canonical=True):
    """Convert RDKit Mol objects (or SMILES) to canonical SMILES strings.

    Args:
        input_data: Mol, SMILES string, list, pd.Series, or pd.DataFrame
        column: DataFrame column name (auto-detected if None)
        canonical: if True (default), return canonical SMILES

    Returns:
        Single SMILES string (for single input) or list[str|None] (for batch).
    """
    data, is_batch = _extract_input(input_data, column)

    if not is_batch:
        return _to_smiles(data, canonical)

    return [_to_smiles(item, canonical) for item in data]


def _to_smiles(mol_or_smi, canonical=True):
    """Convert a single Mol or SMILES to a canonical SMILES string."""
    if mol_or_smi is None:
        return None
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    elif isinstance(mol_or_smi, Chem.rdchem.Mol):
        mol = mol_or_smi
    else:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=canonical)


def _extract_input(input_data, column=None):
    """Extract data from various input types. Returns (data, is_batch)."""
    if isinstance(input_data, str):
        return input_data, False

    if isinstance(input_data, Chem.rdchem.Mol):
        return input_data, False

    try:
        import pandas as pd
        if isinstance(input_data, pd.DataFrame):
            if column is not None:
                return input_data[column].tolist(), True
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles", "CANONICAL_SMILES",
                        "mol", "Mol", "MOL", "molecule", "smi"]:
                if col in input_data.columns:
                    return input_data[col].tolist(), True
            raise ValueError(
                "DataFrame has no recognized SMILES column. "
                "Pass column= or use a column named 'SMILES', 'smiles', or 'canonical_smiles'."
            )
        if isinstance(input_data, pd.Series):
            return input_data.tolist(), True
    except ImportError:
        pass

    if hasattr(input_data, '__iter__') and not isinstance(input_data, (str, bytes)):
        return list(input_data), True

    return input_data, False


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string. Returns canonical SMILES or None."""
    mol = standardize_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_inchikey(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> str | None:
    """Compute InChIKey for a molecule (after standardization)."""
    mol = standardize_mol(smiles_or_mol)
    if mol is None:
        return None
    inchi_str = rdInchi.MolToInchi(mol)
    if inchi_str is None:
        return None
    return rdInchi.InchiToInchiKey(inchi_str)


def desalt(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol | None:
    """Remove salts/counterions, keep the largest fragment."""
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
    if mol is None:
        return None
    return rdMolStandardize.FragmentParent(mol)


def neutralize(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol | None:
    """Remove charges where chemically reasonable."""
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    return _uncharger.uncharge(mol)


def canonicalize_tautomer(smiles_or_mol: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol | None:
    """Pick the canonical tautomer form."""
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    return _te.Canonicalize(mol)


# ── Batch operations ──────────────────────────────────────────────────────────

def standardize_batch(smiles_list: list[str], return_failures: bool = False) -> list[str | None] | tuple[list[str | None], list[tuple[int, str]]]:
    """Standardize a list of SMILES. Returns list of canonical SMILES.

    Args:
        smiles_list: list of SMILES strings
        return_failures: if True, return (results, failures) tuple

    Returns:
        list of standardized SMILES (None for failures)
        if return_failures: (results, [(idx, original_smiles), ...])
    """
    results = []
    failures = []
    for i, smi in enumerate(smiles_list):
        std = standardize_smiles(smi)
        results.append(std)
        if std is None:
            failures.append((i, smi))

    if return_failures:
        return results, failures
    return results


def deduplicate(smiles_list: list[str], method: str = "inchikey") -> dict[str, Any]:
    """Deduplicate molecules using InChIKey or canonical SMILES.

    Args:
        smiles_list: list of SMILES strings
        method: "inchikey" (default, handles tautomers/salts) or "canonical"

    Returns:
        dict with:
            unique_smiles: list of unique standardized SMILES
            unique_indices: indices of first occurrence
            duplicates: list of (idx, original_smiles, duplicate_of_idx)
            mapping: {original_idx: unique_idx}
            n_input: total input
            n_unique: unique count
            n_duplicates: duplicate count
            n_failed: failed to parse
    """
    seen = {}
    unique_smiles = []
    unique_indices = []
    duplicates = []
    mapping = {}
    failed = 0

    for i, smi in enumerate(smiles_list):
        mol = standardize_mol(smi)
        if mol is None:
            failed += 1
            mapping[i] = None
            continue

        if method == "inchikey":
            inchi_str = rdInchi.MolToInchi(mol)
            key = rdInchi.InchiToInchiKey(inchi_str) if inchi_str else None
        else:
            key = Chem.MolToSmiles(mol)

        if key is None:
            failed += 1
            mapping[i] = None
            continue

        if key in seen:
            duplicates.append((i, smi, seen[key]))
            mapping[i] = seen[key]
        else:
            seen[key] = len(unique_smiles)
            unique_smiles.append(Chem.MolToSmiles(mol))
            unique_indices.append(i)
            mapping[i] = seen[key]

    return {
        "unique_smiles": unique_smiles,
        "unique_indices": unique_indices,
        "duplicates": duplicates,
        "mapping": mapping,
        "n_input": len(smiles_list),
        "n_unique": len(unique_smiles),
        "n_duplicates": len(duplicates),
        "n_failed": failed,
    }


def preprocess(smiles_list: list[str], deduplicate_method: str = "inchikey") -> dict[str, Any]:
    """Full preprocessing pipeline: standardize + deduplicate.

    Convenience function combining standardize_batch and deduplicate.

    Returns:
        dict with:
            clean_smiles: list of unique standardized SMILES
            clean_mols: list of RDKit Mol objects (ready for FP computation)
            dedup_result: full deduplication result dict
            standardization_failures: [(idx, smiles), ...]
    """
    std_smiles, failures = standardize_batch(smiles_list, return_failures=True)
    valid_smiles = [s for s in std_smiles if s is not None]
    valid_indices = [i for i, s in enumerate(std_smiles) if s is not None]

    dedup = deduplicate(valid_smiles, method=deduplicate_method)

    clean_mols = []
    for smi in dedup["unique_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            clean_mols.append(mol)

    return {
        "clean_smiles": dedup["unique_smiles"],
        "clean_mols": clean_mols,
        "dedup_result": dedup,
        "standardization_failures": failures,
        "n_input": len(smiles_list),
        "n_after_standardization": len(valid_smiles),
        "n_after_deduplication": len(dedup["unique_smiles"]),
    }
