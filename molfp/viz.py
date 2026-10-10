"""
Molecule visualization for Jupyter, Colab, and Streamlit.

2D rendering uses RDKit Draw (no extra dependencies).
3D interactive viewing uses py3Dmol (optional: pip install py3Dmol).
"""

from __future__ import annotations

import sys
from typing import Union

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from .fingerprints import _ensure_mol, _ensure_conformer, _extract_smiles


# ── Environment detection ────────────────────────────────────────────────────

def _detect_env():
    """Detect runtime environment: 'jupyter', 'streamlit', or 'script'."""
    if "streamlit" in sys.modules:
        return "streamlit"
    try:
        shell = get_ipython().__class__.__name__
        if shell in ("ZMQInteractiveShell", "Shell"):
            return "jupyter"
    except NameError:
        pass
    return "script"


# ── 2D Drawing ───────────────────────────────────────────────────────────────

def draw(input_data, *, highlight_bits=None, fp_type=None, legends=None,
         size=(300, 300), mols_per_row=3, column=None):
    """Draw 2D molecular structure(s).

    Works in Jupyter, Colab, and Streamlit (pass returned image to st.image()).

    Args:
        input_data: SMILES string, list of SMILES, RDKit Mol, pd.Series, or pd.DataFrame.
        highlight_bits: list of fingerprint bit indices to highlight (requires fp_type).
        fp_type: fingerprint type for bit highlighting (e.g. "ecfp4", "maccs").
        legends: list of labels for each molecule in grid mode.
        size: (width, height) per molecule in pixels.
        mols_per_row: molecules per row in grid mode.
        column: column name for DataFrame input.

    Returns:
        PIL.Image.Image — auto-displays in Jupyter; pass to st.image() in Streamlit.
    """
    data, is_batch = _extract_smiles(input_data, column=column)

    if not is_batch:
        mol = _ensure_mol(data)
        highlight_atoms = _get_highlight_atoms(mol, highlight_bits, fp_type)
        img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
        return img

    smiles_list = data
    mols = []
    for s in smiles_list:
        try:
            mols.append(_ensure_mol(s))
        except (ValueError, TypeError):
            mols.append(None)

    if legends is None:
        legends = [s if isinstance(s, str) else "" for s in smiles_list]

    if highlight_bits and fp_type:
        highlight_atom_lists = []
        for mol in mols:
            if mol is not None:
                highlight_atom_lists.append(
                    _get_highlight_atoms(mol, highlight_bits, fp_type)
                )
            else:
                highlight_atom_lists.append([])
    else:
        highlight_atom_lists = None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=size,
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
    )
    return img


def _get_highlight_atoms(mol, highlight_bits, fp_type):
    """Get atom indices to highlight for given fingerprint bits."""
    if not highlight_bits or not fp_type:
        return []

    from .explain import explain

    smiles = Chem.MolToSmiles(mol)
    bit_info = explain(smiles, fp_type)

    atoms = set()
    for bit in highlight_bits:
        if bit in bit_info:
            for atom_info in bit_info[bit]:
                if isinstance(atom_info, (list, tuple)):
                    for a in atom_info:
                        if isinstance(a, int):
                            atoms.add(a)
                elif isinstance(atom_info, int):
                    atoms.add(atom_info)
    return list(atoms)


# ── 3D Viewing ───────────────────────────────────────────────────────────────

def view3d(input_data, *, style="stick", size=(600, 400), spin=False,
           background="white"):
    """Interactive 3D molecular viewer.

    Auto-detects environment:
      - Jupyter/Colab: renders inline interactive widget
      - Streamlit: renders via st.components.v1.html()
      - Script: returns HTML string

    Requires: pip install py3Dmol

    Args:
        input_data: SMILES string or RDKit Mol.
        style: "stick", "sphere", "line", or "cartoon".
        size: (width, height) in pixels.
        spin: if True, auto-rotate the molecule.
        background: background color (default "white").

    Returns:
        In script mode: HTML string. In notebook/Streamlit: displays inline.
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D viewing. "
            "Install it with: pip install py3Dmol"
        )

    mol = _ensure_mol(input_data)
    mol = _ensure_conformer(mol)
    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mol_block, "mol")

    style_spec = _build_style(style)
    viewer.setStyle(style_spec)
    viewer.setBackgroundColor(background)
    if spin:
        viewer.spin(True)
    viewer.zoomTo()

    env = _detect_env()

    if env == "jupyter":
        return viewer.show()
    elif env == "streamlit":
        import streamlit.components.v1 as components
        html = _viewer_to_html(viewer, size)
        components.html(html, height=size[1] + 20, width=size[0] + 20)
    else:
        return _viewer_to_html(viewer, size)


def _build_style(style):
    """Convert style name to py3Dmol style dict."""
    styles = {
        "stick": {"stick": {}},
        "sphere": {"sphere": {"radius": 0.4}},
        "line": {"line": {}},
        "cartoon": {"cartoon": {}},
        "ball_and_stick": {"stick": {}, "sphere": {"radius": 0.3}},
    }
    if style in styles:
        return styles[style]
    return {"stick": {}}


def _viewer_to_html(viewer, size):
    """Extract standalone HTML from a py3Dmol viewer."""
    html = viewer._make_html()
    return html
