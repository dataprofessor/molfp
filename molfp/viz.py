"""
Molecule visualization for Jupyter, Colab, and Streamlit.

2D rendering uses RDKit Draw (no extra dependencies).
3D interactive viewing uses py3Dmol (optional: pip install py3Dmol).
Attribution maps use RDKit SimilarityMaps + matplotlib.
"""

from __future__ import annotations

import sys
from typing import Any, Union

import numpy as np
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


# ── Attribution Maps ──────────────────────────────────────────────────────────

def _bit_shap_to_atom_weights(mol, smiles, fp_type, shap_per_bit):
    """Aggregate bit-level SHAP values to per-atom weights via explain()."""
    from .explain import explain
    bit_atoms = explain(smiles, fp_type)
    n_atoms = mol.GetNumAtoms()
    weights = np.zeros(n_atoms)
    for bit, shap_val in shap_per_bit.items():
        if bit not in bit_atoms:
            continue
        for atom_info in bit_atoms[bit]:
            if isinstance(atom_info, (list, tuple)):
                atom_indices = [a for a in atom_info if isinstance(a, int)]
            elif isinstance(atom_info, int):
                atom_indices = [atom_info]
            else:
                continue
            if atom_indices:
                share = shap_val / len(atom_indices)
                for idx in atom_indices:
                    if 0 <= idx < n_atoms:
                        weights[idx] += share
    return weights


def draw_attribution(smiles, interpret_result, *, mode="dual",
                     class_active="active", class_inactive="inactive",
                     size=(400, 400), alpha=0.5, save=None):
    """Draw 2D structure colored by class-specific SHAP attribution.

    Args:
        smiles: SMILES string of the molecule to visualize.
        interpret_result: dict returned by molfp.interpret().
        mode: "active" (green), "inactive" (red), or "dual" (green + red).
        class_active: name of the active class in interpret_result.
        class_inactive: name of the inactive class in interpret_result.
        size: (width, height) in pixels.
        alpha: transparency of the heatmap overlay.
        save: file path to save the image (e.g. "attribution.png"). If None, image is not saved.

    Returns:
        PIL.Image.Image
    """
    mol = _ensure_mol(smiles)
    AllChem.Compute2DCoords(mol)
    fp_type = interpret_result["fp_type"]
    model = interpret_result["model"]
    label_set = interpret_result["label_set"]

    from .fingerprints import FINGERPRINTS
    fn = FINGERPRINTS[fp_type][0]
    fp = np.array([fn(smiles)], dtype=np.int32)

    # Get class indices
    active_idx = label_set.index(class_active) if class_active in label_set else 0
    inactive_idx = label_set.index(class_inactive) if class_inactive in label_set else -1

    # Compute SHAP for this molecule
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(fp)

    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        mol_shap_active = sv[0, :, active_idx]
        mol_shap_inactive = sv[0, :, inactive_idx] if inactive_idx >= 0 else -sv[0, :, active_idx]
    elif isinstance(sv, list):
        mol_shap_active = sv[active_idx][0]
        mol_shap_inactive = sv[inactive_idx][0] if inactive_idx >= 0 else -sv[active_idx][0]
    else:
        mol_shap_active = sv[0]
        mol_shap_inactive = -sv[0]

    # Build bit → SHAP dicts
    fp_vec = fn(smiles)
    on_bits = [i for i, v in enumerate(fp_vec) if v > 0]

    active_bit_shap = {b: float(mol_shap_active[b]) for b in on_bits}
    inactive_bit_shap = {b: float(mol_shap_inactive[b]) for b in on_bits}

    active_weights = _bit_shap_to_atom_weights(mol, smiles, fp_type, active_bit_shap)
    inactive_weights = _bit_shap_to_atom_weights(mol, smiles, fp_type, inactive_bit_shap)

    def _render(weights, color_pos, color_neg):
        max_w = max(abs(weights.max()), abs(weights.min()), 1e-8)
        norm = weights / max_w

        highlight_atoms = []
        highlight_colors = {}
        for i, w in enumerate(norm):
            if abs(w) < 0.05:
                continue
            highlight_atoms.append(i)
            if w > 0:
                intensity = min(abs(w), 1.0)
                r = color_pos[0] + (1 - color_pos[0]) * (1 - intensity)
                g = color_pos[1] + (1 - color_pos[1]) * (1 - intensity)
                b = color_pos[2] + (1 - color_pos[2]) * (1 - intensity)
            else:
                intensity = min(abs(w), 1.0)
                r = color_neg[0] + (1 - color_neg[0]) * (1 - intensity)
                g = color_neg[1] + (1 - color_neg[1]) * (1 - intensity)
                b = color_neg[2] + (1 - color_neg[2]) * (1 - intensity)
            highlight_colors[i] = (r, g, b)

        # Color bonds by averaging the weights of their two atoms
        highlight_bonds = []
        highlight_bond_colors = {}
        highlight_atom_set = set(highlight_atoms)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in highlight_atom_set or a2 in highlight_atom_set:
                bond_idx = bond.GetIdx()
                highlight_bonds.append(bond_idx)
                w1 = norm[a1] if a1 < len(norm) else 0
                w2 = norm[a2] if a2 < len(norm) else 0
                avg_w = (w1 + w2) / 2
                if avg_w > 0:
                    intensity = min(abs(avg_w), 1.0)
                    br = color_pos[0] + (1 - color_pos[0]) * (1 - intensity)
                    bg = color_pos[1] + (1 - color_pos[1]) * (1 - intensity)
                    bb = color_pos[2] + (1 - color_pos[2]) * (1 - intensity)
                else:
                    intensity = min(abs(avg_w), 1.0)
                    br = color_neg[0] + (1 - color_neg[0]) * (1 - intensity)
                    bg = color_neg[1] + (1 - color_neg[1]) * (1 - intensity)
                    bb = color_neg[2] + (1 - color_neg[2]) * (1 - intensity)
                highlight_bond_colors[bond_idx] = (br, bg, bb)

        from rdkit.Chem.Draw import rdMolDraw2D
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        except Exception:
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.highlightRadius = 0.5

        highlight_radii = {i: 0.5 for i in highlight_atoms}
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightAtomRadii=highlight_radii,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        from PIL import Image as PILImage
        from io import BytesIO
        return PILImage.open(BytesIO(png)).copy()

    green = (0.13, 0.55, 0.13)
    red = (0.8, 0.1, 0.1)

    if mode == "active":
        img = _render(np.clip(active_weights, 0, None), green, green)
    elif mode == "inactive":
        # Inactive SHAP values are typically negative (pushes away from active),
        # so we negate to get positive weights for atoms that drive inactivity
        img = _render(np.clip(-inactive_weights, 0, None), red, red)
    elif mode == "dual":
        # Combine: positive = active-driving (green), negative = inactive-driving (red)
        active_pos = np.clip(active_weights, 0, None)
        inactive_pos = np.clip(-inactive_weights, 0, None)
        # Merge into single weight vector: green where active dominates, red where inactive dominates
        dual_weights = active_pos - inactive_pos
        img = _render(dual_weights, green, red)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'active', 'inactive', or 'dual'.")

    if save:
        img.save(save)
    return img


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
