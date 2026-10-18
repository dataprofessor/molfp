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


def draw_attribution(smiles, interpret_result, *,
                     class_active="active", class_inactive="inactive",
                     size=(400, 400), alpha=0.5, style="highlight", save=None):
    """Draw 2D structure colored by class-specific SHAP attribution.

    Green regions drive the active prediction, red regions drive inactive.
    Intensity is proportional to SHAP magnitude. Neutral atoms are uncolored.

    Args:
        smiles: SMILES string of the molecule to visualize.
        interpret_result: dict returned by molfp.interpret().
        class_active: name of the active class in interpret_result.
        class_inactive: name of the inactive class in interpret_result.
        size: (width, height) in pixels.
        alpha: transparency of the heatmap overlay.
        style: "highlight" (solid atom/bond circles) or "contour" (gaussian contour map).
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

    # ── Contour style (SimilarityMaps gaussian) ──────────────────────────
    def _render_contour(weights, cmap):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rdkit.Chem.Draw import SimilarityMaps
        from PIL import Image as PILImage
        from io import BytesIO

        max_w = max(abs(weights.max()), abs(weights.min()), 1e-8)
        normalized = list(weights / max_w)
        fig = SimilarityMaps.GetSimilarityMapFromWeights(
            mol, normalized, colorMap=cmap, alpha=alpha, size=size
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf).copy()

    # ── Highlight style (smooth gaussian blend) ────────────────────────────
    def _render_highlight(weights, color_pos, color_neg):
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image as PILImage, ImageFilter, ImageDraw
        from io import BytesIO

        max_w = max(abs(weights.max()), abs(weights.min()), 1e-8)
        norm = weights / max_w

        # Draw plain molecule and get pixel coordinates
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        except Exception:
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Get pixel coordinates from the drawer
        atom_coords = []
        for i in range(mol.GetNumAtoms()):
            pt = drawer.GetDrawCoords(i)
            atom_coords.append((int(pt.x), int(pt.y)))

        png = drawer.GetDrawingText()
        mol_img = PILImage.open(BytesIO(png)).convert("RGBA").copy()

        # Make white background transparent so heatmap shows through
        data = mol_img.getdata()
        new_data = []
        for item in data:
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        mol_img.putdata(new_data)

        # Paint gaussian blobs at each atom's pixel position
        heatmap = PILImage.new("RGBA", (size[0], size[1]), (255, 255, 255, 0))
        radius = int(size[0] * 0.10)

        for i, w in enumerate(norm):
            if abs(w) < 0.05:
                continue
            px, py = atom_coords[i]
            color = color_pos if w > 0 else color_neg
            intensity = min(abs(w), 1.0)
            r_val = int(color[0] * 255)
            g_val = int(color[1] * 255)
            b_val = int(color[2] * 255)
            a_val = int(intensity * 255)

            blob = PILImage.new("RGBA", (size[0], size[1]), (0, 0, 0, 0))
            draw_ctx = ImageDraw.Draw(blob)
            draw_ctx.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=(r_val, g_val, b_val, a_val)
            )
            heatmap = PILImage.alpha_composite(heatmap, blob)

        # Blur for smooth blending
        blur_radius = int(size[0] * 0.05)
        r, g, b, a = heatmap.split()
        r = r.filter(ImageFilter.GaussianBlur(blur_radius))
        g = g.filter(ImageFilter.GaussianBlur(blur_radius))
        b = b.filter(ImageFilter.GaussianBlur(blur_radius))
        a = a.filter(ImageFilter.GaussianBlur(blur_radius))
        heatmap = PILImage.merge("RGBA", (r, g, b, a))

        # Composite: white bg → heatmap → molecule
        result = PILImage.alpha_composite(
            PILImage.new("RGBA", size, (255, 255, 255, 255)),
            heatmap
        )
        result = PILImage.alpha_composite(result, mol_img)
        return result.convert("RGB")

    green = (0.13, 0.55, 0.13)
    red = (0.8, 0.1, 0.1)

    active_pos = np.clip(active_weights, 0, None)
    inactive_pos = np.clip(-inactive_weights, 0, None)
    dual_weights = active_pos - inactive_pos

    if style == "contour":
        from matplotlib.colors import LinearSegmentedColormap
        cmap_dual = LinearSegmentedColormap.from_list("dual", [red, (1, 1, 1), green])
        img = _render_contour(dual_weights, cmap_dual)
    elif style == "highlight":
        img = _render_highlight(dual_weights, green, red)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'highlight' or 'contour'.")

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
