"""
SHAP-based feature interpretation with chemical meaning.

Connects SHAP feature importance to fingerprint bit annotations and
atom-level explanations. Answers: "Which molecular substructures drive
active vs inactive predictions, and what do they chemically mean?"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fingerprints import _batch, _ensure_mol, FINGERPRINTS
from .annotate import annotate, get_annotations
from .explain import explain as explain_bits
from .bioactivity import classify


# ── SMARTS → human-readable chemical meaning ─────────────────────────────────

_SMARTS_MEANINGS = {
    "[C;H3,H4]": "methyl or methane carbon",
    "[CH3]": "methyl group (-CH3)",
    "[#8]": "oxygen atom",
    "[#7]": "nitrogen atom",
    "[#6]": "carbon atom",
    "[#16]": "sulfur atom",
    "[#15]": "phosphorus atom",
    "[#9]": "fluorine atom",
    "c:n": "aromatic C-N bond (heterocyclic ring)",
    "c:c": "aromatic C-C bond",
    "c:o": "aromatic C-O bond (furan-type)",
    "c:s": "aromatic C-S bond (thiophene-type)",
    "[#6]#[#7]": "nitrile/cyano group (C≡N)",
    "[#6]=[#8]": "carbonyl (C=O)",
    "[#7]=[#6]": "imine (N=C)",
    "[#6]=[#6]": "alkene (C=C)",
    "[#6]=[#7]": "imine (C=N)",
    "[O;!H0]": "hydroxyl group (-OH)",
    "[!#6;!#1;!H0]": "heteroatom with hydrogen (NH, OH, SH)",
    "[F,Cl,Br,I]": "halogen atom",
    "[N;!H0]": "NH group (amine, amide)",
    "[S;!H0]": "thiol group (-SH)",
}


def _chemical_meaning(smarts_str):
    """Derive human-readable chemical meaning from a SMARTS pattern."""
    if not smarts_str:
        return ""

    # Direct lookup
    if smarts_str in _SMARTS_MEANINGS:
        return _SMARTS_MEANINGS[smarts_str]

    meaning_parts = []

    # Ring patterns
    if "1~*~*~*~*~*~1" in smarts_str or "1@" in smarts_str:
        if "[!#6;!#1]" in smarts_str:
            meaning_parts.append("heterocyclic ring")
        else:
            meaning_parts.append("ring system")

    # Chain patterns
    if "~[CH2]~" in smarts_str:
        ch2_count = smarts_str.count("[CH2]")
        if ch2_count >= 2:
            meaning_parts.append("alkyl chain (multiple -CH2-)")
        else:
            meaning_parts.append("methylene bridge (-CH2-)")

    # Functional groups from substrings
    if "[F,Cl,Br,I]" in smarts_str:
        if "!@*@*" in smarts_str:
            meaning_parts.append("halogen on ring-adjacent atom")
        elif "~*(~*)~*" in smarts_str:
            meaning_parts.append("halogen on branched atom")
        else:
            meaning_parts.append("halogen substituent")

    if "~[#8]" in smarts_str or "[#8]~" in smarts_str:
        meaning_parts.append("oxygen-containing")

    if "~[#7]" in smarts_str or "[#7]~" in smarts_str:
        meaning_parts.append("nitrogen-containing")

    if "[C;H3,H4]" in smarts_str or "[CH3]" in smarts_str:
        meaning_parts.append("methyl group")

    if "#" in smarts_str and "[#" not in smarts_str.replace("[#6]#[#7]", ""):
        meaning_parts.append("triple bond")

    if meaning_parts:
        return "; ".join(meaning_parts)

    # Fallback
    if "1" in smarts_str:
        return "ring-containing substructure"
    return "substructure pattern"


def interpret(smiles_list: list[str], activities: list[float] | np.ndarray, fp_type: str = "maccs", metric: str = "pchembl",
              thresholds: dict[str, float] | None = None, model: Any = None, top_n: int = 20, n_background: int = 50) -> dict[str, Any]:
    """Full interpretation pipeline: FP → model → SHAP → chemical meaning.

    Args:
        smiles_list: list of SMILES
        activities: list of activity values
        fp_type: fingerprint type
        metric: activity metric for classification ("pchembl", "pic50", etc.)
        thresholds: custom activity thresholds
        model: sklearn classifier (default: RandomForestClassifier)
        top_n: number of top features to return
        n_background: SHAP background samples

    Returns:
        dict with:
            shap_values: raw SHAP array (n_samples, n_bits) or per-class
            feature_importance: sorted list of (bit, mean_abs_shap, annotation)
            class_drivers: {class: [(bit, shap, annotation, direction), ...]}
            model: the trained model
            labels: activity class labels
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier

    fn, n_bits, _ = FINGERPRINTS[fp_type]
    X = _batch(fn, smiles_list)
    labels = classify(activities, metric=metric, thresholds=thresholds)
    label_set = sorted(set(labels))

    label_to_int = {l: i for i, l in enumerate(label_set)}
    y = np.array([label_to_int[l] for l in labels])

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    bg_idx = np.random.RandomState(42).choice(len(X), min(n_background, len(X)), replace=False)
    background = X[bg_idx]
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X)

    # Get annotations for this FP type
    annotations = get_annotations(fp_type)

    # Global feature importance (mean |SHAP| across all samples and classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    elif isinstance(shap_values, list):
        all_shap = np.stack(shap_values, axis=0)
        mean_abs = np.mean(np.abs(all_shap), axis=(0, 1))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    top_idx = [int(x) for x in np.argsort(mean_abs)[::-1][:top_n]]
    feature_importance = []
    for bit in top_idx:
        ann = annotations.get(bit, {"name": f"bit_{bit}", "description": "", "smarts": ""})
        feature_importance.append({
            "bit": bit,
            "mean_abs_shap": float(mean_abs[bit]),
            "name": ann.get("name", ""),
            "smarts": ann.get("smarts", ""),
            "description": ann.get("description", ""),
            "category": ann.get("category", ""),
            "chemical_meaning": _chemical_meaning(ann.get("smarts", "")),
        })

    # Per-class drivers: which bits push toward each class?
    # For each bit, compute SHAP when ON vs OFF to determine if
    # presence or absence of the substructure drives the prediction.
    class_drivers = {}
    for cls_idx, cls_name in enumerate(label_set):
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            cls_shap = shap_values[:, :, cls_idx]
        elif isinstance(shap_values, list):
            cls_shap = shap_values[cls_idx]
        else:
            continue

        mean_shap = np.mean(cls_shap, axis=0)
        top_pos = [int(x) for x in np.argsort(mean_shap)[::-1][:top_n]]
        top_neg = [int(x) for x in np.argsort(mean_shap)[:top_n]]

        drivers = []
        for bit in top_pos:
            if mean_shap[bit] <= 0:
                continue
            ann = annotations.get(bit, {"name": f"bit_{bit}"})

            # Conditional SHAP: when bit is ON vs OFF
            on_mask = X[:, bit] == 1
            off_mask = X[:, bit] == 0
            shap_when_on = float(np.mean(cls_shap[on_mask, bit])) if on_mask.any() else 0
            shap_when_off = float(np.mean(cls_shap[off_mask, bit])) if off_mask.any() else 0
            freq_in_class = float(np.mean(X[np.array(labels) == cls_name, bit])) if any(l == cls_name for l in labels) else 0
            freq_overall = float(np.mean(X[:, bit]))

            # Determine if presence or absence is favorable
            if shap_when_on > shap_when_off:
                value_effect = "present"
            else:
                value_effect = "absent"

            drivers.append({
                "bit": bit,
                "mean_shap": float(mean_shap[bit]),
                "direction": "promotes",
                "name": ann.get("name", ""),
                "smarts": ann.get("smarts", ""),
                "description": ann.get("description", ""),
                "chemical_meaning": _chemical_meaning(ann.get("smarts", "")),
                "shap_when_on": shap_when_on,
                "shap_when_off": shap_when_off,
                "value_effect": value_effect,
                "freq_in_class": freq_in_class,
                "freq_overall": freq_overall,
            })
        for bit in top_neg:
            if mean_shap[bit] >= 0:
                continue
            ann = annotations.get(bit, {"name": f"bit_{bit}"})

            on_mask = X[:, bit] == 1
            off_mask = X[:, bit] == 0
            shap_when_on = float(np.mean(cls_shap[on_mask, bit])) if on_mask.any() else 0
            shap_when_off = float(np.mean(cls_shap[off_mask, bit])) if off_mask.any() else 0
            freq_in_class = float(np.mean(X[np.array(labels) == cls_name, bit])) if any(l == cls_name for l in labels) else 0
            freq_overall = float(np.mean(X[:, bit]))

            if shap_when_on < shap_when_off:
                value_effect = "present"
            else:
                value_effect = "absent"

            drivers.append({
                "bit": bit,
                "mean_shap": float(mean_shap[bit]),
                "direction": "inhibits",
                "name": ann.get("name", ""),
                "smarts": ann.get("smarts", ""),
                "description": ann.get("description", ""),
                "chemical_meaning": _chemical_meaning(ann.get("smarts", "")),
                "shap_when_on": shap_when_on,
                "shap_when_off": shap_when_off,
                "value_effect": value_effect,
                "freq_in_class": freq_in_class,
                "freq_overall": freq_overall,
            })

        class_drivers[cls_name] = sorted(drivers, key=lambda d: abs(d["mean_shap"]), reverse=True)

    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "class_drivers": class_drivers,
        "model": model,
        "labels": labels,
        "label_set": label_set,
        "fp_type": fp_type,
        "n_bits": n_bits,
        "accuracy": float(model.score(X, y)),
    }


def interpret_molecule(smiles: str, interpretation_result: dict[str, Any]) -> dict[str, Any]:
    """Explain a single molecule's prediction using a previous interpret() result.

    Maps SHAP values to specific atoms in the molecule.

    Returns:
        dict with:
            prediction: predicted class
            prediction_proba: class probabilities
            top_bits: [(bit, shap, annotation, matching_atoms), ...]
    """
    model = interpretation_result["model"]
    fp_type = interpretation_result["fp_type"]
    label_set = interpretation_result["label_set"]

    fn = FINGERPRINTS[fp_type][0]
    fp = np.array([fn(smiles)], dtype=np.int32)

    pred_idx = model.predict(fp)[0]
    pred_proba = model.predict_proba(fp)[0]
    pred_class = label_set[pred_idx]

    # Explain which atoms set which bits
    bit_atoms = explain_bits(smiles, fp_type)
    annotations = get_annotations(fp_type)

    # Build per-bit explanation for ON bits
    fp_vec = fn(smiles)
    on_bits = [i for i, v in enumerate(fp_vec) if v > 0]

    # Get feature importances from the model for this molecule
    importances = interpretation_result["feature_importance"]
    important_bits = {f["bit"]: f for f in importances}

    top_bits = []
    for bit in on_bits:
        ann = annotations.get(bit, {"name": f"bit_{bit}"})
        atoms = bit_atoms.get(bit, [])
        importance = important_bits.get(bit, {}).get("mean_abs_shap", 0)

        top_bits.append({
            "bit": bit,
            "importance": importance,
            "name": ann.get("name", ""),
            "smarts": ann.get("smarts", ""),
            "description": ann.get("description", ""),
            "atoms": atoms,
        })

    top_bits.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "smiles": smiles,
        "prediction": pred_class,
        "prediction_proba": {label_set[i]: float(p) for i, p in enumerate(pred_proba)},
        "top_bits": top_bits[:20],
        "n_bits_on": len(on_bits),
    }


def compare_molecules(smiles_a: str, smiles_b: str, interpretation_result: dict[str, Any]) -> dict[str, Any]:
    """Compare two molecules: which bits differ and how important are they?

    Returns:
        dict with shared_important_bits, unique_to_a, unique_to_b,
        each annotated with chemical meaning and SHAP importance.
    """
    fp_type = interpretation_result["fp_type"]
    fn = FINGERPRINTS[fp_type][0]
    annotations = get_annotations(fp_type)
    importances = {f["bit"]: f for f in interpretation_result["feature_importance"]}

    fp_a = fn(smiles_a)
    fp_b = fn(smiles_b)

    on_a = set(i for i, v in enumerate(fp_a) if v > 0)
    on_b = set(i for i, v in enumerate(fp_b) if v > 0)

    shared = on_a & on_b
    only_a = on_a - on_b
    only_b = on_b - on_a

    def annotate_bits(bits):
        result = []
        for bit in bits:
            ann = annotations.get(bit, {"name": f"bit_{bit}"})
            imp = importances.get(bit, {}).get("mean_abs_shap", 0)
            result.append({
                "bit": bit,
                "importance": imp,
                "name": ann.get("name", ""),
                "smarts": ann.get("smarts", ""),
            })
        return sorted(result, key=lambda x: x["importance"], reverse=True)

    return {
        "smiles_a": smiles_a,
        "smiles_b": smiles_b,
        "shared_bits": len(shared),
        "unique_to_a": len(only_a),
        "unique_to_b": len(only_b),
        "tanimoto": len(shared) / len(on_a | on_b) if on_a | on_b else 0,
        "important_shared": annotate_bits(shared)[:10],
        "important_unique_a": annotate_bits(only_a)[:10],
        "important_unique_b": annotate_bits(only_b)[:10],
    }
