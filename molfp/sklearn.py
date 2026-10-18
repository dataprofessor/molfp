"""
scikit-learn compatible transformer classes for all fingerprint types.

Each transformer implements fit/transform/fit_transform and get_params/set_params,
making fingerprints usable in sklearn Pipelines, GridSearchCV, cross_val_score, etc.

For feature selection, use the standalone functions after transforming:
    X = molfp.ECFPTransformer().fit_transform(smiles)
    X_clean = molfp.remove_low_variance(X, threshold=0.01)["X"]
    result = molfp.select_features(X, y, k=500)
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import fingerprints as fp_mod


class _BaseFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Base class for all molfp sklearn transformers."""

    _fp_fn_name = None
    _default_n_bits = None

    def __init__(self, n_jobs: int = 1, sparse: bool = False) -> None:
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = getattr(fp_mod, self._fp_fn_name)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        prefix = self._fp_fn_name.upper()
        n = self._default_n_bits
        return np.array([f"{prefix}_{i}" for i in range(n)])


# ── Fixed-size fingerprint transformers ──────────────────────────────────────

class MACCSTransformer(_BaseFingerprintTransformer):
    """MACCS Keys (166 bits) sklearn transformer."""
    _fp_fn_name = "maccs"
    _default_n_bits = 166


class PubChemTransformer(_BaseFingerprintTransformer):
    """PubChem/CACTVS (881 bits) sklearn transformer."""
    _fp_fn_name = "pubchem"
    _default_n_bits = 881


class SubstructureTransformer(_BaseFingerprintTransformer):
    """SubstructureFP (307 bits) sklearn transformer."""
    _fp_fn_name = "substructure"
    _default_n_bits = 307


class SubstructureCountTransformer(_BaseFingerprintTransformer):
    """Substructure Count (307 features) sklearn transformer."""
    _fp_fn_name = "substructure_count"
    _default_n_bits = 307


class KlekotaRothTransformer(_BaseFingerprintTransformer):
    """Klekota-Roth (4860 bits) sklearn transformer."""
    _fp_fn_name = "klekota_roth"
    _default_n_bits = 4860


class KlekotaRothCountTransformer(_BaseFingerprintTransformer):
    """Klekota-Roth Count (4860 features) sklearn transformer."""
    _fp_fn_name = "klekota_roth_count"
    _default_n_bits = 4860


class EStateTransformer(_BaseFingerprintTransformer):
    """E-State (79 bits) sklearn transformer."""
    _fp_fn_name = "estate"
    _default_n_bits = 79


# ── Variable-size fingerprint transformers ───────────────────────────────────

class ECFPTransformer(BaseEstimator, TransformerMixin):
    """ECFP4 / Morgan (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._ecfp4_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ECFP4_{i}" for i in range(self.n_bits)])


class FCFPTransformer(BaseEstimator, TransformerMixin):
    """FCFP4 / Feature Morgan (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._fcfp4_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"FCFP4_{i}" for i in range(self.n_bits)])


class LingoTransformer(BaseEstimator, TransformerMixin):
    """Lingo (hashed SMILES 4-grams) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._lingo_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"Lingo_{i}" for i in range(self.n_bits)])


class ShortestPathTransformer(BaseEstimator, TransformerMixin):
    """Shortest Path sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._shortest_path_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ShortPath_{i}" for i in range(self.n_bits)])


class HybridizationTransformer(BaseEstimator, TransformerMixin):
    """Hybridization fingerprint sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._hybridization_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"Hybrid_{i}" for i in range(self.n_bits)])


class SignatureTransformer(BaseEstimator, TransformerMixin):
    """Signature fingerprint sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._signature_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"Sig_{i}" for i in range(self.n_bits)])


class AtomPairs2DTransformer(BaseEstimator, TransformerMixin):
    """AtomPairs2D (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._atompairs2d_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD2D_{i}" for i in range(self.n_bits)])


class AtomPairs2DCountTransformer(BaseEstimator, TransformerMixin):
    """AtomPairs2D Count (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._atompairs2d_count_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD2DC_{i}" for i in range(self.n_bits)])


class RDKTransformer(BaseEstimator, TransformerMixin):
    """RDKit path-based (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._rdk_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDK_{i}" for i in range(self.n_bits)])


class RDKExtTransformer(BaseEstimator, TransformerMixin):
    """RDKit Extended (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._rdk_ext_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDKEXT_{i}" for i in range(self.n_bits)])


class RDKGraphOnlyTransformer(BaseEstimator, TransformerMixin):
    """RDKit GraphOnly (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._rdk_graphonly_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDKGO_{i}" for i in range(self.n_bits)])


# ── 3D fingerprint transformers ──────────────────────────────────────────────

class AtomPairs3DTransformer(BaseEstimator, TransformerMixin):
    """3D Atom Pairs (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._atompairs3d_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD3D_{i}" for i in range(self.n_bits)])


class AtomPairs3DCountTransformer(BaseEstimator, TransformerMixin):
    """3D Atom Pairs Count (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = lambda s: fp_mod._atompairs3d_count_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD3DC_{i}" for i in range(self.n_bits)])


# ── Registry ──────────────────────────────────────────────────────────────────

TRANSFORMERS = {
    "maccs": MACCSTransformer,
    "ecfp4": ECFPTransformer,
    "fcfp4": FCFPTransformer,
    "pubchem": PubChemTransformer,
    "substructure": SubstructureTransformer,
    "substructure_count": SubstructureCountTransformer,
    "klekota_roth": KlekotaRothTransformer,
    "klekota_roth_count": KlekotaRothCountTransformer,
    "estate": EStateTransformer,
    "atompairs2d": AtomPairs2DTransformer,
    "atompairs2d_count": AtomPairs2DCountTransformer,
    "rdk": RDKTransformer,
    "rdk_ext": RDKExtTransformer,
    "rdk_graphonly": RDKGraphOnlyTransformer,
    "lingo": LingoTransformer,
    "shortest_path": ShortestPathTransformer,
    "hybridization": HybridizationTransformer,
    "signature": SignatureTransformer,
    "topological_torsion": type("TopologicalTorsionTransformer", (BaseEstimator, TransformerMixin), {
        "__init__": lambda self, n_bits=2048, n_jobs=1, sparse=False: [setattr(self, k, v) for k, v in {"n_bits": n_bits, "n_jobs": n_jobs, "sparse": sparse}.items()] and None,
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: fp_mod._batch(lambda s: fp_mod._topological_torsion_single(s, n_bits=self.n_bits), X, n_jobs=self.n_jobs, sparse=self.sparse),
        "get_feature_names_out": lambda self, input_features=None: np.array([f"TT_{i}" for i in range(self.n_bits)]),
        "__doc__": "Topological Torsion sklearn transformer.",
    }),
    "avalon": type("AvalonTransformer", (BaseEstimator, TransformerMixin), {
        "__init__": lambda self, n_bits=1024, n_jobs=1, sparse=False: [setattr(self, k, v) for k, v in {"n_bits": n_bits, "n_jobs": n_jobs, "sparse": sparse}.items()] and None,
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: fp_mod._batch(lambda s: fp_mod._avalon_single(s, n_bits=self.n_bits), X, n_jobs=self.n_jobs, sparse=self.sparse),
        "get_feature_names_out": lambda self, input_features=None: np.array([f"Avalon_{i}" for i in range(self.n_bits)]),
        "__doc__": "Avalon sklearn transformer.",
    }),
    "mhfp": type("MHFPTransformer", (BaseEstimator, TransformerMixin), {
        "__init__": lambda self, n_bits=1024, n_jobs=1, sparse=False: [setattr(self, k, v) for k, v in {"n_bits": n_bits, "n_jobs": n_jobs, "sparse": sparse}.items()] and None,
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: fp_mod._batch(lambda s: fp_mod._mhfp_single(s, n_bits=self.n_bits), X, n_jobs=self.n_jobs, sparse=self.sparse),
        "get_feature_names_out": lambda self, input_features=None: np.array([f"MHFP_{i}" for i in range(self.n_bits)]),
        "__doc__": "MHFP (MinHashed Fingerprint) sklearn transformer.",
    }),
    "map4": type("MAP4Transformer", (BaseEstimator, TransformerMixin), {
        "__init__": lambda self, n_bits=1024, n_jobs=1, sparse=False: [setattr(self, k, v) for k, v in {"n_bits": n_bits, "n_jobs": n_jobs, "sparse": sparse}.items()] and None,
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: fp_mod._batch(lambda s: fp_mod._map4_single(s, n_bits=self.n_bits), X, n_jobs=self.n_jobs, sparse=self.sparse),
        "get_feature_names_out": lambda self, input_features=None: np.array([f"MAP4_{i}" for i in range(self.n_bits)]),
        "__doc__": "MAP4 (MinHashed Atom Pair) sklearn transformer.",
    }),
    "map4c": type("MAP4CTransformer", (BaseEstimator, TransformerMixin), {
        "__init__": lambda self, n_bits=1024, n_jobs=1, sparse=False: [setattr(self, k, v) for k, v in {"n_bits": n_bits, "n_jobs": n_jobs, "sparse": sparse}.items()] and None,
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: fp_mod._batch(lambda s: fp_mod._map4c_single(s, n_bits=self.n_bits), X, n_jobs=self.n_jobs, sparse=self.sparse),
        "get_feature_names_out": lambda self, input_features=None: np.array([f"MAP4C_{i}" for i in range(self.n_bits)]),
        "__doc__": "MAP4C (MinHashed Atom Pair Chiral) sklearn transformer.",
    }),
    "atompairs3d": AtomPairs3DTransformer,
    "atompairs3d_count": AtomPairs3DCountTransformer,
}

# Add 3D descriptor transformers dynamically (fixed size, no configurable n_bits)
_3D_DESCRIPTOR_FPS = {
    "usr": ("_usr_single", 12, "USR"),
    "usrcat": ("_usrcat_single", 60, "USRCAT"),
    "autocorr3d": ("_autocorr3d_single", 80, "AUTOCORR3D"),
    "rdf": ("_rdf_single", 210, "RDF"),
    "morse": ("_morse_single", 224, "MORSE"),
    "whim": ("_whim_single", 114, "WHIM"),
    "getaway": ("_getaway_single", 273, "GETAWAY"),
}

for _name, (_fn_name, _n_bits, _prefix) in _3D_DESCRIPTOR_FPS.items():
    _cls_name = f"{_prefix}Transformer"
    _cls = type(_cls_name, (_BaseFingerprintTransformer,), {
        "_fp_fn_name": _fn_name,
        "_default_n_bits": _n_bits,
        "__doc__": f"{_prefix} ({_n_bits} features) sklearn transformer.",
    })
    globals()[_cls_name] = _cls
    TRANSFORMERS[_name] = _cls


def get_transformer(name: str, **kwargs: Any) -> BaseEstimator:
    """Get a sklearn transformer by fingerprint name.

    Example: molfp.get_transformer("ecfp4", n_bits=2048, n_jobs=-1)
    """
    if name not in TRANSFORMERS:
        raise KeyError(f"Unknown: {name}. Available: {list(TRANSFORMERS.keys())}")
    return TRANSFORMERS[name](**kwargs)
