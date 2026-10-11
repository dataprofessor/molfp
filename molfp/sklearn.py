"""
scikit-learn compatible transformer classes for all fingerprint types.

Each transformer implements fit/transform/fit_transform and get_params/set_params,
making fingerprints usable in sklearn Pipelines, GridSearchCV, cross_val_score, etc.

All transformers support optional feature selection:
  - variance_threshold: remove features with variance below this value
  - select_k: keep the top-k features by mutual information (requires y in fit)
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import fingerprints as fp_mod


# ── Feature selection mixin ──────────────────────────────────────────────────

class _FeatureSelectionMixin:
    """Adds variance_threshold and select_k feature selection to transformers."""

    def _init_selection(self, variance_threshold=None, select_k=None):
        self.variance_threshold = variance_threshold
        self.select_k = select_k
        self._vt = None
        self._mi = None

    def _fit_selection(self, X, y=None):
        if self.variance_threshold is not None:
            from sklearn.feature_selection import VarianceThreshold
            self._vt = VarianceThreshold(threshold=self.variance_threshold)
            X = self._vt.fit_transform(X)

        if self.select_k is not None and y is not None:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            k = min(self.select_k, X.shape[1])
            self._mi = SelectKBest(mutual_info_classif, k=k)
            self._mi.fit(X, y)

    def _apply_selection(self, X):
        if self._vt is not None:
            X = self._vt.transform(X)
        if self._mi is not None:
            X = self._mi.transform(X)
        return X


# ── Base transformer ─────────────────────────────────────────────────────────

class _BaseFingerprintTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """Base class for all molfp sklearn transformers."""

    _fp_fn_name = None
    _default_n_bits = None

    def __init__(self, n_jobs: int = 1, sparse: bool = False,
                 variance_threshold=None, select_k=None) -> None:
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            fn = getattr(fp_mod, self._fp_fn_name)
            X_raw = fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)
            self._fit_selection(X_raw, y)
        return self

    def transform(self, X):
        fn = getattr(fp_mod, self._fp_fn_name)
        X_raw = fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False if (self._vt or self._mi) else self.sparse)
        return self._apply_selection(X_raw)

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

class ECFPTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """ECFP4 / Morgan (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._ecfp4_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ECFP4_{i}" for i in range(self.n_bits)])


class AtomPairs2DTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """AtomPairs2D (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._atompairs2d_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD2D_{i}" for i in range(self.n_bits)])


class AtomPairs2DCountTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """AtomPairs2D Count (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._atompairs2d_count_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD2DC_{i}" for i in range(self.n_bits)])


class RDKTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """RDKit path-based (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._rdk_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDK_{i}" for i in range(self.n_bits)])


class RDKExtTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """RDKit Extended (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._rdk_ext_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDKEXT_{i}" for i in range(self.n_bits)])


class RDKGraphOnlyTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """RDKit GraphOnly (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=1024, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._rdk_graphonly_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"RDKGO_{i}" for i in range(self.n_bits)])


# ── 3D fingerprint transformers ──────────────────────────────────────────────

class AtomPairs3DTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """3D Atom Pairs (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._atompairs3d_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD3D_{i}" for i in range(self.n_bits)])


class AtomPairs3DCountTransformer(BaseEstimator, TransformerMixin, _FeatureSelectionMixin):
    """3D Atom Pairs Count (configurable bits) sklearn transformer."""

    def __init__(self, n_bits=780, n_jobs=1, sparse=False,
                 variance_threshold=None, select_k=None):
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse
        self._init_selection(variance_threshold, select_k)

    def _compute_raw(self, X):
        fn = lambda s: fp_mod._atompairs3d_count_single(s, n_bits=self.n_bits)
        return fp_mod._batch(fn, X, n_jobs=self.n_jobs, sparse=False)

    def fit(self, X, y=None):
        if self.variance_threshold is not None or (self.select_k is not None and y is not None):
            self._fit_selection(self._compute_raw(X), y)
        return self

    def transform(self, X):
        X_raw = self._compute_raw(X)
        return self._apply_selection(X_raw)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AD3DC_{i}" for i in range(self.n_bits)])


# ── Registry ──────────────────────────────────────────────────────────────────

TRANSFORMERS = {
    "maccs": MACCSTransformer,
    "ecfp4": ECFPTransformer,
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
