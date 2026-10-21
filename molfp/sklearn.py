"""
scikit-learn compatible transformer for all fingerprint types.

A single FingerprintTransformer wraps any fingerprint registered in
FINGERPRINTS, making all 40+ types usable in sklearn Pipelines,
GridSearchCV, cross_val_score, etc.

Usage:
    import molfp

    # Basic
    t = molfp.FingerprintTransformer("ecfp4")
    X = t.fit_transform(smiles)

    # With custom bit size
    t = molfp.FingerprintTransformer("ecfp4", n_bits=2048)

    # In a Pipeline with GridSearchCV over fp_type
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    pipe = Pipeline([
        ("fp", molfp.FingerprintTransformer("ecfp4")),
        ("clf", RandomForestClassifier()),
    ])
    search = GridSearchCV(pipe, {
        "fp__fp_type": ["ecfp4", "rdk", "maccs", "map4"],
        "fp__n_bits": [512, 1024, 2048],
    }, cv=5)
    search.fit(smiles, y)
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import fingerprints as fp_mod


def _accepts_n_bits(fn) -> bool:
    """Check if a fingerprint single function accepts an n_bits parameter."""
    return "n_bits" in inspect.signature(fn).parameters


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """Universal sklearn transformer for any molfp fingerprint.

    Parameters
    ----------
    fp_type : str
        Fingerprint name (must exist in ``molfp.FINGERPRINTS``).
    n_bits : int or None
        Bit size override. Ignored for fixed-size fingerprints (e.g. maccs).
        When None, uses the default from the FINGERPRINTS registry.
    n_jobs : int
        Number of parallel workers (passed to joblib). -1 uses all cores.
    sparse : bool
        If True, return a scipy sparse matrix instead of a dense array.
    """

    def __init__(self, fp_type: str = "ecfp4", n_bits: int | None = None,
                 n_jobs: int = 1, sparse: bool = False) -> None:
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn, default_n_bits, _ = fp_mod.FINGERPRINTS[self.fp_type]
        if _accepts_n_bits(fn):
            n_bits = self.n_bits if self.n_bits is not None else default_n_bits
            wrapped = lambda s, _fn=fn, _nb=n_bits: _fn(s, n_bits=_nb)
        else:
            wrapped = fn
        return fp_mod._batch(wrapped, X, n_jobs=self.n_jobs, sparse=self.sparse)

    def get_feature_names_out(self, input_features=None):
        fn, default_n_bits, _ = fp_mod.FINGERPRINTS[self.fp_type]
        if _accepts_n_bits(fn) and self.n_bits is not None:
            n = self.n_bits
        else:
            n = default_n_bits
        prefix = self.fp_type.upper()
        return np.array([f"{prefix}_{i}" for i in range(n)])


# ── Registry ──────────────────────────────────────────────────────────────────

TRANSFORMERS: dict[str, type] = {}


def _rebuild_transformers():
    """Rebuild TRANSFORMERS dict from current FINGERPRINTS registry."""
    TRANSFORMERS.clear()
    for name in fp_mod.FINGERPRINTS:
        TRANSFORMERS[name] = FingerprintTransformer


_rebuild_transformers()


def get_transformer(name: str, **kwargs: Any) -> FingerprintTransformer:
    """Get a sklearn transformer by fingerprint name.

    Example: molfp.get_transformer("ecfp4", n_bits=2048, n_jobs=-1)
    """
    if name not in fp_mod.FINGERPRINTS:
        raise KeyError(f"Unknown: {name}. Available: {list(fp_mod.FINGERPRINTS.keys())}")
    return FingerprintTransformer(fp_type=name, **kwargs)
