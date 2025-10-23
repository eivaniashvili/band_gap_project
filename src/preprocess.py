# src/preprocess.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd



def _family_of(col: str) -> str:
    if col.startswith(("MagpieData", "magpie")): return "magpie"
    if col.startswith(("frac ", "frac_")):       return "ef"
    # bare element symbols (H, He, Li, ...)
    if len(col) in (1,2) and col[:1].isupper() and col[1:2].islower() or (len(col)==1 and col.isupper()):
        return "ef"
    return "other"

def report_missingness(
    X_by_split: Dict[str, pd.DataFrame],
    *,
    per_feature_on: str = "train",
    top_k: int = 20,
) -> Dict[str, dict]:
    """
    Build a comprehensive missingness report WITHOUT modifying data.
    - Per split: n_rows, n_features, frac_rows_with_any_missing
    - On `per_feature_on` split: per-feature missing rate, family summaries, top-K list
    """
    out: Dict[str, dict] = {}
    for split, X in X_by_split.items():
        miss = X.isna()
        entry = {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "frac_rows_with_any_missing": float(miss.any(axis=1).mean()),
        }
        if split == per_feature_on:
            rates = miss.mean()
            entry["per_feature_missing_rate"] = rates.to_dict()
            fam = pd.Series({_c: _family_of(_c) for _c in X.columns})
            fam_summ = (
                pd.DataFrame({"family": fam, "miss": rates})
                .groupby("family")
                .agg(n_features=("miss","size"),
                     n_missing_features=("miss", lambda s: int((s>0).sum())),
                     mean_missing_rate=("miss","mean"))
                .to_dict(orient="index")
            )
            entry["family_summary"] = fam_summ
            top = rates.sort_values(ascending=False).head(top_k)
            entry["top_missing_features"] = [{ "feature": k, "missing_rate": float(v) } for k,v in top.items()]
        out[split] = entry
    return out



@dataclass
class NanAwareScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: List[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = X.values.astype("float64", copy=True)
        # Subtract mean, divide by std; NaNs remain NaN
        A -= self.mean_
        # Avoid divide-by-zero: where scale==0, keep values as 0 shift (no scale)
        safe_scale = self.scale_.copy()
        safe_scale[safe_scale == 0.0] = 1.0
        A /= safe_scale
        return A

def fit_nanaware_scaler(X_train: pd.DataFrame) -> NanAwareScaler:
    A = X_train.values.astype("float64", copy=False)
    mean = np.nanmean(A, axis=0)
    std  = np.nanstd(A, axis=0, ddof=0)
    return NanAwareScaler(mean_=mean, scale_=std, feature_names=list(X_train.columns))

def scale_only(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], NanAwareScaler]:
    """
    Compute train-only mean/std with NaN-robust stats and scale all splits, preserving NaNs.
    Returns scaled arrays, feature names (unchanged), and the fitted scaler.
    """
    scaler = fit_nanaware_scaler(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    return Xtr, Xva, Xte, scaler.feature_names, scaler