from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

def chemical_system(struct) -> str:
    """Canonical unordered element set for a structure (e.g., 'Al-O')."""
    return "-".join(sorted({el.symbol for el in struct.composition.elements}))

def strat_bins(y: np.ndarray, q: int = 10) -> np.ndarray:
    """Quantile bins for stratification (handles ties/zeros)."""
    return pd.qcut(y, q=q, labels=False, duplicates="drop").astype(int)

def strat_group_train_val_test(
    clean: pd.DataFrame,
    test_frac: float = 0.20,
    seed_outer: int = 42,
    seed_inner: int = 7,
):
    """
    Two-stage split with leakage control + target balance:
    - Groups: chemical system (no element-set overlap across splits)
    - Stratify: band_gap_eV deciles
    """
    y = clean["band_gap_eV"].to_numpy()
    groups = clean["structure"].apply(chemical_system).to_numpy()
    bins = strat_bins(y, q=10)

    # Outer: pool vs test
    try:
        n_splits = int(round(1 / test_frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_outer)
        pool_idx, test_idx = next(sgkf.split(np.zeros_like(y), y=bins, groups=groups))
    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed_outer)
        pool_idx, test_idx = next(gss.split(np.zeros_like(y), groups=groups))

    pool = clean.iloc[pool_idx].reset_index(drop=True)
    test = clean.iloc[test_idx].reset_index(drop=True)

    # Inner: train vs val within pool
    y_pool = pool["band_gap_eV"].to_numpy()
    g_pool = pool["structure"].apply(chemical_system).to_numpy()
    b_pool = strat_bins(y_pool, q=10)

    try:
        sgkf_in = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_inner)
        tr_idx, va_idx = next(sgkf_in.split(np.zeros_like(y_pool), y=b_pool, groups=g_pool))
    except Exception:
        gss_in = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_inner)
        tr_idx, va_idx = next(gss_in.split(np.zeros_like(y_pool), groups=g_pool))

    train = pool.iloc[tr_idx].reset_index(drop=True)
    val   = pool.iloc[va_idx].reset_index(drop=True)
    return train, val, test

def decile_counts(y: np.ndarray, q: int = 10) -> dict:
    """Small helper to summarize decile balance."""
    dc = pd.qcut(y, q=q, labels=False, duplicates="drop")
    return pd.Series(dc).value_counts().sort_index().to_dict()