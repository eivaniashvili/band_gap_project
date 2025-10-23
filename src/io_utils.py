from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import ART

def save_np(name: str, arr):
    """Save a numpy array to artifacts/ as .npy."""
    np.save(ART / name, arr)

def load_np(name: str):
    """Load a numpy array from artifacts/."""
    return np.load(ART / name)

def save_df_pickle(name: str, df: pd.DataFrame):
    """Pickle a DataFrame into artifacts/."""
    df.to_pickle(ART / name)

def save_json(name: str, obj):
    """Write JSON into artifacts/ with pretty formatting."""
    (ART / name).write_text(json.dumps(obj, indent=2))

def load_df_pickle(name: str) -> pd.DataFrame:
    """Load a pickled DataFrame from artifacts/."""
    path = ART / name
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    return pd.read_pickle(path)