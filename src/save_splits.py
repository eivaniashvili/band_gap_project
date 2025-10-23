from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def _to_csv(df: pd.DataFrame, feature_cols: list[str], path: Path) -> None:
    out = df[feature_cols].copy()
    out.insert(0, "band_gap_eV", df["band_gap_eV"].values)
    out.to_csv(path, index=False)

def _records_with_structure(df: pd.DataFrame, chem_sys_fn) -> list[dict]:
    recs = []
    for s, y in zip(df["structure"].values, df["band_gap_eV"].values):
        recs.append({
            "band_gap_eV": float(y),
            "chemical_system": chem_sys_fn(s),
            "structure": s.as_dict(),         
        })
    return recs

def save_splits_csv_json(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    outdir: Path,
    chem_sys_fn,  
) -> dict:
    """
    Save CSVs (features+target) and JSONs (structure+target) for each split,
    plus a manifest. Returns the manifest dict.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    _to_csv(train, feature_cols, outdir / "train.csv")
    _to_csv(val,   feature_cols, outdir / "val.csv")
    _to_csv(test,  feature_cols, outdir / "test.csv")

    (outdir / "train_with_structure.json").write_text(
        json.dumps(_records_with_structure(train, chem_sys_fn), indent=2)
    )
    (outdir / "val_with_structure.json").write_text(
        json.dumps(_records_with_structure(val,   chem_sys_fn), indent=2)
    )
    (outdir / "test_with_structure.json").write_text(
        json.dumps(_records_with_structure(test,  chem_sys_fn), indent=2)
    )

    manifest = {
        "n_features": len(feature_cols),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "csv_files": ["train.csv", "val.csv", "test.csv"],
        "json_files": [
            "train_with_structure.json",
            "val_with_structure.json",
            "test_with_structure.json",
        ],
    }
    (outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest