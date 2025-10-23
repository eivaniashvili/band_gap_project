from __future__ import annotations
from typing import Iterable, Optional, List
import pandas as pd
from matminer.featurizers.composition import ElementFraction, ElementProperty

REQ_COLS = ("structure", "band_gap_eV")

def _check_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame must contain columns {REQ_COLS}; missing={missing}")

def add_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a pymatgen Composition object as a new column.
    Keeps only the required columns to reduce memory.
    """
    _check_input(df)
    out = df[list(REQ_COLS)].copy()
    out["composition"] = out["structure"].apply(lambda s: s.composition)
    return out

def featurize_element_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fractional presence of each element (one column per element).
    Leaves 'structure', 'band_gap_eV', and 'composition' intact.
    """
    ef = ElementFraction()
    return ef.featurize_dataframe(df.copy(), col_id="composition",
                                  ignore_errors=True, inplace=False)

def featurize_magpie(df: pd.DataFrame,
                     props: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Compute Magpie elemental-property statistics (mean, max, min, range, etc.).
    If 'props' is provided, it must be a subset of the preset; otherwise use full preset.
    """
    if props is None:
        ep = ElementProperty.from_preset("magpie")
    else:
        ep = ElementProperty.from_preset("magpie", features=list(props))
    return ep.featurize_dataframe(df.copy(), col_id="composition",
                                  ignore_errors=True, inplace=False)

def make_feature_matrix(df: pd.DataFrame,
                        use_element_fraction: bool = True,
                        use_magpie: bool = True,
                        magpie_props: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Composition-based featurization pipeline.
    Returns a DataFrame with columns:
      ['structure', 'band_gap_eV', <frac_*>..., <MagpieData*>...]
    May contain NaNs (to be handled AFTER the split via imputation).

    Parameters
    ----------
    df : DataFrame with ['structure', 'band_gap_eV'] at minimum.
    use_element_fraction : include ElementFraction features.
    use_magpie : include Magpie features.
    magpie_props : optional subset of Magpie properties.
    """
    if not (use_element_fraction or use_magpie):
        raise ValueError("At least one of {use_element_fraction, use_magpie} must be True.")


    dfc = add_composition(df)


    feats: List[pd.DataFrame] = [dfc]
    if use_element_fraction:
        feats.append(featurize_element_fraction(dfc))
    if use_magpie:
        feats.append(featurize_magpie(dfc, props=magpie_props))


    out = pd.concat(feats, axis=1)


    out = out.loc[:, ~out.columns.duplicated()].copy()


    if "composition" in out.columns:
        out = out.drop(columns=["composition"])


    feature_cols = [c for c in out.columns
                    if c not in ("structure", "band_gap_eV")]
    out = out[["structure", "band_gap_eV"] + feature_cols]

    return out.reset_index(drop=True)