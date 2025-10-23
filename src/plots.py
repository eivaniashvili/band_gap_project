from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import SLIDES

def fig_bandgap_hist(df: pd.DataFrame, fname: str = "fig1_bandgap_hist.png"):
    """Overall band-gap histogram; saves into slides/."""
    bins = np.linspace(0, np.percentile(df["band_gap_eV"], 99.5), 120)
    plt.figure(figsize=(7,5))
    plt.hist(df["band_gap_eV"], bins=bins, alpha=0.8)
    plt.xlabel("Band gap (eV)")
    plt.ylabel("Count")
    plt.title("Distribution of Band Gaps")
    plt.tight_layout()
    plt.savefig(SLIDES / fname, dpi=200)
    plt.show()

def fig_bandgap_hist_logx(df: pd.DataFrame, fname: str = "fig1b_bandgap_hist_logx.png", save: bool = True):
    """
    Band-gap histogram with *log-scaled x-axis* (counts on linear scale).
    Zeros (metals) are handled separately and annotated.

    Parameters
    ----------
    df : DataFrame with column 'band_gap_eV'
    fname : output filename under SLIDES/
    save : if True, writes PNG; otherwise only displays
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    x = df["band_gap_eV"].to_numpy()
    zero_mask = x <= 0.0
    x_pos = x[~zero_mask]
    if x_pos.size == 0:
        raise ValueError("No positive band-gap values to plot on a log x-axis.")

    lo = max(np.nanmin(x_pos[x_pos > 0]), 1e-4)
    hi = np.nanpercentile(x_pos, 99.5)
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(x_pos, bins=bins, alpha=0.85, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Band gap (eV, log scale)")
    ax.set_ylabel("Count")

    zero_pct = 100.0 * zero_mask.mean()
    ax.set_title(f"Distribution of Band Gaps (log x) — zero-gap (metals): {zero_pct:.1f}%")

    # Add an annotation box for the zero bin
    ax.annotate(
        f"0 eV bin: {zero_mask.sum():,} ({zero_pct:.1f}%)",
        xy=(lo, ax.get_ylim()[1]*0.9), xycoords=("data", "data"),
        xytext=(0.02, 0.95), textcoords="axes fraction",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9)
    )

    plt.tight_layout()
    if save:
        (SLIDES / Path(fname)).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SLIDES / fname, dpi=200, bbox_inches="tight")
    plt.show()

def fig_metal_fraction_by_nelems(df: pd.DataFrame, fname: str = "fig2_metal_fraction_by_nelems.png"):
    """Metal fraction vs number of distinct elements in composition."""
    def n_elems(s): return len(s.composition.elements)
    tmp = df[["structure","band_gap_eV"]].copy()
    tmp["n_elements"] = tmp["structure"].apply(n_elems)
    tmp["is_metal"] = (tmp["band_gap_eV"] <= 1e-9)
    metal_frac = tmp.groupby("n_elements")["is_metal"].mean().rename("metal_fraction").reset_index()

    plt.figure(figsize=(7,5))
    plt.bar(metal_frac["n_elements"], metal_frac["metal_fraction"])
    plt.xticks(metal_frac["n_elements"])
    plt.ylim(0,1)
    plt.xlabel("# distinct elements")
    plt.ylabel("Fraction metallic (gap ≈ 0)")
    plt.title("Metallicity vs. Composition Complexity")
    plt.tight_layout()
    plt.savefig(SLIDES / fname, dpi=200)
    plt.show()

def fig_oxygen_violin(df: pd.DataFrame, fname: str = "fig3_oxygen_violin.png"):
    """Band-gap distributions for O-containing vs O-free compounds."""
    def has_O(s): return any(el.symbol == "O" for el in s.composition.elements)
    t = df[["structure","band_gap_eV"]].copy()
    t["has_O"] = t["structure"].apply(has_O)
    groups = [t.loc[t["has_O"], "band_gap_eV"], t.loc[~t["has_O"], "band_gap_eV"]]

    plt.figure(figsize=(7,5))
    plt.violinplot(groups, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks([1,2], ["Contains O", "No O"])
    plt.ylabel("Band gap (eV)")
    plt.title("Effect of Oxygen on Band Gap")
    ymax = min(6, t["band_gap_eV"].max())
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(SLIDES / fname, dpi=200)
    plt.show()

def fig_bandgap_summary(df, fname="fig0_bandgap_summary.png", save=True):
    """Compact panel with histogram + pie chart summarizing dataset."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    
    # Histogram
    bins = np.linspace(0, np.percentile(df["band_gap_eV"], 99.5), 60)
    ax[0].hist(df["band_gap_eV"], bins=bins, color="steelblue", alpha=0.8)
    ax[0].set_xlabel("Band gap (eV)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Band Gap Distribution")

    # Pie chart
    metals = (df["band_gap_eV"] < 0.05).mean()
    semis = ((df["band_gap_eV"] >= 0.05) & (df["band_gap_eV"] < 3)).mean()
    insul = (df["band_gap_eV"] >= 3).mean()
    ax[1].pie(
        [metals, semis, insul],
        labels=["Metals", "Semiconductors", "Insulators"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4477AA", "#66CCEE", "#DDCC77"]
    )
    ax[1].set_title("Electronic Regime Fractions")

    plt.tight_layout()
    if save:
        plt.savefig(SLIDES / fname, dpi=200)
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_bandgap_summary_table(
    df: pd.DataFrame,
    metal_thr: float = 0.05,
    semi_hi: float = 3.0,
    save_csv: bool = False,
    save_png: bool = False,
    csv_name: str = "eda_bandgap_summary.csv",
    png_name: str = "eda_bandgap_summary.png",
    dpi: int = 220,
):
    """
    Build a summary table for EDA Highlights.
    - df must contain columns: ['structure', 'band_gap_eV']
    - Returns a pandas DataFrame with the metrics discussed.
    - Optionally saves a CSV and a PNG figure of the table.
    """
    y = df["band_gap_eV"].astype(float).values

    # basic stats
    n = len(y)
    stats = {
        "n_rows": n,
        "mean_eV": float(np.mean(y)),
        "median_eV": float(np.median(y)),
        "std_eV": float(np.std(y, ddof=1)),
        "min_eV": float(np.min(y)),
        "q25_eV": float(np.percentile(y, 25)),
        "q75_eV": float(np.percentile(y, 75)),
        "max_eV": float(np.max(y)),
        "iqr_eV": float(np.percentile(y, 75) - np.percentile(y, 25)),
        "skewness": float(pd.Series(y).skew()),
        "kurtosis": float(pd.Series(y).kurt()),  # Fisher (0 = normal)
    }

    metals = (y < metal_thr).mean()
    semis  = ((y >= metal_thr) & (y < semi_hi)).mean()
    insul  = (y >= semi_hi).mean()

    n_elems = df["structure"].apply(lambda s: len(s.composition.elements)).to_numpy()
    contains_O = df["structure"].apply(
        lambda s: any(el.symbol == "O" for el in s.composition.elements)
    ).mean()

    comp_stats = {
        "median_num_elements": float(np.median(n_elems)),
        "mean_num_elements": float(np.mean(n_elems)),
        "share_contains_O": float(contains_O),
    }

    table = pd.DataFrame({
        "Metric": [
            "Rows",
            "Mean (eV)", "Median (eV)", "Std (eV)",
            "Min (eV)", "Max (eV)", 
            "Metals (<0.05 eV)", "Semiconductors (0.05–3 eV)", "Insulators (≥3 eV)",
            "Median # elements", "Mean # elements", "Share with O",
        ],
        "Value": [
            stats["n_rows"],
            stats["mean_eV"], stats["median_eV"], stats["std_eV"],
            stats["min_eV"], stats["max_eV"],
            f"{metals*100:.2f}%", f"{semis*100:.2f}%", f"{insul*100:.2f}%",
            comp_stats["median_num_elements"], comp_stats["mean_num_elements"],
            f"{comp_stats['share_contains_O']*100:.2f}%",
        ],
    })


    if save_png:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.axis("off")
        tbl = ax.table(
            cellText=table.values,
            colLabels=table.columns.tolist(),
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.2)
        plt.tight_layout()
        (SLIDES / png_name).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(SLIDES / png_name, dpi=dpi, bbox_inches="tight")
        plt.show()

    return table
