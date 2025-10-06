# analytics/common.py
from __future__ import annotations
import os
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ----------- IO & paths -----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
OUT_DIR = os.path.join(DATA_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
STATS_DIR = os.path.join(OUT_DIR, "stats")

def ensure_out_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

def load_enhanced() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "breast_cancer_trials_enhanced.parquet")
    return pd.read_parquet(path)

def save_fig(fig: mpl.figure.Figure, filename: str, dpi: int = 300):
    ensure_out_dirs()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_stats(text: str, filename: str):
    ensure_out_dirs()
    with open(os.path.join(STATS_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

# ----------- Plot style (consistent across all) -----------
# Neutral palette with high contrast.
PALETTE = {
    "primary": "#1f77b4",   # blue
    "secondary": "#ff7f0e", # orange
    "tertiary": "#2ca02c",  # green
    "quaternary": "#d62728",# red
    "quinary": "#9467bd",   # purple
    "senary": "#8c564b",    # brown
    "gray": "#7f7f7f",
}

mpl.rcParams.update({
    "figure.figsize": (9, 5.5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.size": 11,
})

def bars(ax, x, y, color="primary", label=None, rotate=False):
    ax.bar(x, y, color=PALETTE[color], label=label, edgecolor="none")
    if rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

def line(ax, x, y, color="primary", label=None, marker="o"):
    ax.plot(x, y, color=PALETTE[color], marker=marker, linewidth=2, label=label)

def pct(n, d) -> float:
    if d == 0: return 0.0
    return 100.0 * n / d

# ----------- Common derived fields -----------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # has_posted_results: already present? if not, derive from results_outcome_measures
    if "has_results" not in out.columns:
        out["has_results"] = out["results_outcome_measures"].notna().astype(int)

    # years_since_completion
    def _to_year(d):
        try:
            return pd.to_datetime(d, errors="coerce").year
        except Exception:
            return np.nan
    out["completion_year"] = out["completion_date"].apply(_to_year)
    out["years_since_completion"] = pd.Timestamp.now().year - out["completion_year"]
    out.loc[out["completion_year"].isna(), "years_since_completion"] = np.nan

    # multi_country flag
    out["multi_country_flag"] = out["site_countries"].fillna("").str.contains(";").astype(int)

    # phase buckets (normalize)
    def _phase_bucket(s):
        s = str(s) if pd.notna(s) else ""
        if "PHASE3" in s: return "Phase 3"
        if "PHASE2" in s: return "Phase 2"
        if "PHASE1" in s: return "Phase 1"
        if "NA" in s or s == "": return "NA/Other"
        return "Other"
    out["phase_bucket"] = out["phase"].apply(_phase_bucket)

    return out

# Endpoint columns (15)
ENDPOINT_COLS = [
    "Pathologic_Complete_Response",
    "Event_Free_Survival",
    "Disease_Free_Survival",
    "Overall_Survival",
    "Best_Overall_Response",
    "Duration_of_Response",
    "Progression_Free_Survival",
    "Overall_Response",
    "Target_Response",
    "Non_target_Response",
    "Symptomatic_Deterioration",
    "Disease_Recurrence",
    "Pathologic_Complete_Response_pCR",
    "Objective_Response_Rate",
    "Time_to_Progression",
]
