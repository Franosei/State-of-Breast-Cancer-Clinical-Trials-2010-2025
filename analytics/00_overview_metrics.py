# analytics/00_overview_metrics.py
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from analytics.common import load_enhanced, add_derived, save_stats


def normalize_text(x: object) -> str:
    """Generic, non-opinionated normalizer for categorical text."""
    if pd.isna(x):
        return "Unspecified"
    s = str(x).strip()
    # unify separators/spaces; keep semicolons if present (combined labels)
    s = s.replace("_", " ").replace("-", " ")
    # collapse multiple spaces
    s = " ".join(s.split())
    # standardize case, but keep punctuation like '/' and ';'
    # (e.g., 'PHASE1; PHASE2' -> 'Phase1; Phase2')
    parts = []
    for token in s.split(";"):
        token = token.strip()
        # Title-case words inside each token
        token = " ".join(w.capitalize() for w in token.split())
        parts.append(token)
    return "; ".join(parts) if ";" in s else parts[0] if parts else "Unspecified"


def value_counts_norm(series: pd.Series) -> pd.DataFrame:
    """Counts + % for a normalized categorical series; fully data-driven."""
    norm = series.map(normalize_text).fillna("Unspecified")
    counts = norm.value_counts(dropna=False)
    total = int(norm.shape[0])
    pct = (100 * counts / total).round(1)
    return pd.DataFrame({"Count": counts.astype(int), "%": pct})


def main():
    df = add_derived(load_enhanced())
    n_total = len(df)

    # -------- Trial starts by year (data-driven) --------
    start_year = pd.to_numeric(df.get("start_year"), errors="coerce")
    trials_by_year = (
        df.assign(_start_year=start_year)
          .dropna(subset=["_start_year"])
          .assign(_start_year=lambda d: d["_start_year"].astype(int))
          .groupby("_start_year")["nct_id"]
          .count()
          .reset_index(name="trial_count")
          .rename(columns={"_start_year": "start_year"})
          .sort_values("start_year")
    )
    earliest = int(trials_by_year["start_year"].min()) if not trials_by_year.empty else None
    latest   = int(trials_by_year["start_year"].max()) if not trials_by_year.empty else None
    header_years = f"{earliest}–{latest}" if earliest is not None and latest is not None else "N/A"

    # -------- Status (overall_status) — no manual mapping --------
    if "overall_status" in df.columns:
        status_tbl = value_counts_norm(df["overall_status"])
    else:
        status_tbl = pd.DataFrame({"Count": [], "%": []})

    # -------- Phase (phase) — no manual mapping --------
    if "phase" in df.columns:
        phase_tbl = value_counts_norm(df["phase"])
    else:
        phase_tbl = pd.DataFrame({"Count": [], "%": []})

    # -------- Sponsor (sponsor_class) — no manual mapping --------
    warn = ""
    if "sponsor_class" in df.columns:
        sponsor_tbl = value_counts_norm(df["sponsor_class"])
    else:
        sponsor_tbl = pd.DataFrame({"Count": [], "%": []})
        warn = "\n[WARN] Column 'sponsor_class' not found. Add/derive it upstream to see sponsor distribution."

    # -------- Summary text --------
    txt = f"""
Breast Cancer Clinical Trials: Overview ({header_years})

Total registered trials analysed: {n_total:,}
Years covered (by start_year): {header_years}

Status distribution (data-driven from overall_status):
{status_tbl.to_string() if not status_tbl.empty else "N/A"}

Phase distribution (data-driven from phase):
{phase_tbl.to_string() if not phase_tbl.empty else "N/A"}

Sponsor distribution (data-driven from sponsor_class):
{sponsor_tbl.to_string() if not sponsor_tbl.empty else "N/A"}{warn}

Trial starts by year (last 5 shown):
{trials_by_year.tail().to_string(index=False) if not trials_by_year.empty else "N/A"}

Notes:
• No hard-coded category lists. Values are taken directly from your columns and lightly normalized (case/spacing only).
• Combined labels (e.g., "PHASE1; PHASE2") are preserved as combined categories (no manual roll-ups).
• To change grouping (e.g., roll Phase I/II together), do that upstream when creating the columns themselves.
"""
    save_stats(txt, "00_overview_metrics.txt")


if __name__ == "__main__":
    main()
