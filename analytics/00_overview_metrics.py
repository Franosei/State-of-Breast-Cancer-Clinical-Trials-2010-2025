# analytics/00_overview_metrics.py
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import pandas as pd
import numpy as np
from analytics.common import load_enhanced, add_derived, save_stats


def main():
    df = add_derived(load_enhanced())
    n_total = len(df)

    # --- Phase breakdown ---
    # Normalise phase strings
    def clean_phase(ph):
        if pd.isna(ph):
            return "Unspecified"
        ph = str(ph).upper()
        if "PHASE1" in ph and "PHASE2" in ph:
            return "Phase I/II"
        if "PHASE2" in ph and "PHASE3" in ph:
            return "Phase II/III"
        if "PHASE1" in ph:
            return "Phase I"
        if "PHASE2" in ph:
            return "Phase II"
        if "PHASE3" in ph:
            return "Phase III"
        if "PHASE4" in ph:
            return "Phase IV"
        return "Unspecified"

    df["phase_clean"] = df["phase"].map(clean_phase)
    phase_counts = df["phase_clean"].value_counts(dropna=False).sort_index()
    phase_pct = 100 * phase_counts / n_total

    # --- Sponsor breakdown ---
    # Simplify sponsor classification
    def classify_sponsor(s):
        if pd.isna(s):
            return "Unspecified"
        s = str(s).lower()
        if any(x in s for x in ["pharma", "pfizer", "novartis", "astrazeneca", "roche", "bayer", "sanofi", "lilly", "merck", "gsk", "abbvie", "bristol", "takeda", "genentech", "johnson", "boehringer"]):
            return "Industry"
        if any(x in s for x in ["university", "college", "institute", "center", "centre", "hospital"]):
            return "Academic"
        if any(x in s for x in ["nih", "nci", "government", "department of health", "ministry", "agency"]):
            return "Government"
        return "Other"

    df["sponsor_class_auto"] = df["sponsor"].map(classify_sponsor)
    sponsor_counts = df["sponsor_class_auto"].value_counts(dropna=False)
    sponsor_pct = 100 * sponsor_counts / n_total

    # --- Trial momentum ---
    trials_by_year = (
        df.groupby("start_year")["nct_id"]
        .count()
        .reset_index(name="trial_count")
        .sort_values("start_year")
    )
    earliest = int(df["start_year"].min()) if df["start_year"].notna().any() else None
    latest = int(df["start_year"].max()) if df["start_year"].notna().any() else None

    # --- Summary text ---
    txt = f"""
Breast Cancer Clinical Trials: Overview (2010–2025)

Total registered trials analysed: {n_total:,}
Years covered: {earliest}–{latest}

Phase distribution:
{pd.DataFrame({'Count': phase_counts, '%': phase_pct.round(1)}).to_string()}

Sponsor distribution:
{pd.DataFrame({'Count': sponsor_counts, '%': sponsor_pct.round(1)}).to_string()}

Trial starts by year (last 5 shown):
{trials_by_year.tail().to_string(index=False)}

Note:
Phases inferred from free-text labels (e.g., 'PHASE1; PHASE2' → 'Phase I/II').
Sponsor categories automatically derived by pattern matching of sponsor names.
"""
    save_stats(txt, "00_overview_metrics.txt")

if __name__ == "__main__":
    main()
