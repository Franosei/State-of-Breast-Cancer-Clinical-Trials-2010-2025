# analytics/a04_biomarker_cohorts.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, ENDPOINT_COLS, save_fig, save_stats, PALETTE

# focus endpoints most relevant for HER2/BRCAm narratives
FOCUS_ENDPOINTS = [
    "Pathologic_Complete_Response_pCR",
    "Objective_Response_Rate",
    "Progression_Free_Survival",
    "Overall_Survival",
]

RECENT_YEARS = 2  # exclude last N start_years from reporting-rate denominators

def _mature_mask(df: pd.DataFrame) -> pd.Series:
    """Mask to exclude trials started in the last RECENT_YEARS (reporting lag)."""
    if df["start_year"].notna().any():
        max_yr = int(df["start_year"].max())
        cutoff = max_yr - (RECENT_YEARS - 1)
        return (df["start_year"].notna()) & (df["start_year"] < cutoff)
    return pd.Series([True] * len(df), index=df.index)

def _endpoint_metrics(df_all: pd.DataFrame, df_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Build comparative metrics for each endpoint:
      planned_pct_all, planned_pct_cohort, lift,
      reported_pct_all, reported_pct_cohort
    Reporting rate uses only 'mature' trials (excludes last RECENT_YEARS).
    """
    mature_all = df_all[_mature_mask(df_all)]
    mature_coh = df_cohort[_mature_mask(df_cohort)]

    rows = []
    for ep in FOCUS_ENDPOINTS:
        # prevalence (all trials in group)
        planned_all = (df_all[ep] == 1).mean() * 100
        planned_coh = (df_cohort[ep] == 1).mean() * 100
        lift = (planned_coh / planned_all) if planned_all > 0 else np.nan

        # reporting (among trials that planned the endpoint, restricted to mature set)
        def _reported_pct(sub: pd.DataFrame) -> float:
            planned_mask = (sub[ep] == 1)
            denom = planned_mask.sum()
            if denom == 0:
                return 0.0
            num = (planned_mask & (sub["has_results"] == 1)).sum()
            return 100 * num / denom

        rep_all = _reported_pct(mature_all)
        rep_coh = _reported_pct(mature_coh)

        rows.append((ep, planned_all, planned_coh, lift, rep_all, rep_coh))

    m = pd.DataFrame(rows, columns=[
        "endpoint",
        "planned_pct_all", "planned_pct_cohort", "lift",
        "reported_pct_all", "reported_pct_cohort"
    ])
    return m

def _plot_cohort(m: pd.DataFrame, cohort_name: str, n_all: int, n_coh: int):
    # Two-panel figure: left = planned prevalence; right = reporting rate among planned
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]})

    y = np.arange(len(m))
    w = 0.38

    # Panel 1: prevalence (planned %)
    ax1.barh(y - w/2, m["planned_pct_all"], height=w, color=PALETTE["gray"], label="Overall")
    ax1.barh(y + w/2, m["planned_pct_cohort"], height=w, color=PALETTE["primary"], label=cohort_name)
    ax1.set_yticks(y)
    ax1.set_yticklabels(m["endpoint"])
    ax1.invert_yaxis()
    ax1.set_xlabel("% of trials planning endpoint")
    ax1.set_title(f"Planned prevalence — {cohort_name} vs Overall\n(overall n={n_all}, {cohort_name} n={n_coh})")
    ax1.set_xlim(0, 100)
    ax1.legend(loc="lower right")

    # annotate lift (× vs overall)
    for yi, (pct_all, pct_coh, lift) in enumerate(zip(m["planned_pct_all"], m["planned_pct_cohort"], m["lift"])):
        if np.isfinite(lift) and pct_all > 0:
            ax1.text(max(pct_all, pct_coh) + 2, yi, f"{lift:.2f}×", va="center", fontsize=9)

    # Panel 2: reporting rate among those that planned (mature only)
    ax2.barh(y - w/2, m["reported_pct_all"], height=w, color=PALETTE["gray"], label="Overall")
    ax2.barh(y + w/2, m["reported_pct_cohort"], height=w, color=PALETTE["secondary"], label=cohort_name)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])  # avoid duplicate labels
    ax2.invert_yaxis()
    ax2.set_xlabel("% reported (among trials that planned, mature only)")
    ax2.set_title(f"Reporting performance — {cohort_name} vs Overall")
    ax2.set_xlim(0, 100)
    ax2.legend(loc="lower right")

    fig.suptitle(f"{cohort_name} endpoint mix & reporting (excludes last {RECENT_YEARS} years for reporting)", y=0.98)
    fig.tight_layout()
    save_fig(fig, f"04_{cohort_name.lower()}_endpoint_mix_reporting.png")

def main():
    df = add_derived(load_enhanced())
    cohorts = {
        "HER2": df[df["HER2_flag"] == 1],
        "BRCAm": df[df["BRCAm_flag"] == 1],
    }

    # overall baseline
    n_all = len(df)

    # stats text
    lines = []

    for cname, sub in cohorts.items():
        n_coh = len(sub)
        m = _endpoint_metrics(df, sub)
        _plot_cohort(m, cname, n_all, n_coh)

        # text summary
        lines.append(f"Cohort: {cname} (n={n_coh}) vs Overall (n={n_all})")
        lines.append(f"- % with posted results (all trials): {100*sub['has_results'].mean():.1f}%  | Overall: {100*df['has_results'].mean():.1f}%")
        lines.append(f"- Mean CONSORT results score (0–7): {sub['consort_results_score'].mean():.2f}  | Overall: {df['consort_results_score'].mean():.2f}")
        lines.append("Endpoint-level:")
        for _, r in m.iterrows():
            lines.append(
                f"  • {r['endpoint']}: planned% {cname}={r['planned_pct_cohort']:.1f}% (lift {r['lift']:.2f}× vs overall {r['planned_pct_all']:.1f}%), "
                f"reported% {cname}={r['reported_pct_cohort']:.1f}% vs overall {r['reported_pct_all']:.1f}%"
            )
        lines.append("")

    save_stats("\n".join(lines), "04_biomarker_cohorts.txt")

if __name__ == "__main__":
    main()
