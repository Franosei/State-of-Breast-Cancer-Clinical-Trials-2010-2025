# analytics/a04_biomarker_cohorts.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, ENDPOINT_COLS, save_fig, save_stats, PALETTE

# Focus endpoints most relevant for HER2/BRCAm narratives
FOCUS_ENDPOINTS = [
    "Pathologic_Complete_Response_pCR",
    "Objective_Response_Rate",
    "Progression_Free_Survival",
    "Overall_Survival",
]

# Policy: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365


def _safe_pct(num: int | float, den: int | float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return 0.0
    return float(100.0 * num / den)


def _endpoint_metrics_eligible(df_all_elig: pd.DataFrame, df_cohort_elig: pd.DataFrame) -> pd.DataFrame:
    """
    Build comparative metrics for each endpoint, restricted to ELIGIBLE trials only:
      - planned_pct_all (among eligible overall)
      - planned_pct_cohort (among eligible in cohort)
      - lift (cohort vs overall planned prevalence)
      - reported_pct_all (among eligible overall that planned, proxy = has_results)
      - reported_pct_cohort (among eligible cohort that planned, proxy = has_results)
    """
    rows = []
    for ep in FOCUS_ENDPOINTS:
        # --- Planned prevalence among ELIGIBLE ---
        planned_all_den = len(df_all_elig)
        planned_coh_den = len(df_cohort_elig)

        planned_all_num = int((df_all_elig[ep] == 1).sum())
        planned_coh_num = int((df_cohort_elig[ep] == 1).sum())

        planned_pct_all = _safe_pct(planned_all_num, planned_all_den)
        planned_pct_cohort = _safe_pct(planned_coh_num, planned_coh_den)
        lift = (planned_pct_cohort / planned_pct_all) if planned_pct_all > 0 else np.nan

        # --- Reporting among those that PLANNED (eligible only) ---
        # Proxy: if trial posted any results (has_results == 1), we count that endpoint as "reported"
        planned_mask_all = (df_all_elig[ep] == 1)
        planned_mask_coh = (df_cohort_elig[ep] == 1)

        reported_all_num = int((planned_mask_all & (df_all_elig["has_results"] == 1)).sum())
        reported_coh_num = int((planned_mask_coh & (df_cohort_elig["has_results"] == 1)).sum())

        reported_all_den = int(planned_mask_all.sum())
        reported_coh_den = int(planned_mask_coh.sum())

        reported_pct_all = _safe_pct(reported_all_num, reported_all_den)
        reported_pct_cohort = _safe_pct(reported_coh_num, reported_coh_den)

        rows.append({
            "endpoint": ep,
            "planned_pct_all": round(planned_pct_all, 1),
            "planned_pct_cohort": round(planned_pct_cohort, 1),
            "lift": (round(lift, 2) if np.isfinite(lift) else np.nan),
            "reported_pct_all": round(reported_pct_all, 1),
            "reported_pct_cohort": round(reported_pct_cohort, 1),
            # useful counts for text/debugging
            "planned_all_num": planned_all_num,
            "planned_all_den": planned_all_den,
            "planned_coh_num": planned_coh_num,
            "planned_coh_den": planned_coh_den,
            "reported_all_num": reported_all_num,
            "reported_all_den": reported_all_den,
            "reported_coh_num": reported_coh_num,
            "reported_coh_den": reported_coh_den,
        })

    m = pd.DataFrame(rows)
    # Sort by how often the endpoint is planned in the cohort (eligible only)
    return m.sort_values("planned_pct_cohort", ascending=False).reset_index(drop=True)


def _plot_cohort(m_elig: pd.DataFrame, cohort_name: str, n_all_elig: int, n_coh_elig: int):
    """
    Two-panel figure:
      - Left: planned prevalence among ELIGIBLE trials (Overall vs Cohort)
      - Right: reporting rate among ELIGIBLE trials that PLANNED (Overall vs Cohort)
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]})

    # Labels
    y = np.arange(len(m_elig))
    w = 0.38

    # Colors (publication-ready; fallbacks if PALETTE keys absent)
    c_overall = PALETTE.get("gray", "#98a2b3")
    c_cohort1 = PALETTE.get("primary", "#1f77b4")
    c_cohort2 = PALETTE.get("secondary", "#4e79a7")

    # Panel 1: planned prevalence among eligible
    ax1.barh(y - w/2, m_elig["planned_pct_all"], height=w, color=c_overall, label="Overall (eligible)")
    ax1.barh(y + w/2, m_elig["planned_pct_cohort"], height=w, color=c_cohort1, label=f"{cohort_name} (eligible)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(m_elig["endpoint"])
    ax1.invert_yaxis()
    ax1.set_xlabel("% of ELIGIBLE trials planning endpoint")
    ax1.set_title(f"Planned prevalence")
    ax1.set_xlim(0, 100)
    ax1.legend(loc="upper right")

    # annotate lift (× vs overall)
    for yi, (pct_all, pct_coh, lift) in enumerate(zip(m_elig["planned_pct_all"], m_elig["planned_pct_cohort"], m_elig["lift"])):
        if np.isfinite(lift) and pct_all > 0:
            ax1.text(max(pct_all, pct_coh) + 2, yi, f"{lift:.2f}×", va="center", fontsize=9)

    # Panel 2: reporting performance among eligible that PLANNED
    ax2.barh(y - w/2, m_elig["reported_pct_all"], height=w, color=c_overall, label="Overall (eligible & planned)")
    ax2.barh(y + w/2, m_elig["reported_pct_cohort"], height=w, color=c_cohort2, label=f"{cohort_name} (eligible & planned)")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])  # avoid duplicate labels
    ax2.invert_yaxis()
    ax2.set_xlabel("% reported (ELIGIBLE & planned)")
    ax2.set_title(f"Reporting performance")
    ax2.set_xlim(0, 100)
    ax2.legend(loc="upper right")

    fig.suptitle(f"{cohort_name} endpoint mix & reporting (ELIGIBLE trials only: PCD + 12 months)", y=0.98)
    fig.tight_layout()
    save_fig(fig, f"04_{cohort_name.lower()}_endpoint_mix_reporting.png")


def main():
    df = add_derived(load_enhanced()).copy()

    # --- Eligibility mask (PCD + 12 months) ---
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    eligible_mask = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))

    # Restrict to ELIGIBLE trials only
    df_elig = df.loc[eligible_mask].copy()

    # Define cohorts (not necessarily mutually exclusive)
    cohorts = {
        "HER2": df_elig[df_elig["HER2_flag"] == 1],
        "BRCAm": df_elig[df_elig["BRCAm_flag"] == 1],
    }

    # Overall baseline (eligible only)
    n_all_elig = len(df_elig)

    lines = []
    for cname, sub_elig in cohorts.items():
        n_coh_elig = len(sub_elig)

        # Endpoint metrics (eligible only)
        m_elig = _endpoint_metrics_eligible(df_elig, sub_elig)
        _plot_cohort(m_elig, cname, n_all_elig, n_coh_elig)

        # Eligible-only % with any results (context)
        pct_results_coh = _safe_pct(int((sub_elig["has_results"] == 1).sum()), n_coh_elig)
        pct_results_all = _safe_pct(int((df_elig["has_results"] == 1).sum()), n_all_elig)

        # CONSORT score comparison among eligible (if present)
        cons_coh = sub_elig["consort_results_score"].mean() if "consort_results_score" in sub_elig.columns else np.nan
        cons_all = df_elig["consort_results_score"].mean() if "consort_results_score" in df_elig.columns else np.nan

        # Text summary block
        lines.append(f"Cohort: {cname} (eligible n={n_coh_elig}) vs Overall (eligible n={n_all_elig})")
        lines.append(f"- % with posted results (eligible only): {pct_results_coh:.1f}%  | Overall: {pct_results_all:.1f}%")
        if not pd.isna(cons_coh) and not pd.isna(cons_all):
            lines.append(f"- Mean CONSORT results score (0–7, eligible only): {cons_coh:.2f}  | Overall: {cons_all:.2f}")
        lines.append("Endpoint-level (eligible only):")
        for _, r in m_elig.iterrows():
            lines.append(
                f"  • {r['endpoint']}: planned% {cname}={r['planned_pct_cohort']:.1f}% "
                f"(lift {r['lift'] if np.isfinite(r['lift']) else float('nan'):.2f}× vs overall {r['planned_pct_all']:.1f}%), "
                f"reported% {cname}={r['reported_pct_cohort']:.1f}% vs overall {r['reported_pct_all']:.1f}%"
            )
        lines.append("")

    save_stats("\n".join(lines), "04_biomarker_cohorts.txt")


if __name__ == "__main__":
    main()
