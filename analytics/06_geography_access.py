# analytics/a06_geography_access.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, PALETTE

# Policy: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365

def _fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:.1f}%"

def main():
    df = add_derived(load_enhanced()).copy()
    n_total = len(df)

    # ---------- Common fields ----------
    # Sites
    site_count = pd.to_numeric(df.get("site_count"), errors="coerce")
    # Start year
    start_year = pd.to_numeric(df.get("start_year"), errors="coerce")
    # Eligibility (PCD + 12mo)
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    eligible = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))

    # Results / CONSORT flags
    has_results = df.get("has_results", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    consort_score = pd.to_numeric(df.get("consort_results_score"), errors="coerce")

    # Multi-country label
    mul = df.get("multi_country_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    lbl_map = {0: "Single-country", 1: "Multi-country"}

    # ---------- Panel 1: Sites per trial — distribution (all trials; clipped tail) ----------
    sc = site_count.dropna()
    if not sc.empty:
        p95 = float(np.nanpercentile(sc, 95))
        p99 = float(np.nanpercentile(sc, 99))
        smax = int(np.nanmax(sc))
        clipped = np.minimum(sc.values, p99)

        fig1, ax1 = plt.subplots(figsize=(9.5, 5.6))
        ax1.hist(clipped, bins=40, color=PALETTE.get("primary", "#1f77b4"), edgecolor="white", linewidth=0.5)
        ax1.set_title("Sites per Trial — Distribution (clipped at 99th percentile)", pad=10, fontweight="bold")
        ax1.set_xlabel("Sites per trial")
        ax1.set_ylabel("Trials")

        ax1.axvline(p95, linestyle="--", linewidth=1, color=PALETTE.get("secondary", "#4e79a7"))
        ax1.text(p95, ax1.get_ylim()[1]*0.92, f"p95 ≈ {p95:.0f}", rotation=90, va="top", ha="right")
        ax1.axvline(p99, linestyle="--", linewidth=1, color=PALETTE.get("secondary", "#4e79a7"))
        ax1.text(p99, ax1.get_ylim()[1]*0.92, f"p99 ≈ {p99:.0f}", rotation=90, va="top", ha="right")

        ax1.text(0.01, -0.18,
                 f"Note: histogram clipped at p99 ({p99:.0f}). True max = {smax} sites.",
                 transform=ax1.transAxes, fontsize=9)
        fig1.tight_layout()
        save_fig(fig1, "06a_sites_distribution_capped.png")
    else:
        p95 = p99 = smax = 0

    # ---------- Panel 2: Network scale over time (median & p90; all trials) ----------
    df_time = pd.DataFrame({"start_year": start_year, "site_count": site_count, "nct_id": df.get("nct_id")}).dropna(subset=["start_year"])
    if not df_time.empty:
        yearly = (df_time.groupby("start_year")
                          .agg(median_sites=("site_count", "median"),
                               p90_sites=("site_count", lambda x: np.nanpercentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
                               trials=("nct_id", "count"))
                          .reset_index()
                          .sort_values("start_year"))

        fig2, ax2 = plt.subplots(figsize=(9.5, 5.6))
        ax2.plot(yearly["start_year"], yearly["median_sites"], marker="o",
                 color=PALETTE.get("primary", "#1f77b4"), label="Median sites")
        ax2.plot(yearly["start_year"], yearly["p90_sites"], marker="o",
                 color=PALETTE.get("secondary", "#4e79a7"), label="90th percentile")
        ax2.set_title("Network scale over time", pad=10, fontweight="bold")
        ax2.set_xlabel("Start Year")
        ax2.set_ylabel("Sites per trial")
        ax2.legend(loc="upper left", frameon=False)
        fig2.tight_layout()
        save_fig(fig2, "06b_sites_over_time_median_p90.png")
    else:
        yearly = pd.DataFrame(columns=["start_year", "median_sites", "p90_sites", "trials"])

    # ---------- Panel 3: Transparency by geography footprint (ELIGIBLE ONLY) ----------
    # Compute among eligible trials:
    #   - % reported (eligible)      → left axis
    #   - % overdue  (eligible)      → left axis
    #   - Mean CONSORT score (among eligible & posted) → right axis
    df_elig = pd.DataFrame({
        "multi_country_flag": mul,
        "eligible": eligible,
        "has_results": has_results,
        "consort_results_score": consort_score,
        "nct_id": df.get("nct_id"),
    })

    cohorts = []
    for flag_val in (0, 1):
        mask_all = (df_elig["multi_country_flag"] == flag_val)
        mask_elig = mask_all & df_elig["eligible"]

        n_all = int(mask_all.sum())
        n_elig = int(mask_elig.sum())

        n_reported = int((mask_elig & df_elig["has_results"]).sum())
        n_overdue  = int((mask_elig & (~df_elig["has_results"])).sum())

        pct_reported = np.nan if n_elig == 0 else 100.0 * n_reported / n_elig
        pct_overdue  = np.nan if n_elig == 0 else 100.0 * n_overdue  / n_elig

        mean_consort = np.nan
        if n_reported > 0:
            mean_consort = float(df_elig.loc[mask_elig & df_elig["has_results"], "consort_results_score"].mean())

        cohorts.append({
            "label": lbl_map.get(flag_val, str(flag_val)),
            "n_all": n_all,
            "n_eligible": n_elig,
            "pct_reported_eligible": pct_reported,
            "pct_overdue_eligible": pct_overdue,
            "mean_consort_reported": mean_consort,
        })

    geo = pd.DataFrame(cohorts)

    x = np.arange(len(geo))
    w = 0.34

    c_reported = PALETTE.get("primary", "#1f77b4")
    c_overdue  = PALETTE.get("danger",  "#d62728")
    c_consort  = PALETTE.get("secondary", "#4e79a7")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars_rep = ax3.bar(x - w, geo["pct_reported_eligible"], width=w, color=c_reported, label="% reported (eligible)")
    bars_od  = ax3.bar(x,       geo["pct_overdue_eligible"],  width=w, color=c_overdue,  label="% overdue (eligible)")

    ax3.set_ylim(0, 100)
    ax3.set_ylabel("% of eligible trials in cohort")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{r['label']}\n(n={r['n_all']}, eligible={r['n_eligible']})" for _, r in geo.iterrows()])
    ax3.set_title("Transparency by geography footprint (ELIGIBLE trials only)", pad=10, fontweight="bold")

    # Second axis for CONSORT (eligible & posted)
    ax3b = ax3.twinx()
    bars_cs = ax3b.bar(x + w, geo["mean_consort_reported"], width=w, color=c_consort, label="Mean CONSORT (eligible & posted)")
    ax3b.set_ylim(0, 7)
    ax3b.set_ylabel("CONSORT results score (0–7)")

    # Legend (top-right, unified)
    l1, lab1 = ax3.get_legend_handles_labels()
    l2, lab2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1 + l2, lab1 + lab2, loc="upper right", frameon=False)

    # Annotate bar tops
    def _annotate(bars, fmt="{:.1f}%"):
        for b in bars:
            val = b.get_height()
            if not pd.isna(val):
                ax = b.axes
                ax.text(b.get_x() + b.get_width()/2, val + (1.2 if ax is ax3 else 0.12),
                        fmt.format(val), ha="center", va="bottom", fontsize=9)
    _annotate(bars_rep)
    _annotate(bars_od)
    _annotate(bars_cs, fmt="{:.2f}")

    fig3.tight_layout()
    save_fig(fig3, "06c_transparency_single_vs_multi.png")

    # ---------- Text summary ----------
    txt = f"""Geography summary (aligned with policy = PCD + 12 months where relevant)

Total trials: {n_total:,}

Sites per trial (all trials):
- p95 ≈ {p95:.0f}, p99 ≈ {p99:.0f}, max = {smax}

Scale over time (last 8 rows):
{yearly[['start_year','trials','median_sites','p90_sites']].tail(8).to_string(index=False)}

Transparency by footprint (ELIGIBLE only):
{geo[['label','n_all','n_eligible','pct_reported_eligible','pct_overdue_eligible','mean_consort_reported']]
    .rename(columns={'label':'cohort','n_all':'n','n_eligible':'eligible','pct_reported_eligible':'% reported (eligible)',
                     'pct_overdue_eligible':'% overdue (eligible)','mean_consort_reported':'CONSORT mean (reported-eligible)'})
    .to_string(index=False, formatters={'% reported (eligible)': _fmt_pct, '% overdue (eligible)': _fmt_pct,
                                        'CONSORT mean (reported-eligible)': (lambda v: "NA" if pd.isna(v) else f"{v:.2f}")})}
"""
    save_stats(txt, "06_geography_access.txt")


if __name__ == "__main__":
    main()
