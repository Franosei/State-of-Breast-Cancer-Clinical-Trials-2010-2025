# analytics/a06_geography_access.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, PALETTE

RECENT_YEARS = 2  # you can reuse this to exclude very recent years if you wish

def main():
    df = add_derived(load_enhanced())
    sc = df["site_count"].astype("float").dropna()
    n_total = len(df)

    # ---------- Panel 1: capped distribution ----------
    if len(sc) > 0:
        p95 = float(sc.quantile(0.95))
        p99 = float(sc.quantile(0.99))
        smax = int(sc.max())
        clipped = np.minimum(sc.values, p99)

        fig1, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.hist(clipped, bins=40, color=PALETTE["primary"])
        ax1.set_title("Sites per Trial — Distribution (capped at 99th percentile)")
        ax1.set_xlabel("Sites per trial")
        ax1.set_ylabel("Trials")

        # annotate tail info
        ax1.axvline(p95, linestyle="--", linewidth=1, color=PALETTE["secondary"])
        ax1.text(p95, ax1.get_ylim()[1]*0.9, f"p95 ≈ {p95:.0f}", rotation=90, va="top", ha="right")
        ax1.axvline(p99, linestyle="--", linewidth=1, color=PALETTE["secondary"])
        ax1.text(p99, ax1.get_ylim()[1]*0.9, f"p99 ≈ {p99:.0f}", rotation=90, va="top", ha="right")

        # footer note with the true max
        ax1.text(0.01, -0.18,
                 f"Note: histogram clipped at 99th percentile ({p99:.0f}). True max = {smax} sites.",
                 transform=ax1.transAxes, fontsize=9)
        save_fig(fig1, "06a_sites_distribution_capped.png")
    else:
        p95 = p99 = smax = 0

    # ---------- Panel 2: scaling over time (median & 90th pct) ----------
    df_time = df[df["start_year"].notna()].copy()
    if not df_time.empty:
        yearly = (df_time.groupby("start_year")
                           .agg(median_sites=("site_count", "median"),
                                p90_sites=("site_count", lambda x: np.nanpercentile(x, 90)),
                                trials=("nct_id", "count"))
                           .reset_index()
                           .sort_values("start_year"))

        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        ax2.plot(yearly["start_year"], yearly["median_sites"], marker="o",
                 color=PALETTE["primary"], label="Median sites")
        ax2.plot(yearly["start_year"], yearly["p90_sites"], marker="o",
                 color=PALETTE["secondary"], label="90th percentile")
        ax2.set_title("Network scale over time")
        ax2.set_xlabel("Start Year")
        ax2.set_ylabel("Sites per trial")
        ax2.legend()
        save_fig(fig2, "06b_sites_over_time_median_p90.png")
    else:
        yearly = pd.DataFrame(columns=["start_year", "median_sites", "p90_sites", "trials"])

    # ---------- Panel 3: transparency by geography footprint ----------
    # side-by-side bars for Single vs Multi: % with results and mean CONSORT score
    geo = (df.groupby("multi_country_flag")
             .agg(results_pct=("has_results", lambda x: 100 * np.nanmean(x)),
                  consort_mean=("consort_results_score", "mean"),
                  n=("nct_id", "count"))
             .reset_index())
    maplbl = {0: "Single-country", 1: "Multi-country"}
    geo["label"] = geo["multi_country_flag"].map(maplbl)

    x = np.arange(len(geo))
    w = 0.38
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    b1 = ax3.bar(x - w/2, geo["results_pct"], width=w, color=PALETTE["primary"], label="% with results")
    # second axis for mean CONSORT score (0–7)
    ax3b = ax3.twinx()
    b2 = ax3b.bar(x + w/2, geo["consort_mean"], width=w, color=PALETTE["secondary"], label="Mean CONSORT score")

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(geo["label"], geo["n"])])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("% with posted results")
    ax3b.set_ylim(0, 7)
    ax3b.set_ylabel("CONSORT results score (0–7)")
    ax3.set_title("Transparency by geography footprint (single vs multi)")

    # unified legend
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper left")
    save_fig(fig3, "06c_transparency_single_vs_multi.png")

    # ---------- Stats text ----------
    txt = f"""Geography summary
Total trials: {n_total}
Distribution (sites per trial):
- p95 ≈ {p95:.0f}, p99 ≈ {p99:.0f}, max = {smax}

Scale over time (selected years):
{yearly[['start_year','trials','median_sites','p90_sites']].tail(8).to_string(index=False)}

Transparency by footprint:
{geo[['label','n','results_pct','consort_mean']].to_string(index=False, formatters={'results_pct':lambda v:f'{v:.1f}'})}
"""
    save_stats(txt, "06_geography_access.txt")

if __name__ == "__main__":
    main()
