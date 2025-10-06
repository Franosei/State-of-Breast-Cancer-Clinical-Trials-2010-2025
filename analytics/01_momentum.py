# analytics/a01_momentum.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, line

RECENT_YEARS = 2  # treat the last 2 calendar years as "too recent" for fair results % comparisons

def main():
    df = add_derived(load_enhanced())

    # annual starts
    g = df.groupby("start_year", dropna=False).size().reset_index(name="count")
    g = g[g["start_year"].notna()].sort_values("start_year")

    # results transparency trend (% with any posted results)
    r = df.groupby("start_year", dropna=False)["has_results"].mean().reset_index()
    r = r[r["start_year"].notna()].sort_values("start_year")
    r["results_pct"] = 100 * r["has_results"]

    # define recent window & mature window years
    if not g.empty:
        min_year = int(g["start_year"].min())
        max_year = int(g["start_year"].max())
    else:
        min_year = max_year = None

    recent_start = None
    mature_latest_year = None
    if max_year is not None:
        recent_start = max_year - (RECENT_YEARS - 1)
        mature_latest_year = max_year - RECENT_YEARS

    # plot
    fig, ax = plt.subplots()
    line(ax, g["start_year"], g["count"], color="primary", label="Trial starts")
    ax2 = ax.twinx()
    line(ax2, r["start_year"], r["results_pct"], color="secondary", label="% with posted results")
    ax.set_title("Breast Cancer Trials: Momentum & Results Transparency (2010–2025)")
    ax.set_xlabel("Start Year")
    ax.set_ylabel("Trial starts")
    ax2.set_ylabel("% with results")

    # clamp right axis to 0..100 so it never looks negative
    ax2.set_ylim(0, 100)

    # shade the "recent" years (last 2)
    if recent_start is not None:
        ax.axvspan(recent_start - 0.5, max_year + 0.5, color="#cccccc", alpha=0.15, label="Recent window")

    # merged legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    save_fig(fig, "01_momentum_transparency.png")

    # stats text
    # recent window (e.g., 2024–2025)
    if recent_start is not None:
        recent_mask_g = g["start_year"] >= recent_start
        recent_mask_r = r["start_year"] >= recent_start
        recent_starts = int(g.loc[recent_mask_g, "count"].sum())
        recent_results_pct = float(r.loc[recent_mask_r, "has_results"].mean() * 100) if recent_mask_r.any() else 0.0
    else:
        recent_starts = 0
        recent_results_pct = 0.0

    # latest mature year (max_year - RECENT_YEARS)
    if mature_latest_year is not None and (r["start_year"] == mature_latest_year).any():
        mature_starts = int(g.loc[g["start_year"] == mature_latest_year, "count"].sum())
        mature_results_pct = float(r.loc[r["start_year"] == mature_latest_year, "results_pct"].iloc[0])
    else:
        mature_starts = 0
        mature_results_pct = 0.0

    txt = f"""
Key stats:
- Years covered: {min_year if min_year is not None else 'NA'}–{max_year if max_year is not None else 'NA'}

Recent window (last {RECENT_YEARS} years: {recent_start}–{max_year}):
- Trial starts (sum): {recent_starts}
- % with posted results (expected low due to lag): {recent_results_pct:.1f}%

Latest mature year (before recent window): {mature_latest_year if mature_latest_year is not None else 'NA'}
- Trial starts: {mature_starts}
- % with posted results: {mature_results_pct:.1f}%
"""
    save_stats(txt, "01_momentum_transparency.txt")

if __name__ == "__main__":
    main()
