# analytics/a03_nm_vs_ei.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, PALETTE

def main():
    df = add_derived(load_enhanced())

    # ---------- TREND: NM vs EI share by start year ----------
    share = (
        df.groupby("start_year", dropna=False)[["NM_new_medicine", "EI_extension_of_indication"]]
          .mean()
          .reset_index()
    )
    share = share[share["start_year"].notna()].sort_values("start_year")
    share["NM_pct"] = 100 * share["NM_new_medicine"]
    share["EI_pct"] = 100 * share["EI_extension_of_indication"]

    fig_trend, ax_t = plt.subplots()
    ax_t.plot(share["start_year"], share["NM_pct"], color=PALETTE["primary"], linewidth=2, label="NM (New Medicine)")
    ax_t.plot(share["start_year"], share["EI_pct"], color=PALETTE["secondary"], linewidth=2, label="EI (Extension of Indication)")
    ax_t.set_title("Portfolio Mix: NM vs EI over time")
    ax_t.set_xlabel("Start Year")
    ax_t.set_ylabel("% of trials (by year)")
    ax_t.set_ylim(0, 100)
    ax_t.legend(loc="upper left")
    save_fig(fig_trend, "03_nm_vs_ei_trend.png")

    # ---------- COHORTS: results & gaps (side-by-side) ----------
    cohorts = [
        ("NM=1", df["NM_new_medicine"] == 1),
        ("EI=1", df["EI_extension_of_indication"] == 1),
        ("NM=0 & EI=0", (df["NM_new_medicine"] == 0) & (df["EI_extension_of_indication"] == 0)),
    ]

    rows = []
    for name, mask in cohorts:
        sub = df[mask]
        n = len(sub)
        results_pct = (sub["has_results"].mean() * 100) if n else 0.0
        gap_pct = (sub["reporting_gap_flag"].mean() * 100) if n else 0.0
        rows.append((name, n, results_pct, gap_pct))
    s = pd.DataFrame(rows, columns=["cohort", "n", "results_pct", "gap_pct"])

    # bars side-by-side
    x = np.arange(len(s))
    w = 0.36

    fig_bar, ax_b = plt.subplots(figsize=(10, 5.5))
    bars1 = ax_b.bar(x - w/2, s["results_pct"], width=w, color=PALETTE["primary"], label="% with results")
    bars2 = ax_b.bar(x + w/2, s["gap_pct"], width=w, color=PALETTE["secondary"], label="% with reporting gap")

    # x labels include cohort sizes
    labels = [f"{c}\n(n={n})" for c, n in zip(s["cohort"], s["n"])]
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("% of trials in cohort")
    ax_b.set_title("NM/EI cohorts: results & gaps (side-by-side)")
    ax_b.legend(loc="upper right")

    # annotate values
    def _annotate(bars):
        for b in bars:
            val = b.get_height()
            ax_b.text(b.get_x() + b.get_width()/2, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    _annotate(bars1)
    _annotate(bars2)

    save_fig(fig_bar, "03_nm_ei_results_gaps.png")

    # stats text
    txt = "NM vs EI summary (cohort sizes, % with results, % with gap):\n" + s.to_string(index=False)
    save_stats(txt, "03_nm_vs_ei.txt")

if __name__ == "__main__":
    main()
