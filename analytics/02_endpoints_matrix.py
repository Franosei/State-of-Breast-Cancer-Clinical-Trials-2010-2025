# analytics/02_endpoints_matrix.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, ENDPOINT_COLS, save_fig, save_stats, PALETTE

def main():
    df = add_derived(load_enhanced()).copy()

    # Reported endpoints: infer by searching titles mapped earlier (results_outcome_measures mapping).
    # Practical proxy: an endpoint is "reported" if planned flag ==1 AND has_results==1 (conservative).
    # If you later add explicit reported flags per endpoint, swap this.
    metrics = []
    for col in ENDPOINT_COLS:
        planned = int((df[col] == 1).sum())
        reported = int(((df[col] == 1) & (df["has_results"] == 1)).sum())
        rate = (100 * reported / planned) if planned > 0 else 0.0
        metrics.append((col, planned, reported, rate))

    mdf = pd.DataFrame(metrics, columns=["endpoint", "planned", "reported", "reported_pct"])
    mdf = mdf.sort_values("planned", ascending=False)

    # plot: horizontal bars for planned vs reported
    y = np.arange(len(mdf))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, mdf["planned"], color=PALETTE["gray"], label="Planned")
    ax.barh(y, mdf["reported"], color=PALETTE["primary"], label="Reported")
    ax.set_yticks(y)
    ax.set_yticklabels(mdf["endpoint"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of trials")
    ax.set_title("Endpoints: Planned vs Reported (proxy via results posted)")
    ax.legend()
    save_fig(fig, "02_endpoints_planned_vs_reported.png")

    txt = "Endpoint planned vs reported (counts & %):\n"
    for _, row in mdf.iterrows():
        txt += f"- {row['endpoint']}: planned={int(row['planned'])}, reported={int(row['reported'])}, reported%={row['reported_pct']:.1f}\n"
    save_stats(txt, "02_endpoints_planned_vs_reported.txt")

if __name__ == "__main__":
    main()
