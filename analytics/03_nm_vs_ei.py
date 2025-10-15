# analytics/a03_nm_vs_ei.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, PALETTE

# Policy window: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365

def _safe_pct(num: int | float, den: int | float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return float("nan")
    return float(100.0 * num / den)

def main():
    df = add_derived(load_enhanced()).copy()

    # ============= 1) TREND: NM vs EI share by start year (ALL trials) =============
    share = (
        df.groupby("start_year", dropna=False)[["NM_new_medicine", "EI_extension_of_indication"]]
          .mean()
          .reset_index()
    )
    share = share[share["start_year"].notna()].sort_values("start_year")
    share["NM_pct"] = 100.0 * share["NM_new_medicine"]
    share["EI_pct"] = 100.0 * share["EI_extension_of_indication"]

    fig_trend, ax_t = plt.subplots()
    ax_t.plot(share["start_year"], share["NM_pct"], color=PALETTE.get("primary", "#1f77b4"), linewidth=2, label="NM (New Medicine)")
    ax_t.plot(share["start_year"], share["EI_pct"], color=PALETTE.get("secondary", "#4e79a7"), linewidth=2, label="EI (Extension of Indication)")
    ax_t.set_title("Portfolio Mix: NM vs EI over time")
    ax_t.set_xlabel("Start Year")
    ax_t.set_ylabel("% of trials (by year)")
    ax_t.set_ylim(0, 100)
    ax_t.legend(loc="upper left")
    fig_trend.tight_layout()
    save_fig(fig_trend, "03_nm_vs_ei_trend.png")

    # ============= 2) REPORTING GAP (ELIGIBLE ONLY): NM/EI cohorts =============
    # Columns we rely on
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")
    has_results = df.get("has_results", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    today = pd.Timestamp.today().normalize()
    eligible_mask = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))

    # Define mutually exclusive cohorts: NM=1, EI=1, neither
    nm = (df["NM_new_medicine"] == 1)
    ei = (df["EI_extension_of_indication"] == 1)
    neither = (df["NM_new_medicine"] == 0) & (df["EI_extension_of_indication"] == 0)

    cohorts = [
        ("NM = 1", nm),
        ("EI = 1", ei),
        ("NM = 0 & EI = 0", neither),
    ]

    rows = []
    for label, mask in cohorts:
        sub_all = df[mask]
        n_all = len(sub_all)

        # Restrict to eligible in this cohort
        elig = eligible_mask & mask
        n_eligible = int(elig.sum())

        # Among eligible, reported vs overdue
        reported_eligible = int((elig & has_results).sum())
        overdue_eligible = int((elig & (~has_results)).sum())

        # Percentages are **among eligible** (policy-correct)
        pct_reported_eligible = _safe_pct(reported_eligible, n_eligible)
        pct_overdue_eligible = _safe_pct(overdue_eligible, n_eligible)

        rows.append({
            "cohort": label,
            "n_all": n_all,
            "n_eligible": n_eligible,
            "reported_eligible": reported_eligible,
            "overdue_eligible": overdue_eligible,
            "pct_reported_eligible": pct_reported_eligible,
            "pct_overdue_eligible": pct_overdue_eligible,
        })

    s = pd.DataFrame(rows)

    # ---- Bar plot (eligible only): % reported vs % overdue by cohort ----
    x = np.arange(len(s))
    w = 0.38

    c_reported = PALETTE.get("primary", "#1f77b4")
    c_overdue  = PALETTE.get("danger",  "#d62728")  # red for attention

    fig_bar, ax_b = plt.subplots(figsize=(10, 6))
    bars1 = ax_b.bar(x - w/2, s["pct_reported_eligible"], width=w, color=c_reported, label="% reported (eligible)")
    bars2 = ax_b.bar(x + w/2, s["pct_overdue_eligible"],  width=w, color=c_overdue,  label="% overdue (eligible)")

    # X labels: cohort + sizes (total and eligible)
    xticklabels = [f"{c}\n(n={n_all}, eligible={n_elig})" for c, n_all, n_elig in zip(s["cohort"], s["n_all"], s["n_eligible"])]
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xticklabels)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("% of eligible trials in cohort")
    ax_b.set_title("NM/EI cohorts among trials SUPPOSED to report (PCD + 12 months): % reported vs % overdue")
    ax_b.legend(loc="upper right")

    # annotate bar values
    def _annotate(bars):
        for b in bars:
            val = b.get_height()
            if not pd.isna(val):
                ax_b.text(b.get_x() + b.get_width()/2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    _annotate(bars1)
    _annotate(bars2)

    fig_bar.tight_layout()
    save_fig(fig_bar, "03_nm_ei_results_gaps.png")

    # ---- Text summary (eligible only) ----
    lines = [
        "NM vs EI reporting gap (policy = PCD + 12 months):",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"- {r['cohort']}: "
            f"total n={int(r['n_all'])}, eligible n={int(r['n_eligible'])} | "
            f"reported (eligible)={int(r['reported_eligible'])} "
            f"({r['pct_reported_eligible']:.1f}%), "
            f"overdue (eligible)={int(r['overdue_eligible'])} "
            f"({r['pct_overdue_eligible']:.1f}%)"
        )

    save_stats("\n".join(lines), "03_nm_vs_ei.txt")


if __name__ == "__main__":
    main()
