# analytics/a01_momentum.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, line

# VISUAL ONLY: shade the last N start cohorts as "recent", since many won’t be eligible yet
RECENT_YEARS = 2
# POLICY: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365

def _safe_pct(num: int | float, den: int | float) -> float:
    return float(np.nan if den in (0, None, np.nan) else (100.0 * num / den))

def _summarize_block(df_masked: pd.DataFrame, label: str) -> str:
    """Return a text block with numbers and percentages for a masked slice."""
    total_trials = int(df_masked["n_total"].sum())
    total_eligible = int(df_masked["eligible"].sum())
    total_overdue = int(df_masked["overdue"].sum())
    total_reported_eligible = int(df_masked["reported_eligible"].sum())
    total_reported_on_time = int(df_masked["reported_on_time"].sum())
    total_reported_late_unknown = int(df_masked["reported_late_or_unknown"].sum())

    pct_reported_of_all = _safe_pct(total_reported_eligible, total_trials)
    pct_overdue_of_all = _safe_pct(total_overdue, total_trials)
    pct_on_time_among_eligible = _safe_pct(total_reported_on_time, total_eligible)
    pct_late_among_eligible = _safe_pct(total_reported_late_unknown, total_eligible)
    pct_on_time_among_reported_eligible = _safe_pct(total_reported_on_time, total_reported_eligible)
    pct_late_among_reported_eligible = _safe_pct(total_reported_late_unknown, total_reported_eligible)

    return f"""
{label}
- Total trials: {total_trials:,}
- Eligible to report (PCD+12mo): {total_eligible:,}
- Reported (eligible): {total_reported_eligible:,}  | Overdue (eligible, no results): {total_overdue:,}
- % of ALL trials reported (eligible): {pct_reported_of_all:.1f}%  | Overdue: {pct_overdue_of_all:.1f}%

Among eligible that REPORTED:
- On-time: {total_reported_on_time:,} ({pct_on_time_among_eligible:.1f}% of eligible; {pct_on_time_among_reported_eligible:.1f}% of reported-eligible)
- Late/Unknown date: {total_reported_late_unknown:,} ({pct_late_among_eligible:.1f}% of eligible; {pct_late_among_reported_eligible:.1f}% of reported-eligible)
""".rstrip()

def main():
    df = add_derived(load_enhanced())

    # --- Pull key columns with safe defaults ---
    start_year = pd.to_numeric(df.get("start_year"), errors="coerce")
    has_results = df.get("has_results", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    # Primary Completion Date (PCD) – policy anchor
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")

    # Results-posted date (if available). If not present, fall back to last_update_date as a proxy
    if "results_first_posted_date" in df.columns:
        res_posted = pd.to_datetime(df["results_first_posted_date"], errors="coerce")
        posted_source = "results_first_posted_date"
    else:
        res_posted = pd.to_datetime(df.get("last_update_date"), errors="coerce")
        posted_source = "last_update_date (proxy)"

    today = pd.Timestamp.today().normalize()

    # --- Policy logic per record ---
    due_date = pcd + pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)
    eligible = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))
    reported = has_results

    # On-time vs late/unknown among eligible & reported
    reported_on_time = eligible & reported & res_posted.notna() & (res_posted <= due_date)
    reported_late_or_unknown = eligible & reported & (~reported_on_time)

    overdue = eligible & (~reported)

    # --- Build per-record, then per-year aggregates ---
    df_year_rows = pd.DataFrame({
        "start_year": start_year,
        "n_total": 1,
        "eligible": eligible,
        "overdue": overdue,
        "reported_eligible": reported,  # reported AND eligible (we'll ensure eligible later)
        "reported_on_time": reported_on_time,
        "reported_late_or_unknown": reported_late_or_unknown,
        "nct_id": df.get("nct_id"),
    }).dropna(subset=["start_year"])

    # Make sure reported_eligible excludes non-eligible
    df_year_rows["reported_eligible"] = df_year_rows["reported_eligible"] & df_year_rows["eligible"]

    # trial starts per year
    g_starts = (
        df_year_rows.groupby("start_year")["nct_id"]
             .count()
             .reset_index(name="count")
             .sort_values("start_year")
    )

    # outcomes per year (counts)
    g_out = (
        df_year_rows.groupby("start_year")
             .agg(
                 n_total=("n_total", "sum"),
                 eligible=("eligible", "sum"),
                 overdue=("overdue", "sum"),
                 reported_eligible=("reported_eligible", "sum"),
                 reported_on_time=("reported_on_time", "sum"),
                 reported_late_or_unknown=("reported_late_or_unknown", "sum"),
             )
             .reset_index()
             .sort_values("start_year")
    )

    # % over the whole cohort for that start_year (not just among eligible)
    g_out["pct_reported_of_all"] = np.where(
        g_out["n_total"] > 0, 100.0 * g_out["reported_eligible"] / g_out["n_total"], np.nan
    )
    g_out["pct_overdue_of_all"] = np.where(
        g_out["n_total"] > 0, 100.0 * g_out["overdue"] / g_out["n_total"], np.nan
    )

    # Context: among eligible (kept out of plot, used in text)
    g_out["pct_on_time_among_eligible"] = np.where(
        g_out["eligible"] > 0, 100.0 * g_out["reported_on_time"] / g_out["eligible"], np.nan
    )
    g_out["pct_late_or_unknown_among_eligible"] = np.where(
        g_out["eligible"] > 0, 100.0 * g_out["reported_late_or_unknown"] / g_out["eligible"], np.nan
    )

    # --- Year bounds & "recent" shading window (visual only) ---
    if not g_starts.empty:
        min_year = int(g_starts["start_year"].min())
        max_year = int(g_starts["start_year"].max())
        recent_start = max_year - (RECENT_YEARS - 1)
        mature_latest_year = max_year - RECENT_YEARS
    else:
        min_year = max_year = recent_start = mature_latest_year = None

    # --- Plot ---
    fig, ax = plt.subplots()
    # Trial starts
    line(ax, g_starts["start_year"], g_starts["count"], color="primary", label="Trial starts")

    # % outcomes over the whole cohort
    ax2 = ax.twinx()
    mask_rep = g_out["pct_reported_of_all"].notna()
    line(
        ax2,
        g_out.loc[mask_rep, "start_year"],
        g_out.loc[mask_rep, "pct_reported_of_all"],
        color="secondary",
        label="% reported (eligible) of all trials",
    )
    mask_od = g_out["pct_overdue_of_all"].notna()
    line(
        ax2,
        g_out.loc[mask_od, "start_year"],
        g_out.loc[mask_od, "pct_overdue_of_all"],
        color="secondary",
        label="% overdue (eligible, no results) of all trials",
        marker="x",
    )

    ax.set_title(
        f"Breast Cancer Trials: Momentum & Results Reporting (policy = PCD + 12 months) "
        f"({min_year if min_year is not None else 'NA'}–{max_year if max_year is not None else 'NA'})"
    )
    ax.set_xlabel("Start Year")
    ax.set_ylabel("Trial starts")
    ax2.set_ylabel("% of all trials in cohort")
    ax2.set_ylim(0, 100)

    # Shade the recent cohorts (most are not yet eligible)
    if max_year is not None:
        ax.axvspan(recent_start - 0.5, max_year + 0.5, color="#cccccc", alpha=0.15, label="Recent window")

    # merged legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    save_fig(fig, "01_momentum_transparency.png")

    # --- TEXT SUMMARY ---
    # Build a “flat” DF for slicing summary blocks easily
    flat = g_out.copy()

    # Overall block (all years)
    overall_txt = _summarize_block(flat, "OVERALL (all start years)")

    # 2020+ block (trials that STARTED from 2020 onward)
    if not flat.empty:
        since_2020 = flat[flat["start_year"] >= 2020]
        if not since_2020.empty:
            since_2020_txt = _summarize_block(since_2020, "SINCE 2020 (start_year ≥ 2020)")
        else:
            since_2020_txt = "SINCE 2020 (start_year ≥ 2020)\n- No trials in this range."
    else:
        since_2020_txt = "SINCE 2020 (start_year ≥ 2020)\n- No trials."

    # Recent window (visual shading) quick roll-up for context
    if max_year is not None:
        recent_mask = (flat["start_year"] >= (max_year - (RECENT_YEARS - 1)))
        recent_pct_reported_all = float(flat.loc[recent_mask, "pct_reported_of_all"].dropna().mean()) if recent_mask.any() else float("nan")
        recent_pct_overdue_all = float(flat.loc[recent_mask, "pct_overdue_of_all"].dropna().mean()) if recent_mask.any() else float("nan")
        recent_txt = f"RECENT window (last {RECENT_YEARS} years: {max_year - (RECENT_YEARS - 1)}–{max_year})\n- Avg % reported of ALL: {recent_pct_reported_all:.1f}% | Avg % overdue of ALL: {recent_pct_overdue_all:.1f}%"
    else:
        recent_txt = "RECENT window: N/A"

    txt = f"""
Key policy: Results are due 12 months after the Primary Completion Date (PCD).
Using results date source: {posted_source}

Coverage:
- Years covered: {min_year if min_year is not None else 'NA'}–{max_year if max_year is not None else 'NA'}

{overall_txt}

{since_2020_txt}

{recent_txt}
""".strip()

    save_stats(txt, "01_momentum_transparency.txt")


if __name__ == "__main__":
    main()
