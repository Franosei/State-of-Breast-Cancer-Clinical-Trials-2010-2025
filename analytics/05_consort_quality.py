# analytics/05_consort_quality.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats

# CONSORT checklist items (excluding 'consort_numbers_analyzed')
CONSORT_FLAGS = [
    "consort_participant_flow",
    "consort_recruitment",
    "consort_baseline",
    "consort_outcomes_estimation",
    "consort_ancillary_analyses",
    "consort_harms",
]

# Display order for columns in the heatmap (phase buckets)
ORDERED_PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Other", "NA/Other"]

# Policy: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365


def _titlecase_item(label: str) -> str:
    """Prettify CONSORT flag names for axis labels."""
    return (
        label.replace("consort_", "")
             .replace("_", " ")
             .strip()
             .title()
    )


def main():
    df = add_derived(load_enhanced()).copy()

    # ---- Eligibility (PCD + 12 months) ----
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    eligible_mask = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))

    # Restrict to ELIGIBLE; quality can only be measured where results exist
    df_elig = df.loc[eligible_mask].copy()
    df_posted_elig = df_elig[df_elig.get("has_results", 0) == 1].copy()

    # Phase bucket should already be created by add_derived; fall back if missing
    if "phase_bucket" not in df_posted_elig.columns:
        df_posted_elig["phase_bucket"] = (
            df_posted_elig.get("phase_clean", df_posted_elig.get("phase", pd.Series("NA/Other", index=df_posted_elig.index)))
            .fillna("NA/Other")
        )

    # ---- Compute % reporting of each CONSORT flag among ELIGIBLE+POSTED (by phase) ----
    # mean() over 0/1 flags → share of posted-eligible trials that reported that checklist item
    rates = (
        df_posted_elig
        .groupby("phase_bucket")[CONSORT_FLAGS]
        .mean()
        .mul(100)
        .reindex(ORDERED_PHASES)
    )

    # Keep only phases that actually exist (drop all-NaN columns from reindex)
    rates = rates[[c for c in rates.columns]]
    rates = rates.dropna(how="all", axis=0)

    # Axes labels
    row_labels = [_titlecase_item(c) for c in CONSORT_FLAGS]
    col_labels = [p for p in rates.index]

    # Reorder rows to your CONSORT_FLAGS order (already in that order)
    rates_disp = rates.copy()
    rates_disp = rates_disp[CONSORT_FLAGS] if set(CONSORT_FLAGS).issubset(rates_disp.columns) else rates_disp

    # Transpose for heatmap: rows = CONSORT items, cols = phases
    H = rates_disp.T.reindex(CONSORT_FLAGS)
    H.index = row_labels
    H.columns = col_labels

    # ---- Plot: heatmap with improved readability ----
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Use a perceptually uniform colormap (better gradient, no washed whites)
    im = ax.imshow(
        H.values,
        aspect="auto",
        cmap="YlGnBu",   # better contrast and readability
        vmin=0,
        vmax=100
    )

    # Improve axis labels & layout
    ax.set_yticks(np.arange(len(H.index)), labels=H.index)
    ax.set_xticks(np.arange(len(H.columns)), labels=H.columns, rotation=0)
    ax.set_xlabel("Phase (eligible & posted)", fontsize=11, labelpad=6)
    ax.set_title(
        "CONSORT reporting completeness by phase\n(ELIGIBLE trials only; percentages among posted)",
        fontsize=13, fontweight="bold", pad=10
    )

    # Annotate cells dynamically (white or black text depending on brightness)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            val = H.values[i, j]
            if np.isnan(val):
                continue
            # Adjust text color based on the background intensity
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    # Add colorbar with better label and tick styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("% of eligible-posted trials reporting item", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Clean gridlines and borders for a sharp figure
    ax.spines[:].set_visible(False)
    ax.set_facecolor("#f8f9fa")  # subtle background contrast
    fig.tight_layout()
    save_fig(fig, "05_consort_reporting_by_phase.png")


    # ---- Text summary ----
    lines = []
    # Context: how many eligible & how many posted by phase
    if "phase_bucket" not in df_elig.columns:
        df_elig["phase_bucket"] = (
            df_elig.get("phase_clean", df_elig.get("phase", pd.Series("NA/Other", index=df_elig.index)))
            .fillna("NA/Other")
        )

    lines.append("CONSORT reporting quality (ELIGIBLE trials only; percentages among those with posted results)\n")

    # Per-phase counts & coverage
    phase_counts = (
        df_elig.groupby("phase_bucket")["nct_id"]
        .count()
        .reindex(ORDERED_PHASES)
        .rename("eligible_n")
        .fillna(0)
        .astype(int)
        .to_frame()
    )
    phase_counts["posted_n"] = (
        df_posted_elig.groupby("phase_bucket")["nct_id"]
        .count()
        .reindex(ORDERED_PHASES)
        .fillna(0)
        .astype(int)
    )
    phase_counts["% posted among eligible"] = np.where(
        phase_counts["eligible_n"] > 0,
        100.0 * phase_counts["posted_n"] / phase_counts["eligible_n"],
        np.nan
    ).round(1)

    # Append phase coverage to text
    for phase, row in phase_counts.dropna(how="all").iterrows():
        lines.append(
            f"Phase: {phase} — eligible n={int(row.get('eligible_n', 0))}, "
            f"posted n={int(row.get('posted_n', 0))}, "
            f"% posted={row.get('% posted among eligible', np.nan)}%"
        )

    lines.append("\nItem-level completeness per phase (%, among eligible & posted):")
    for phase in H.columns:
        lines.append(f"\n{phase}:")
        if phase in rates.index:
            for flag, label in zip(CONSORT_FLAGS, row_labels):
                val = rates.loc[phase, flag] if flag in rates.columns else np.nan
                if pd.isna(val):
                    continue
                lines.append(f"- {label}: {val:.1f}%")
        else:
            lines.append("- No eligible-posted trials in this phase.")

    save_stats("\n".join(lines), "05_consort_quality.txt")


if __name__ == "__main__":
    main()
