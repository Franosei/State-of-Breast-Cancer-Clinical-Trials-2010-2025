# analytics/05_consort_quality.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, save_fig, save_stats, PALETTE

# CONSORT checklist items (excluding 'consort_numbers_analyzed')
CONSORT_FLAGS = [
    "consort_participant_flow",
    "consort_recruitment",
    "consort_baseline",
    "consort_outcomes_estimation",
    "consort_ancillary_analyses",
    "consort_harms",
]

ORDERED_PHASES = ["Phase 1", "Phase 2", "Phase 3", "Other", "NA/Other"]

def main():
    df = add_derived(load_enhanced())
    posted = df[df["has_results"] == 1].copy()

    # Calculate mean reporting rate (%) for each phase × CONSORT item
    rates = (
        posted.groupby("phase_bucket")[CONSORT_FLAGS]
        .mean()
        .mul(100)
        .reindex(ORDERED_PHASES)
    )

    # --- Plot stacked horizontal bars ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(CONSORT_FLAGS)
    colors = [
        PALETTE.get("primary", "#1f77b4"),
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd"
    ]

    for i, phase in enumerate(rates.index):
        vals = rates.loc[phase].values
        ax.barh(CONSORT_FLAGS, vals, left=bottom, color=colors[i % len(colors)], label=phase)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel("% of trials reporting")
    ax.set_title("CONSORT Reporting Completeness by Phase (Trials with Posted Results)")
    ax.invert_yaxis()
    ax.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(fig, "05_consort_reporting_by_phase.png")

    # --- Save statistics text summary ---
    lines = ["CONSORT reporting rates by phase (posted results only):"]
    for phase, sub in posted.groupby("phase_bucket"):
        r = (sub[CONSORT_FLAGS].mean() * 100).round(1)
        lines.append(f"\nPhase: {phase} (n={len(sub)})")
        for k, v in r.items():
            lines.append(f"- {k}: {v}%")
    save_stats("\n".join(lines), "05_consort_quality.txt")


if __name__ == "__main__":
    main()
