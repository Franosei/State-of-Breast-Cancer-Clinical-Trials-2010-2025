# analytics/02_endpoints_matrix.py
from __future__ import annotations
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics.common import load_enhanced, add_derived, ENDPOINT_COLS, save_fig, save_stats, PALETTE

# Policy: results due 12 months after Primary Completion Date (PCD)
ELIGIBILITY_LAG_DAYS = 365

def _endpoint_label_from_col(col: str) -> str:
    """
    Heuristic to turn a column like 'Progression_Free_Survival' into a label
    likely to match items inside detected_endpoint_names.
    Override here for corner cases if needed.
    """
    pretty = col.replace("_", " ").strip()
    # Title-case but keep common all-caps acronyms readable
    pretty = " ".join(w.upper() if w.isupper() and len(w) <= 4 else w.title() for w in pretty.split())
    # Small refinements
    pretty = pretty.replace(" Pcr", " pCR").replace(" Orr", " ORR")
    return pretty

def _parse_detected_names(s: object) -> set[str]:
    """
    detected_endpoint_names often comes as a JSON-like string (e.g., '["PFS","OS"]').
    Return a lowercase set of names for robust matching. Empty set if missing.
    """
    if pd.isna(s):
        return set()
    try:
        if isinstance(s, (list, tuple)):
            vals = s
        else:
            text = str(s).strip()
            # Accept JSON or a python-literal-like string
            if text.startswith("["):
                vals = json.loads(text)
            else:
                # fallback: split on commas if someone stored "A, B, C"
                vals = [v.strip() for v in text.split(",") if v.strip()]
        return {str(v).strip().lower() for v in vals}
    except Exception:
        return set()

def main():
    df = add_derived(load_enhanced()).copy()

    # --- Inputs we rely on ---
    pcd = pd.to_datetime(df.get("primary_completion_date"), errors="coerce")
    has_results = df.get("has_results", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    # Detect endpoints actually present in posted results
    detected_col = "detected_endpoint_names"
    detected_sets = df.get(detected_col)
    if detected_sets is None:
        # Create an empty series if column not present
        detected_sets = pd.Series([None] * len(df), index=df.index)
    detected_sets = detected_sets.map(_parse_detected_names)

    # --- Eligibility (PCD + 12 months) ---
    today = pd.Timestamp.today().normalize()
    eligible_mask = pcd.notna() & (pcd <= (today - pd.Timedelta(days=ELIGIBILITY_LAG_DAYS)))

    df_elig = df.loc[eligible_mask].copy()
    has_results_elig = has_results.loc[eligible_mask]
    detected_sets_elig = detected_sets.loc[eligible_mask]

    total_eligible = int(eligible_mask.sum())
    total_eligible_with_results = int(has_results_elig.sum())

    # --- Build per-endpoint metrics within the eligible population only ---
    rows = []
    for col in ENDPOINT_COLS:
        label = _endpoint_label_from_col(col)

        planned_mask = (df_elig[col] == 1)

        # Endpoint considered "reported present" if the trial posted results AND we detect the endpoint name
        # in detected_endpoint_names. We also require it was planned, since the matrix is about planned vs reported.
        # NOTE: if your detected names use slightly different wording, tweak _endpoint_label_from_col or add synonyms below.
        label_lc = label.lower()

        # robust containment: any detected name containing the label or vice versa
        def _endpoint_present(name_set: set[str]) -> bool:
            for n in name_set:
                if label_lc in n or n in label_lc:
                    return True
            return False

        reported_present_mask = planned_mask & has_results_elig & detected_sets_elig.map(_endpoint_present)

        # Reported but missing planned endpoint: planned + has_results but we did NOT detect the endpoint label
        reported_missing_mask = planned_mask & has_results_elig & (~reported_present_mask)

        planned = int(planned_mask.sum())
        reported_present = int(reported_present_mask.sum())
        reported_missing = int(reported_missing_mask.sum())

        # Coverage among planned & eligible
        coverage_pct = (100.0 * reported_present / planned) if planned > 0 else 0.0
        missing_pct = (100.0 * reported_missing / planned) if planned > 0 else 0.0

        rows.append({
            "endpoint": label,
            "planned_eligible": planned,
            "reported_present": reported_present,
            "reported_missing": reported_missing,
            "coverage_pct_among_planned": round(coverage_pct, 1),
            "missing_pct_among_planned": round(missing_pct, 1),
        })

    mdf = pd.DataFrame(rows).sort_values("planned_eligible", ascending=False).reset_index(drop=True)

    # --- Plot: 3 bars per endpoint (eligible only) ---
    y = np.arange(len(mdf))

    # Nice publication-ready colours (fallbacks if PALETTE keys not present)
    c_planned  = PALETTE.get("gray",     "#98a2b3")
    c_present  = PALETTE.get("primary",  "#1f77b4")
    c_missing  = PALETTE.get("danger",   "#d62728")  # red for attention

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y, mdf["planned_eligible"], color=c_planned,  label="Planned (eligible)")
    ax.barh(y, mdf["reported_present"], color=c_present,  label="Reported w/ endpoint")
    ax.barh(y, mdf["reported_missing"], color=c_missing,  label="Reported but missing planned endpoint")

    ax.set_yticks(y)
    ax.set_yticklabels(mdf["endpoint"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of trials")
    ax.set_title("Endpoints among trials SUPPOSED to report (PCD + 12 months): planned vs reported presence/missing")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "02_endpoints_planned_vs_reported.png")

    # --- Text summary (eligible only) ---
    lines = [
        "Endpoints analysis (eligible trials only; policy = PCD + 12 months):",
        f"- Eligible trials (should have reported): {total_eligible:,}",
        f"- Eligible trials WITH results: {total_eligible_with_results:,}",
        "",
        "Per-endpoint (counts within eligible trials that PLANNED the endpoint):",
    ]
    for _, r in mdf.iterrows():
        lines.append(
            f"- {r['endpoint']}: planned={int(r['planned_eligible'])}, "
            f"reported w/ endpoint={int(r['reported_present'])} "
            f"({r['coverage_pct_among_planned']:.1f}% of planned), "
            f"reported but MISSING planned endpoint={int(r['reported_missing'])} "
            f"({r['missing_pct_among_planned']:.1f}% of planned)"
        )

    # Totals across endpoints (optional, gives a sense of overall coverage vs missing among planned)
    total_planned = int(mdf["planned_eligible"].sum())
    total_present = int(mdf["reported_present"].sum())
    total_missing = int(mdf["reported_missing"].sum())
    overall_cov = (100.0 * total_present / total_planned) if total_planned > 0 else 0.0
    overall_miss = (100.0 * total_missing / total_planned) if total_planned > 0 else 0.0

    lines += [
        "",
        f"Overall across endpoints (eligible & planned): planned={total_planned:,}, "
        f"reported w/ endpoint={total_present:,} ({overall_cov:.1f}%), "
        f"reported but MISSING planned endpoint={total_missing:,} ({overall_miss:.1f}%)",
        "",
        "Notes:",
        "• Entire analysis is restricted to trials that have reached Primary Completion Date + 12 months.",
        f"• Endpoint presence is inferred from `{detected_col}` by matching names to each endpoint label.",
        "• If you maintain a canonical mapping from ENDPOINT_COLS → reported endpoint labels, add it to improve matching.",
    ]
    save_stats("\n".join(lines), "02_endpoints_planned_vs_reported.txt")


if __name__ == "__main__":
    main()
