# analytics/07_priority_shortlists.py
from __future__ import annotations
import os
import pandas as pd
from analytics.common import load_enhanced, add_derived, DATA_DIR, save_stats

OUT_CSV_DIR = os.path.join(DATA_DIR, "outputs", "shortlists")

def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    df = add_derived(load_enhanced())

    completed = df[df["overall_status"].isin(["COMPLETED", "TERMINATED"])]
    old = completed[completed["years_since_completion"] >= 3]

    # 1) Completed ≥3y, reporting gap, key endpoints planned
    key_eps = ["Objective_Response_Rate","Progression_Free_Survival","Overall_Survival","Pathologic_Complete_Response_pCR"]
    mask_key = (old["reporting_gap_flag"]==1) & (old[key_eps].any(axis=1))
    sl1 = old[mask_key].copy()
    sl1.to_csv(os.path.join(OUT_CSV_DIR, "priority_missing_key_endpoints.csv"), index=False)

    # 2) HER2, pCR planned, gap
    sl2 = completed[(completed["HER2_flag"]==1)
                    & (completed["Pathologic_Complete_Response_pCR"]==1)
                    & (completed["reporting_gap_flag"]==1)]
    sl2.to_csv(os.path.join(OUT_CSV_DIR, "priority_her2_pcr_gap.csv"), index=False)

    # 3) BRCAm, efficacy endpoints missing
    sl3 = completed[(completed["BRCAm_flag"]==1)
                    & (completed[["Progression_Free_Survival","Overall_Survival","Objective_Response_Rate"]].any(axis=1))
                    & (completed["reporting_gap_flag"]==1)]
    sl3.to_csv(os.path.join(OUT_CSV_DIR, "priority_brcam_efficacy_gap.csv"), index=False)

    # 4) NM risk list
    sl4 = completed[(completed["NM_new_medicine"]==1) & (completed["reporting_gap_flag"]==1)]
    sl4.to_csv(os.path.join(OUT_CSV_DIR, "priority_nm_gap.csv"), index=False)

    txt = f"""High-priority shortlists generated:
- priority_missing_key_endpoints.csv (n={len(sl1)})
- priority_her2_pcr_gap.csv (n={len(sl2)})
- priority_brcam_efficacy_gap.csv (n={len(sl3)})
- priority_nm_gap.csv (n={len(sl4)})
"""
    save_stats(txt, "07_priority_shortlists.txt")

if __name__ == "__main__":
    main()
