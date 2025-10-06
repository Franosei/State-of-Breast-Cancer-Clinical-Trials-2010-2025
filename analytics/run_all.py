# analytics/run_all.py
import importlib

MODULES = [
    "analytics.a01_momentum",
    "analytics.a02_endpoints_matrix",
    "analytics.a03_nm_vs_ei",
    "analytics.a04_biomarker_cohorts",
    "analytics.a05_consort_quality",
    "analytics.a06_geography_access",
    "analytics.a07_priority_shortlists",
]

def main():
    for modname in MODULES:
        mod = importlib.import_module(modname)
        if hasattr(mod, "main"):
            print(f"→ Running {modname}...")
            mod.main()
        else:
            print(f"!! {modname} has no main()")

if __name__ == "__main__":
    main()
