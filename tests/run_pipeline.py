# tests/run_pipeline.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from processing.pipeline import run

if __name__ == "__main__":
    df = run()
    print("Rows:", len(df))
    print("Saved:")
    print(" - data/processed/breast_cancer_trials_enhanced.parquet")
    print(" - data/processed/breast_cancer_trials_enhanced.csv")
    print(" - data/processed/breast_cancer_trials_enhanced.jsonl")
