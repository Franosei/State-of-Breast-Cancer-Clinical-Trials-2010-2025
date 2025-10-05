import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retrieval.clinicaltrials_api import fetch_breast_cancer_trials


def main():
    df = fetch_breast_cancer_trials(
        start_year=2010,
        end_year=2025,
        page_size=200,
    )
    print(df.head())
    print(f"Total unique trials: {len(df)}")


if __name__ == "__main__":
    main()
