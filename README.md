# State-of-Breast-Cancer-Clinical-Trials-2010-2025
This proposal has been prepared in recognition of Breast Cancer Awareness Month. The insights generated will be shared to support transparency, equity, and progress in breast cancer research for the benefit of patients, families, and the global healthcare community.

```
breast_cancer_trials/
│── .env
│── requirements.txt
│── README.md
│── main.py
│
├── taug/
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   │   └── breast_cancer_trials_unique.csv
│   ├── interim/
│   │   └── checkpoints/
│   ├── processed/
│   │   ├── endpoint_dictionary.json
│   │   ├── endpoint_dictionary.csv
│   │   ├── endpoint_synonyms.json
│   │   ├── breast_cancer_trials_enhanced.parquet
│   │   └── breast_cancer_trials_enhanced.csv
│   └── outputs/
│       ├── logs/
│       │   └── llm_calls.jsonl
│       ├── figures/                    # NEW: all plots saved here
│       └── stats/                      # NEW: plain-text stats per analysis
│
├── retrieval/
│   └── clinicaltrials_api.py
│
├── processing/
│   ├── pipeline.py
│   ├── endpoints_layer.py
│   ├── reporting_layer.py
│   └── classify_layer.py
│
├── analytics/                         
│   ├── common.py                       # shared theme + helpers (consistent look)
│   ├── run_all.py                      # run all seven analyses
│   ├── 01_momentum.py                  # #1 Momentum & Maturity
│   ├── 02_endpoints_matrix.py          # #2 Endpoints: planned vs reported
│   ├── 03_nm_vs_ei.py                  # #3 New medicine vs extension
│   ├── 04_biomarker_cohorts.py         # #4 HER2 & BRCAm cohorts
│   ├── 05_consort_quality.py           # #5 CONSORT results quality
│   ├── 06_geography_access.py          # #6 Access & scaling (sites/countries)
│   └── 07_priority_shortlists.py       # #7 High-priority follow-up lists
│
└── llm/
│   └── openai_client.py
│   └── openai_client.py
│   └── openai_client.py
│   └── openai_client.py
│
utils/
  ├── helpers.py
  └── io.py

```