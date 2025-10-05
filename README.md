# State-of-Breast-Cancer-Clinical-Trials-2010-2025
This proposal has been prepared in recognition of Breast Cancer Awareness Month. The insights generated will be shared to support transparency, equity, and progress in breast cancer research for the benefit of patients, families, and the global healthcare community.

```
breast_cancer_trials/
│── .env
│── requirements.txt
│── README.md
│── main.py                           # Orchestrates row-by-row enrichment to produce the clean dataset
│
├── taug/                             # TAUG-BrCa source materials (as you noted, at repo root)
│
├── config/
│   └── settings.py                   # OPENAI_API_KEY, paths, batch sizes, etc.
│
├── data/
│   ├── raw/
│   │   └── breast_cancer_trials_unique.csv   # input fetched file (already present)
│   ├── interim/
│   │   └── checkpoints/              # periodic parquet/csv checkpoints during row-by-row processing
│   ├── processed/
│   │   ├── endpoint_dictionary.json  # canonical endpoints (structured)
│   │   ├── endpoint_dictionary.csv   # canonical endpoints (tabular)
│   │   ├── endpoint_synonyms.json    # flat alias map
│   │   ├── breast_cancer_trials_enhanced.parquet  # FINAL clean dataset (authoritative)
│   │   └── breast_cancer_trials_enhanced.csv      # FINAL clean dataset (shareable)
│   └── outputs/
│       └── logs/
│           └── llm_calls.jsonl       # optional audit trail (NM/EI adjudications, ancillary cues)
│
├── retrieval/
│   └── clinicaltrials_api.py         # unchanged (produces ..._unique.csv)
│
├── processing/                       # NEW: everything that builds the clean dataset
│   ├── pipeline.py                   # row-by-row driver (calls the three layers below, in order)
│   ├── endpoints_layer.py            # Layer A: planned endpoints → 15 binary flags (rules + LLM fallback)
│   ├── reporting_layer.py            # Layer B: planned vs reported gap + CONSORT 7 flags
│   └── classify_layer.py             # Layer C: NM/EI/HER2/BRCAm (rules + LLM adjudication)
│
├── llm/
│   └── openai_client.py              # minimal wrapper (JSON-only, temp=0, retries, caching)
│
└── utils/
    ├── helpers.py                    # text normalize, regex, canonicalization, synonym matchers
    └── io.py                         # safe read/write; row-complete writes; checkpointing utilities

```