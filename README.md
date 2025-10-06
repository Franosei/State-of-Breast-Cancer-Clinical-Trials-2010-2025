# **State of Breast Cancer Clinical Trials (2010–2025)**

This project was developed in recognition of **Breast Cancer Awareness Month**, aiming to shed light on global trial activity, reporting transparency, and equity in breast cancer research.  
The insights generated are intended to support patients, families, clinicians, and the global research community by promoting open science and data-driven progress.

---

## **Overview**

The repository analyses **8,915 registered breast cancer clinical trials** between 2010 and 2025.  
It explores trends in trial initiation, endpoint reporting, biomarker-defined subgroups (HER2, BRCAm), geographical access, and adherence to **CONSORT** reporting standards.

Large Language Models (LLMs) were used to:
- Parse and classify trial records.
- Identify endpoints and outcome measures.
- Evaluate reporting quality and transparency.

The project combines automated data extraction with human-led interpretation to make the findings both reliable and understandable.

---

## **Project Structure**

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
---

## **How to Run the Project**

### **1. Set up the environment**

Ensure you have **Python 3.10+** installed.  
Then install all required dependencies:

```bash
pip install -r requirements.txt

Create a .env file in the project root and include your API credentials:

OPENAI_API_KEY=your_openai_api_key_here
CLINICALTRIALS_API_KEY=your_clinicaltrials_api_key_here
