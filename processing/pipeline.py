# processing/pipeline.py
from __future__ import annotations
import os
import pandas as pd
from typing import Dict, Any, Optional, List

from utils.io import (
    RAW_DIR, PROCESSED_DIR, INTERIM_DIR,
    ensure_dirs, read_csv_safely,
    save_csv_parquet_dual, save_jsonl, save_checkpoint_dual,
)
from processing.endpoints_layer import (
    build_resources_from_files, annotate_endpoints_for_row, LLMClientProtocol
)
from processing.reporting_layer import annotate_reporting_for_row
from processing.classify_layer import annotate_classification_for_row

# Optional: only used if OPENAI_API_KEY is set; otherwise we stay rules-only
try:
    from llm.openai_client import OpenAIClient  # provides classify_endpoint / classify_nm_ei
except Exception:  # pragma: no cover
    OpenAIClient = None  # type: ignore


INPUT_CSV = os.path.join(RAW_DIR, "breast_cancer_trials_unique.csv")
SYN_PATH  = os.path.join(PROCESSED_DIR, "endpoint_synonyms.json")
DICT_JSON = os.path.join(PROCESSED_DIR, "endpoint_dictionary.json")
DICT_CSV  = os.path.join(PROCESSED_DIR, "endpoint_dictionary.csv")

OUT_PARQUET = os.path.join(PROCESSED_DIR, "breast_cancer_trials_enhanced.parquet")
OUT_CSV     = os.path.join(PROCESSED_DIR, "breast_cancer_trials_enhanced.csv")
OUT_JSONL   = os.path.join(PROCESSED_DIR, "breast_cancer_trials_enhanced.jsonl")

CHECKPOINT_DIR     = os.path.join(INTERIM_DIR, "checkpoints")
CHECKPOINT_PARQUET = os.path.join(CHECKPOINT_DIR, "enhanced_ckpt.parquet")
CHECKPOINT_CSV     = os.path.join(CHECKPOINT_DIR, "enhanced_ckpt.csv")

ROW_FLUSH_EVERY = 100  # write checkpoint every N rows


def _init_llm_client() -> Optional[LLMClientProtocol]:
    if OpenAIClient is None:
        return None
    try:
        return OpenAIClient()
    except Exception:
        return None


def run() -> pd.DataFrame:
    ensure_dirs()

    df = read_csv_safely(INPUT_CSV)
    if df.empty:
        # still produce empty outputs
        save_csv_parquet_dual(df, OUT_CSV, OUT_PARQUET)
        save_jsonl([], OUT_JSONL)
        return df

    # Load endpoint resources
    dict_path = None
    if os.path.exists(DICT_JSON):
        dict_path = DICT_JSON
    elif os.path.exists(DICT_CSV):
        dict_path = DICT_CSV

    resources = build_resources_from_files(
        endpoint_synonyms_path=SYN_PATH,
        endpoint_dictionary_path=dict_path,
    )

    llm_client = _init_llm_client()

    rows_out: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        rowd: Dict[str, Any] = row.to_dict()

        # Layer A — planned endpoints → 15 flags
        ep_cols = annotate_endpoints_for_row(rowd, resources, client=llm_client)
        rowd.update(ep_cols)

        # Layer B — reporting gap + CONSORT 7
        rep_cols = annotate_reporting_for_row(rowd, resources, client=llm_client)
        rowd.update(rep_cols)

        # Layer C — NM/EI/HER2/BRCAm
        cls_cols = annotate_classification_for_row(rowd, client=llm_client)
        rowd.update(cls_cols)

        rows_out.append(rowd)

        # periodic checkpoint (row-by-row completeness guaranteed)
        if (i + 1) % ROW_FLUSH_EVERY == 0:
            cdf = pd.DataFrame(rows_out)
            save_checkpoint_dual(cdf, CHECKPOINT_CSV, CHECKPOINT_PARQUET)

    final_df = pd.DataFrame(rows_out)

    # final save (csv + parquet + jsonl)
    save_csv_parquet_dual(final_df, OUT_CSV, OUT_PARQUET)
    save_jsonl(final_df.to_dict(orient="records"), OUT_JSONL)

    return final_df


if __name__ == "__main__":
    run()
