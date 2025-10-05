# taug_extraction/page_extractor.py
"""
Per-page LLM extraction of endpoint details from TAUG-BrCa pages.

Inputs:
  data/interim/taug_pages/page_001.txt ... (created by pdf_splitter.py)

Outputs (per page):
  data/interim/taug_pages/page_001.json   # structured endpoint candidates (LLM)

Usage (from project root):
  python -m taug_extraction.page_extractor
  python -m taug_extraction.page_extractor --pages-dir data/interim/taug_pages --limit 20 --overwrite
  python -m taug_extraction.page_extractor --model gpt-4.1
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Optional

from utils.io import ensure_dirs, INTERIM_DIR, append_run_log
from utils.helpers import normalize_ws, retry
from llm.openai_client import OpenAIClient

# Optional: validate with Pydantic models if you add taug_extraction/schemas.py
try:
    from taug_extraction.schemas import EndpointCandidates  # type: ignore
    _HAS_SCHEMAS = True
except Exception:
    EndpointCandidates = None  # type: ignore
    _HAS_SCHEMAS = False


def _default_pages_dir() -> str:
    return os.path.join(INTERIM_DIR, "taug_pages")


def _list_page_txt_files(pages_dir: str) -> List[str]:
    if not os.path.isdir(pages_dir):
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")
    files = [f for f in os.listdir(pages_dir) if re.match(r"^page_\d{3}\.txt$", f, re.I)]
    files.sort()
    return [os.path.join(pages_dir, f) for f in files]


def _out_json_path(txt_path: str) -> str:
    return re.sub(r"\.txt$", ".json", txt_path, flags=re.I)


def _build_system_message() -> str:
    # concise + strict: ask for JSON only, no prose.
    return normalize_ws("""
        You are an expert CDISC/oncology annotator. Extract endpoint information from the
        given TAUG-BrCa page. Return STRICT JSON only (no prose), following this schema:

        {
          "page_summary": "one-paragraph summary of the page, plain text",
          "endpoint_candidates": [
            {
              "endpoint_name": "canonical name (e.g., Overall Survival, PFS, DFS, EFS, ORR, DOR, BOR, pCR)",
              "synonyms": ["list", "of", "aliases"],
              "definition": "short definition in 1–3 sentences",
              "measurement": "what is measured (e.g., time from randomisation to death)",
              "time_window": "timeframe if specified",
              "assessment_rule": "how it is assessed/calculated; include censoring rules if present",
              "population": "target population or subpopulation if specified",
              "biomarker_related": "HER2 | BRCAm | TNBC | HR+ | None",
              "cdisc_domains": ["ADSL","ADTTE","ADRS","ADAE","ADLB","ADVS","ADQS", "Other/Unknown"],
              "cdisc_variables": ["examples like AVAL, CNSR, PARAMCD, PARAM, AVALC"],
              "estimand_notes": "any notes hinting at estimand attributes (population, variable, intercurrent events, summary measure)",
              "quality_flags": ["terms like 'surrogate endpoint', 'exploratory', 'legacy term'", "optional"],
              "confidence": 0.0
            }
          ]
        }

        Confidence is 0–1 reflecting your certainty the page actually defines or discusses that endpoint.

        OUTPUT RULES:
        - Output valid JSON only. No markdown, no extra text.
        - Omit empty fields rather than inventing content.
        - Use canonical endpoint names where obvious.
        - Keep arrays concise and relevant.
    """)


def _build_user_prompt(page_text: str, page_id: str) -> str:
    return f"""
Page: {page_id}

Content:
\"\"\"
{page_text}
\"\"\"

Task:
Extract endpoint candidates truly present on this page (avoid guessing).
If the page is purely narrative with no concrete endpoint details, return an empty "endpoint_candidates" list but still include a meaningful "page_summary".
"""


@retry(attempts=3, delay_sec=2, backoff=1.8)
def _call_llm(client: OpenAIClient, system: str, prompt: str) -> str:
    return client.run(system_message=system, user_prompt=prompt)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    # strip accidental code fences or BOMs
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.I).strip()
    t = re.sub(r"```$", "", t).strip()
    return json.loads(t)


def _validate_or_passthrough(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _HAS_SCHEMAS:
        return payload
    try:
        obj = EndpointCandidates.model_validate(payload)  # type: ignore
        return obj.model_dump()
    except Exception as e:
        # fall back to raw payload, but include a validator error note
        payload.setdefault("_validator_error", str(e))
        return payload


def extract_pages(
    pages_dir: str,
    model: str = "gpt-4.1",
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> List[str]:
    """
    Process page_*.txt files with the LLM and write JSON outputs per page.
    Returns list of written JSON paths.
    """
    ensure_dirs()
    client = OpenAIClient(model=model)

    txt_files = _list_page_txt_files(pages_dir)
    if limit is not None:
        txt_files = txt_files[: int(limit)]

    written: List[str] = []
    system_msg = _build_system_message()

    for idx, txt_path in enumerate(txt_files, start=1):
        page_id = os.path.splitext(os.path.basename(txt_path))[0]  # e.g., page_001
        out_path = _out_json_path(txt_path)

        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] {page_id} -> JSON exists.")
            written.append(out_path)
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            page_text = f.read().strip()

        if not page_text:
            print(f"[WARN] {page_id} is empty. Writing empty result.")
            empty_payload = {"page_summary": "", "endpoint_candidates": []}
            with open(out_path, "w", encoding="utf-8") as w:
                json.dump(empty_payload, w, indent=2, ensure_ascii=False)
            written.append(out_path)
            continue

        user_prompt = _build_user_prompt(page_text=page_text, page_id=page_id)

        try:
            raw = _call_llm(client, system=system_msg, prompt=user_prompt)
            payload = _safe_json_parse(raw)
            payload = _validate_or_passthrough(payload)
        except Exception as e:
            print(f"[ERROR] {page_id}: {e}")
            payload = {
                "page_summary": "",
                "endpoint_candidates": [],
                "_error": str(e),
            }

        with open(out_path, "w", encoding="utf-8") as w:
            json.dump(payload, w, indent=2, ensure_ascii=False)

        print(f"[OK] {page_id} -> {out_path}")
        written.append(out_path)

    append_run_log("taug_page_extraction", {"pages_dir": pages_dir, "written": len(written), "model": model})
    return written


def main():
    parser = argparse.ArgumentParser(description="Extract endpoint details per TAUG-BrCa page via LLM.")
    parser.add_argument("--pages-dir", type=str, default=_default_pages_dir(),
                        help="Directory containing page_*.txt files (default: data/interim/taug_pages)")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="OpenAI model name (default: gpt-4.1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pages")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing JSON files")
    args = parser.parse_args()

    extract_pages(
        pages_dir=args.pages_dir,
        model=args.model,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
