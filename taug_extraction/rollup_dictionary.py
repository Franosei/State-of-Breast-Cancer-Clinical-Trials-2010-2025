# taug_extraction/rollup_dictionary.py
"""
Merge per-page TAUG-BrCa endpoint JSONs into a canonical endpoint dictionary.

Inputs (created by page_extractor.py):
  data/interim/taug_pages/page_001.json, page_002.json, ...

Outputs:
  data/processed/endpoint_pages.parquet    # per-page flattened candidates
  data/processed/endpoint_dictionary.json  # canonical dictionary
  data/processed/endpoint_dictionary.csv   # tabular version
  data/processed/endpoint_synonyms.json    # alias -> canonical map

Usage (from project root):
  python -m taug_extraction.rollup_dictionary
  python -m taug_extraction.rollup_dictionary --use-llm --model gpt-4.1
  python -m taug_extraction.rollup_dictionary --pages-dir data/interim/taug_pages --out-dir data/processed
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set, Iterable, DefaultDict
from collections import defaultdict

import pandas as pd

from utils.io import ensure_dirs, INTERIM_DIR, PROCESSED_DIR, save_parquet, save_json, save_csv, append_run_log
from utils.helpers import normalize_ws, slugify

# LLM is optional; only used if --use-llm is provided.
try:
    from llm.openai_client import OpenAIClient
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False


# -----------------------------
# helpers
# -----------------------------

CANONICAL_NAMES = {
    # common oncology endpoints -> preferred canonical labels
    "overall survival": "Overall Survival",
    "progression-free survival": "Progression-Free Survival",
    "disease-free survival": "Disease-Free Survival",
    "event-free survival": "Event-Free Survival",
    "objective response rate": "Objective Response Rate",
    "overall response rate": "Objective Response Rate",
    "best overall response": "Best Overall Response",
    "duration of response": "Duration of Response",
    "time to progression": "Time to Progression",
    "pathologic complete response": "Pathologic Complete Response",
    "pcr": "Pathologic Complete Response",
    "clinical benefit rate": "Clinical Benefit Rate",
}

ALIAS_SEEDS: Dict[str, List[str]] = {
    "Overall Survival": ["OS", "overall survival"],
    "Progression-Free Survival": ["PFS", "progression-free survival"],
    "Disease-Free Survival": ["DFS", "disease-free survival"],
    "Event-Free Survival": ["EFS", "event-free survival"],
    "Objective Response Rate": ["ORR", "objective response rate", "overall response rate"],
    "Best Overall Response": ["BOR", "best overall response"],
    "Duration of Response": ["DOR", "duration of response"],
    "Time to Progression": ["TTP", "time to progression"],
    "Pathologic Complete Response": ["pCR", "pathological complete response", "pathologic complete response"],
    "Clinical Benefit Rate": ["CBR", "clinical benefit rate"],
}

def _default_pages_dir() -> str:
    return os.path.join(INTERIM_DIR, "taug_pages")

def _default_out_dir() -> str:
    return PROCESSED_DIR

def _norm(s: Optional[str]) -> str:
    """
    Normalise strings for grouping: lowercase, strip punctuation/extra spaces.
    """
    s = normalize_ws(s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canonicalise_name(name: str) -> str:
    """
    Map to a preferred canonical name where known.
    """
    n = _norm(name)
    return CANONICAL_NAMES.get(n, name.strip()) or name

def _collect_text(*vals: Iterable[Optional[str]]) -> str:
    parts: List[str] = []
    for seq in vals:
        for v in (seq or []):
            if v:
                v = normalize_ws(v)
                if v and v not in parts:
                    parts.append(v)
    return " ".join(parts).strip()

def _union_lists(*lists: Iterable[Optional[List[str]]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for lst in lists:
        for v in (lst or []):
            v_norm = _norm(v)
            if v_norm and v_norm not in seen:
                seen.add(v_norm)
                out.append(v.strip())
    return out

def _page_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]  # page_001

def _iter_page_jsons(pages_dir: str) -> List[str]:
    files = [f for f in os.listdir(pages_dir) if re.match(r"^page_\d{3}\.json$", f, re.I)]
    files.sort()
    return [os.path.join(pages_dir, f) for f in files]


# -----------------------------
# LLM tie-break (optional)
# -----------------------------

LLM_CANON_PROMPT = """
You are harmonising oncology endpoint names for a CDISC dictionary.

Given a list of possibly synonymous endpoint labels and their snippets, choose ONE canonical name
(prefer the well-known clinical term) and list all aliases.

Return STRICT JSON:
{
  "canonical_name": "<string>",
  "aliases": ["<string>", "..."]
}

Use widely accepted oncology names (e.g., Overall Survival, Progression-Free Survival, etc.).
"""

def _llm_choose_canonical(model: str, labels: List[str], snippets: List[str]) -> Tuple[str, List[str]]:
    if not _HAS_LLM:
        raise RuntimeError("OpenAI client not available; run without --use-llm or add llm/openai_client.py")

    system = "You are a precise clinical data harmoniser. Output valid JSON only."
    user = f"""
Labels:
{json.dumps(labels, ensure_ascii=False, indent=2)}

Snippets (context):
{json.dumps(snippets[:5], ensure_ascii=False, indent=2)}

Task:
{LLM_CANON_PROMPT}
"""
    client = OpenAIClient(model=model)
    raw = client.run(system_message=system, user_prompt=user)
    try:
        payload = json.loads(raw)
        cname = payload.get("canonical_name") or labels[0]
        aliases = payload.get("aliases") or []
        return cname, aliases
    except Exception:
        return labels[0], list(dict.fromkeys(labels[1:]))


# -----------------------------
# core rollup
# -----------------------------

def _flatten_pages(pages_dir: str) -> pd.DataFrame:
    """
    Flatten page JSONs into a per-candidate DataFrame.
    Columns:
      page_id, endpoint_name, synonyms, definition, measurement, time_window,
      assessment_rule, population, biomarker_related, cdisc_domains, cdisc_variables,
      estimand_notes, quality_flags, confidence, page_summary
    """
    records: List[Dict[str, Any]] = []
    for jp in _iter_page_jsons(pages_dir):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {jp}: {e}")
            continue

        page_id = _page_id_from_path(jp)
        page_summary = data.get("page_summary", "")

        for cand in data.get("endpoint_candidates", []) or []:
            rec = {
                "page_id": page_id,
                "endpoint_name": (cand.get("endpoint_name") or "").strip(),
                "synonyms": cand.get("synonyms") or [],
                "definition": cand.get("definition"),
                "measurement": cand.get("measurement"),
                "time_window": cand.get("time_window"),
                "assessment_rule": cand.get("assessment_rule"),
                "population": cand.get("population"),
                "biomarker_related": cand.get("biomarker_related"),
                "cdisc_domains": cand.get("cdisc_domains") or [],
                "cdisc_variables": cand.get("cdisc_variables") or [],
                "estimand_notes": cand.get("estimand_notes"),
                "quality_flags": cand.get("quality_flags") or [],
                "confidence": cand.get("confidence"),
                "page_summary": page_summary,
            }
            if rec["endpoint_name"]:
                records.append(rec)

    df = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=[
        "page_id","endpoint_name","synonyms","definition","measurement","time_window",
        "assessment_rule","population","biomarker_related","cdisc_domains","cdisc_variables",
        "estimand_notes","quality_flags","confidence","page_summary"
    ])
    return df


def _seed_aliases(canonical: str) -> Set[str]:
    seeds = set(ALIAS_SEEDS.get(canonical, []))
    seeds.add(canonical)
    return {_norm(s) for s in seeds}


def _group_candidates(df: pd.DataFrame, use_llm: bool = False, model: str = "gpt-4.1") -> Dict[str, Dict[str, Any]]:
    """
    Group endpoint candidates into canonical entries.
    Returns dict: canonical_name -> merged_entry
    """
    if df.empty:
        return {}

    # Build initial buckets by normalised endpoint name
    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        raw_name = str(row["endpoint_name"])
        canon_guess = _canonicalise_name(raw_name)
        key = _norm(canon_guess)
        buckets[key].append(row.to_dict())

    # Merge buckets; where names are close variants or aliases, we may merge further
    merged: Dict[str, Dict[str, Any]] = {}

    for key, items in buckets.items():
        # gather label variants and snippets
        labels = list(dict.fromkeys([_canonicalise_name(it["endpoint_name"]) for it in items if it.get("endpoint_name")]))
        snippets = [normalize_ws(_collect_text([it.get("definition")], [it.get("measurement")], [it.get("assessment_rule")])) for it in items]

        # choose canonical
        chosen = labels[0]
        aliases: List[str] = [l for l in labels[1:] if l and l.lower() != chosen.lower()]

        # optional LLM tie-break
        if use_llm and len(labels) > 1:
            try:
                chosen_llm, aliases_llm = _llm_choose_canonical(model=model, labels=labels, snippets=snippets)
                chosen = chosen_llm or chosen
                if aliases_llm:
                    aliases = list(dict.fromkeys(aliases + aliases_llm))
            except Exception as e:
                print(f"[WARN] LLM canonicalisation failed for {labels}: {e}")

        # seed with common aliases
        aliases = list(dict.fromkeys(aliases + list(ALIAS_SEEDS.get(chosen, []))))

        # merge fields
        syns: List[str] = []
        defs: List[str] = []
        meas: List[str] = []
        tws: List[str] = []
        rules: List[str] = []
        pops: List[str] = []
        ests: List[str] = []
        qflags: List[str] = []
        domains: List[str] = []
        vars_: List[str] = []
        biomks: List[str] = []
        confs: List[float] = []

        for it in items:
            syns.extend(it.get("synonyms") or [])
            if it.get("definition"): defs.append(str(it["definition"]))
            if it.get("measurement"): meas.append(str(it["measurement"]))
            if it.get("time_window"): tws.append(str(it["time_window"]))
            if it.get("assessment_rule"): rules.append(str(it["assessment_rule"]))
            if it.get("population"): pops.append(str(it["population"]))
            if it.get("estimand_notes"): ests.append(str(it["estimand_notes"]))
            if it.get("quality_flags"): qflags.extend(it.get("quality_flags") or [])
            domains.extend(it.get("cdisc_domains") or [])
            vars_.extend(it.get("cdisc_variables") or [])
            if it.get("biomarker_related"): biomks.append(str(it["biomarker_related"]))
            if it.get("confidence") is not None:
                try:
                    confs.append(float(it["confidence"]))
                except Exception:
                    pass

        # union and concise summaries
        syns = _union_lists(syns, aliases)
        domains = _union_lists(domains)
        vars_ = _union_lists(vars_)
        qflags = _union_lists(qflags)

        definition = normalize_ws(" ".join(dict.fromkeys(defs)))[:1200] if defs else None
        measurement = normalize_ws(" ".join(dict.fromkeys(meas)))[:800] if meas else None
        time_window = normalize_ws(" ".join(dict.fromkeys(tws)))[:400] if tws else None
        assessment_rule = normalize_ws(" ".join(dict.fromkeys(rules)))[:1200] if rules else None
        population = normalize_ws(" ".join(dict.fromkeys(pops)))[:600] if pops else None
        estimand_notes = normalize_ws(" ".join(dict.fromkeys(ests)))[:800] if ests else None
        biomarker_related = None
        if biomks:
            # pick the most common non-empty biomarker tag
            s = pd.Series([b for b in biomks if b and b.lower() != "none"])
            if not s.empty:
                biomarker_related = s.value_counts().idxmax()

        confidence = float(pd.Series(confs).mean()) if confs else None

        merged[chosen] = {
            "endpoint_name": chosen,
            "synonyms": syns,
            "definition": definition,
            "measurement": measurement,
            "time_window": time_window,
            "assessment_rule": assessment_rule,
            "population": population,
            "biomarker_related": biomarker_related or "None",
            "cdisc_domains": domains or ["Other/Unknown"],
            "cdisc_variables": vars_,
            "estimand_notes": estimand_notes,
            "quality_flags": qflags,
            "confidence_mean": confidence,
        }

    return merged


def _alias_map(canon_dict: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Build alias -> canonical name mapping.
    """
    alias_to_canon: Dict[str, str] = {}
    for cname, obj in canon_dict.items():
        alias_to_canon[_norm(cname)] = cname
        for a in obj.get("synonyms", []) or []:
            alias_to_canon[_norm(a)] = cname
    return alias_to_canon


def rollup_dictionary(
    pages_dir: str,
    out_dir: str,
    use_llm: bool = False,
    model: str = "gpt-4.1",
) -> Tuple[str, str, str, str]:
    """
    Orchestrate:
      1) Flatten pages -> endpoint_pages.parquet
      2) Group into canonical dictionary
      3) Write JSON/CSV dictionary and alias map
    Returns tuple of output paths.
    """
    ensure_dirs()
    os.makedirs(out_dir, exist_ok=True)

    # 1) flatten page candidates
    df_pages = _flatten_pages(pages_dir)
    pages_parquet = os.path.join(out_dir, "endpoint_pages.parquet")
    save_parquet(df_pages, pages_parquet)

    # 2) group into canonical dict
    canon = _group_candidates(df_pages, use_llm=use_llm, model=model)

    # 3) write dictionary + csv
    dict_json_path = os.path.join(out_dir, "endpoint_dictionary.json")
    dict_csv_path = os.path.join(out_dir, "endpoint_dictionary.csv")
    syn_json_path = os.path.join(out_dir, "endpoint_synonyms.json")

    # JSON
    dict_list = list(canon.values())
    save_json(dict_list, dict_json_path)

    # CSV (flatten lists for readability)
    def _flatten(v):
        if isinstance(v, list):
            return "; ".join(map(str, v))
        return v
    df_dict = pd.DataFrame([{k: _flatten(v) for k, v in row.items()} for row in dict_list])
    save_csv(df_dict, dict_csv_path)

    # alias map
    alias_map = _alias_map(canon)
    save_json(alias_map, syn_json_path)

    append_run_log("taug_rollup_dictionary", {
        "pages": len(df_pages),
        "unique_endpoints": len(canon),
        "use_llm": use_llm,
        "model": model if use_llm else None,
        "outputs": {
            "pages_parquet": pages_parquet,
            "dictionary_json": dict_json_path,
            "dictionary_csv": dict_csv_path,
            "synonyms_json": syn_json_path,
        }
    })

    return pages_parquet, dict_json_path, dict_csv_path, syn_json_path


def main():
    parser = argparse.ArgumentParser(description="Merge TAUG page JSONs into a canonical endpoint dictionary.")
    parser.add_argument("--pages-dir", type=str, default=_default_pages_dir(),
                        help="Directory with page_*.json files (default: data/interim/taug_pages)")
    parser.add_argument("--out-dir", type=str, default=_default_out_dir(),
                        help="Directory to write processed outputs (default: data/processed)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM to break ties and harmonise ambiguous names")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model name when --use-llm is set (default: gpt-4.1)")
    args = parser.parse_args()

    rollup_dictionary(
        pages_dir=args.pages_dir,
        out_dir=args.out_dir,
        use_llm=args.use_llm,
        model=args.model,
    )


if __name__ == "__main__":
    main()
