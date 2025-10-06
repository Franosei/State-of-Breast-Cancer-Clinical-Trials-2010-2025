# processing/reporting_layer.py
from __future__ import annotations
import json, re
from typing import Dict, Any, List, Optional
from processing.endpoints_layer import (
    EndpointResources, canonical_to_col, norm_text, LLMClientProtocol, rule_match_canonical
)

def _json_or_none(txt: Any):
    if not isinstance(txt, str) or not txt.strip():
        return None
    t = txt.strip()
    try:
        return json.loads(t)
    except Exception:
        return None

def _collect_reported_endpoint_titles(results_outcome_measures: Any) -> List[str]:
    out: List[str] = []
    obj = results_outcome_measures
    if isinstance(results_outcome_measures, str):
        obj = _json_or_none(results_outcome_measures)
    if not obj:
        return out
    # Our extractor saves a list of outcome dicts with "title" and optional groups.
    for m in obj if isinstance(obj, list) else []:
        t = m.get("title")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out

def _map_reported_to_canonical(titles: List[str], resources: EndpointResources, client: Optional[LLMClientProtocol]) -> List[str]:
    mapped: List[str] = []
    seen = set()
    for t in titles:
        cname = rule_match_canonical(t, resources.synonyms, resources.canonical)
        if cname is None:
            cname = client.classify_endpoint(t, list(resources.canonical)) if client else None
        if cname and cname not in seen:
            seen.add(cname)
            mapped.append(cname)
    return mapped

def annotate_reporting_for_row(
    row: Dict[str, Any],
    resources: EndpointResources,
    client: Optional[LLMClientProtocol] = None,
) -> Dict[str, Any]:
    """Compute the reporting gap flag and CONSORT 7 results flags."""
    res: Dict[str, Any] = {}

    # planned canonical set (from Layer A flags already on row)
    planned = set()
    for cname in resources.canonical:
        col = canonical_to_col(cname)
        if int(row.get(col, 0)) == 1:
            planned.add(cname)

    # reported canonical set
    titles = _collect_reported_endpoint_titles(row.get("results_outcome_measures"))
    reported = set(_map_reported_to_canonical(titles, resources, client))

    # Reporting gap flag
    res["reporting_gap_flag"] = 1 if any(p not in reported for p in planned) else 0

    # CONSORT Results (7 flags)
    # 13a/13b participant flow -> results_participant_flow non-empty structure
    res["consort_participant_flow"] = 1 if _json_or_none(row.get("results_participant_flow")) else 0

    # 14a/14b recruitment period -> infer from start_date and any completion date
    res["consort_recruitment"] = 1 if (row.get("start_date") and (row.get("primary_completion_date") or row.get("completion_date"))) else 0

    # 15 baseline data -> results_baseline non-empty
    res["consort_baseline"] = 1 if _json_or_none(row.get("results_baseline")) else 0

    # 16 numbers analyzed -> detect denominators/group counts in our compact structure
    res["consort_numbers_analyzed"] = 1 if _has_numbers_analyzed(row.get("results_outcome_measures")) else 0

    # 17a/17b outcomes & estimation -> any estimate + precision elements
    res["consort_outcomes_estimation"] = 1 if _has_estimates_with_precision(row.get("results_outcome_measures")) else 0

    # 18 ancillary analyses -> weak signal; check for subgroup/adjusted/exploratory keywords in measure titles (LLM optional)
    res["consort_ancillary_analyses"] = 1 if _has_ancillary_signals(titles, client) else 0

    # 19 harms -> results_adverse_events non-empty
    res["consort_harms"] = 1 if _json_or_none(row.get("results_adverse_events")) else 0

    # optional summary score
    res["consort_results_score"] = (
        res["consort_participant_flow"] + res["consort_recruitment"] + res["consort_baseline"] +
        res["consort_numbers_analyzed"] + res["consort_outcomes_estimation"] +
        res["consort_ancillary_analyses"] + res["consort_harms"]
    )

    return res

def _has_numbers_analyzed(results_outcome_measures: Any) -> bool:
    obj = results_outcome_measures
    if isinstance(obj, str):
        obj = _json_or_none(obj)
    if not isinstance(obj, list):
        return False
    for m in obj:
        # our compact structure places group measurements under classes/categories/measurements
        groups = 0
        for cls in m.get("classes", []) if isinstance(m.get("classes"), list) else []:
            for cat in cls.get("categories", []) if isinstance(cls.get("categories"), list) else []:
                for meas in cat.get("measurements", []) if isinstance(cat.get("measurements"), list) else []:
                    if "value" in meas:
                        groups += 1
        if groups > 0:
            return True
    return False

def _has_estimates_with_precision(results_outcome_measures: Any) -> bool:
    obj = results_outcome_measures
    if isinstance(obj, str):
        obj = _json_or_none(obj)
    if not isinstance(obj, list):
        return False
    for m in obj:
        has_est = bool(m.get("paramType"))
        has_prec = bool(m.get("dispersionType")) or _any_ci_limits(m)
        if has_est and has_prec:
            return True
    return False

def _any_ci_limits(m: Dict[str, Any]) -> bool:
    for cls in m.get("classes", []) if isinstance(m.get("classes"), list) else []:
        for cat in cls.get("categories", []) if isinstance(cls.get("categories"), list) else []:
            for meas in cat.get("measurements", []) if isinstance(cat.get("measurements"), list) else []:
                if any(k in meas for k in ("lowerLimit", "upperLimit")):
                    return True
    return False

_ANCILLARY_PATTERNS = re.compile(
    r"\b(subgroup|post[- ]hoc|exploratory|adjusted|interaction|sensitivity)\b", re.IGNORECASE
)

def _has_ancillary_signals(titles: List[str], client: Optional[LLMClientProtocol]) -> bool:
    for t in titles:
        if _ANCILLARY_PATTERNS.search(t or ""):
            return True
    # optional: ask LLM to judge if any title implies ancillary analysis; keep off by default
    return False
