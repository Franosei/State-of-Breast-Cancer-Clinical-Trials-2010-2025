# processing/endpoints_layer.py
from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Any

CANONICAL_ENDPOINTS: List[str] = [
    "Pathologic Complete Response",
    "Event-Free Survival",
    "Disease-Free Survival",
    "Overall Survival",
    "Best Overall Response",
    "Duration of Response",
    "Progression-Free Survival",
    "Overall Response",
    "Target Response",
    "Non-target Response",
    "Symptomatic Deterioration",
    "Disease Recurrence",
    "Pathologic Complete Response (pCR)",
    "Objective Response Rate",
    "Time to Progression",
]

def canonical_to_col(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_")

CANONICAL_COLS: List[str] = [canonical_to_col(n) for n in CANONICAL_ENDPOINTS]

@dataclass(frozen=True)
class EndpointResources:
    synonyms: Dict[str, str]
    canonical: Tuple[str, ...] = tuple(CANONICAL_ENDPOINTS)

def load_synonyms(text: str | bytes) -> Dict[str, str]:
    obj = json.loads(text)
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip():
            out[k.strip().lower()] = v.strip()
    return out

_PUNCT_RE = re.compile(r"[\s\-/_,.:;(){}\[\]]+")
def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_outcome_cell(cell: Any) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell if isinstance(x, (str, int, float))]
    if isinstance(cell, str):
        t = cell.strip()
        if not t:
            return []
        if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
            try:
                obj = json.loads(t)
                if isinstance(obj, list):
                    return [str(x) for x in obj if isinstance(x, (str, int, float))]
            except Exception:
                pass
        return [t]
    return [str(cell)]

def rule_match_canonical(text: str, synonyms: Dict[str, str], canonical: Iterable[str]) -> Optional[str]:
    if not text:
        return None
    t = norm_text(text)
    for k, cname in synonyms.items():
        if k and k in t and cname in canonical:
            return cname
    for cname in canonical:
        if norm_text(cname) in t:
            return cname
    return None

class LLMClientProtocol:
    def classify_endpoint(self, outcome_text: str, label_set: List[str]) -> Optional[str]:
        raise NotImplementedError

def llm_map_endpoint(text: str, client: Optional[LLMClientProtocol], label_set: List[str]) -> Optional[str]:
    if not client or not text:
        return None
    try:
        return client.classify_endpoint(text, label_set)
    except Exception:
        return None

def annotate_endpoints_for_row(
    row: Dict[str, Any],
    resources: EndpointResources,
    client: Optional[LLMClientProtocol] = None,
) -> Dict[str, Any]:
    res: Dict[str, Any] = {canonical_to_col(c): 0 for c in resources.canonical}
    prim = parse_outcome_cell(row.get("planned_primary_outcomes"))
    sec  = parse_outcome_cell(row.get("planned_secondary_outcomes"))
    texts: List[str] = []
    for chunk in prim + sec:
        if not chunk:
            continue
        parts = [p.strip() for p in re.split(r"[;•\n\r]+", str(chunk)) if p.strip()]
        texts.extend(parts if parts else [str(chunk)])

    matched, seen = [], set()
    for txt in texts:
        cname = rule_match_canonical(txt, resources.synonyms, resources.canonical)
        if cname is None:
            cname = llm_map_endpoint(txt, client, list(resources.canonical))
        if cname and cname not in seen:
            seen.add(cname)
            matched.append(cname)
            res[canonical_to_col(cname)] = 1

    res["detected_endpoint_names"] = json.dumps(matched, ensure_ascii=False, separators=(",", ":"))
    res["planned_endpoint_count"] = len(matched)
    return res

def build_resources_from_files(endpoint_synonyms_path: str, endpoint_dictionary_path: Optional[str] = None) -> EndpointResources:
    with open(endpoint_synonyms_path, "r", encoding="utf-8") as f:
        synonyms = load_synonyms(f.read())
    canonical = tuple(CANONICAL_ENDPOINTS)
    # optional: allow dictionary override if it lists a subset of canonical names
    if endpoint_dictionary_path and endpoint_dictionary_path.lower().endswith(".csv"):
        try:
            import pandas as pd
            df = pd.read_csv(endpoint_dictionary_path)
            for col in ("endpoint", "name", "Endpoint", "Name", "label", "Label"):
                if col in df.columns:
                    vals = [str(x).strip() for x in df[col].dropna().tolist()]
                    keep = [v for v in vals if v in CANONICAL_ENDPOINTS]
                    if keep:
                        canonical = tuple(keep)
                    break
        except Exception:
            pass
    elif endpoint_dictionary_path and endpoint_dictionary_path.lower().endswith(".json"):
        try:
            obj = json.load(open(endpoint_dictionary_path, "r", encoding="utf-8"))
            vals = []
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        v = it.get("endpoint") or it.get("name")
                        if isinstance(v, str):
                            vals.append(v.strip())
                    elif isinstance(it, str):
                        vals.append(it.strip())
            keep = [v for v in vals if v in CANONICAL_ENDPOINTS]
            if keep:
                canonical = tuple(keep)
        except Exception:
            pass
    return EndpointResources(synonyms=synonyms, canonical=canonical)
