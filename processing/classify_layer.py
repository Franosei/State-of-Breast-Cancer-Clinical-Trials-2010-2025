# processing/classify_layer.py
from __future__ import annotations
import re, json
from typing import Dict, Any, Optional

HER2_RE = re.compile(r"\b(her2|erbb2|her2\+|her2-positive|her2\s*pos)\b", re.IGNORECASE)
BRCAM_RE = re.compile(r"\b(brca1|brca2|brcam|gbrca|brca[- ]mutat)\b", re.IGNORECASE)

FIRST_IN_HUMAN_RE = re.compile(r"\b(first[- ]in[- ]human|fi[hH])\b", re.IGNORECASE)
NOVEL_AGENT_RE = re.compile(r"\b(novel|investigational|new agent|new drug|new medicine|NME)\b", re.IGNORECASE)
PHASE1_SIGNAL_RE = re.compile(r"\bphase\s*1\b|\bphase\s*i\b", re.IGNORECASE)
EXTENSION_IND_RE = re.compile(r"\b(adjuvant|neoadjuvant|maintenance|new indication|extension|line of therapy|expansion)\b", re.IGNORECASE)

class LLMClientProtocol:
    def classify_nm_ei(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

def _text_pool(row: Dict[str, Any]) -> str:
    parts = []
    for k in ("title", "official_title"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    for k in ("planned_primary_outcomes", "planned_secondary_outcomes"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " | ".join(parts)[:2000]

def annotate_classification_for_row(
    row: Dict[str, Any],
    client: Optional[LLMClientProtocol] = None,
) -> Dict[str, Any]:
    text = _text_pool(row)

    # HER2 / BRCAm flags via high-precision rules
    her2 = 1 if HER2_RE.search(text) else 0
    brcam = 1 if BRCAM_RE.search(text) else 0

    # NM vs EI via rules; if ambiguous, LLM adjudication
    nm_rule = 1 if (FIRST_IN_HUMAN_RE.search(text) or NOVEL_AGENT_RE.search(text) or PHASE1_SIGNAL_RE.search(text)) else 0
    ei_rule = 1 if EXTENSION_IND_RE.search(text) else 0

    nm, ei, reason = nm_rule, ei_rule, ""

    if client and ((nm_rule == 0 and ei_rule == 0) or (nm_rule == 1 and ei_rule == 1)):
        # ambiguous; ask LLM
        try:
            out = client.classify_nm_ei(text)
            nm = 1 if int(out.get("NM_new_medicine", 0)) == 1 else 0
            ei = 1 if int(out.get("EI_extension_of_indication", 0)) == 1 else 0
            reason = str(out.get("reason", ""))[:300]
        except Exception:
            pass

    res = {
        "HER2_flag": her2,
        "BRCAm_flag": brcam,
        "NM_new_medicine": nm,
        "EI_extension_of_indication": ei,
    }
    if reason:
        res["nm_ei_explanation"] = reason
    return res
