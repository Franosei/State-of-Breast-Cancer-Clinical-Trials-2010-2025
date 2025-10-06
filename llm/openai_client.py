# llm/openai_client.py
from __future__ import annotations
import os
from typing import Optional, List, Dict, Any

# If you prefer to swap providers, just replace this file's internals.
# Uses environment var OPENAI_API_KEY; if missing, ctor raises so pipeline falls back to rules.

class OpenAIClient:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # lazy import to avoid hard dep if unused
        from openai import OpenAI
        self._client = OpenAI(api_key=key)

    # Layer A fallback
    def classify_endpoint(self, outcome_text: str, label_set: List[str]) -> Optional[str]:
        sys = (
            "You map a clinical trial outcome string to EXACTLY ONE endpoint label from the provided list "
            "or return 'null' if none applies. Respond ONLY as JSON: {\"endpoint\": <label or null>}."
        )
        user = f"Labels: {label_set}\nOutcome: {outcome_text}"
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=50,
            )
            data = resp.choices[0].message
            import json
            obj = json.loads(data.content[0].text if isinstance(data.content, list) else data.content)
            val = obj.get("endpoint")
            return val if val in label_set else None
        except Exception:
            return None

    # Layer C adjudicator
    def classify_nm_ei(self, text: str) -> Dict[str, Any]:
        sys = (
            "Classify if a trial text indicates a New Medicine (NM) and/or an Extension of Indication (EI). "
            "Return JSON with keys NM_new_medicine (0/1), EI_extension_of_indication (0/1), reason."
        )
        user = f"Text: {text}"
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=80,
            )
            data = resp.choices[0].message
            import json
            obj = json.loads(data.content[0].text if isinstance(data.content, list) else data.content)
            out = {
                "NM_new_medicine": 1 if int(obj.get("NM_new_medicine", 0)) == 1 else 0,
                "EI_extension_of_indication": 1 if int(obj.get("EI_extension_of_indication", 0)) == 1 else 0,
                "reason": str(obj.get("reason", ""))[:300],
            }
            return out
        except Exception:
            return {"NM_new_medicine": 0, "EI_extension_of_indication": 0, "reason": ""}
