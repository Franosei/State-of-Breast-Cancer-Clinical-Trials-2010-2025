# taug_extraction/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Biomarker = Literal["HER2", "BRCAm", "TNBC", "HR+", "None"]

class EndpointModel(BaseModel):
    endpoint_name: str
    synonyms: Optional[List[str]] = None
    definition: Optional[str] = None
    measurement: Optional[str] = None
    time_window: Optional[str] = None
    assessment_rule: Optional[str] = None
    population: Optional[str] = None
    biomarker_related: Optional[Biomarker] = "None"
    cdisc_domains: Optional[List[str]] = None
    cdisc_variables: Optional[List[str]] = None
    estimand_notes: Optional[str] = None
    quality_flags: Optional[List[str]] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

class EndpointCandidates(BaseModel):
    page_summary: str
    endpoint_candidates: List[EndpointModel] = []
