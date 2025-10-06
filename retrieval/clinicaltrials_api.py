import os
import time
import json
import requests
import pandas as pd
from typing import Optional, List, Dict, Any

from config.settings import CLINICAL_TRIALS_BASE_URL, RAW_DATA_DIR


# HTTP helper
def _http_get_json(
    url: str,
    params: Dict[str, Any],
    retries: int = 4,
    backoff: float = 1.6,
    timeout: int = 90,
    user_agent: str = "BayezianBreastTrials/1.0",
) -> Dict[str, Any]:
    """GET JSON with retry on transient errors; raise on others."""
    attempt = 0
    while True:
        resp = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON from CTG: {e}. url={resp.url}")
        if resp.status_code in (429, 500, 502, 503, 504):
            attempt += 1
            if attempt > retries:
                raise requests.HTTPError(
                    f"Exceeded retries; last status={resp.status_code}; body[:300]={resp.text[:300]}",
                    response=resp,
                )
            ra = resp.headers.get("Retry-After")
            delay = float(ra) if ra and ra.isdigit() else (backoff ** attempt)
            time.sleep(delay)
            continue
        if resp.status_code == 404:
            raise requests.HTTPError(f"404 from CTG. URL={resp.url}", response=resp)
        resp.raise_for_status()


# Small JSON helpers
def _jdump(obj: Any) -> str:
    """CSV-safe JSON serialization (None -> empty string)."""
    if obj is None:
        return ""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# Results extractors (compact, CSV-safe)
def _extract_outcome_measures(results: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    om = (results or {}).get("outcomeMeasuresModule") or {}
    measures = om.get("outcomeMeasures") or []
    if not measures:
        legacy_primary = (results or {}).get("primaryOutcomes") or []
        legacy_secondary = (results or {}).get("secondaryOutcomes") or []
        merged = []
        for block, otype in ((legacy_primary, "PRIMARY"), (legacy_secondary, "SECONDARY")):
            for m in block:
                title = m.get("title")
                if title:
                    merged.append({"type": otype, "title": title})
        return merged or None

    out: List[Dict[str, Any]] = []
    for m in measures:
        group_title_map = {g.get("id"): (g.get("title") or g.get("id")) for g in (m.get("groups") or [])}
        unit = m.get("unitOfMeasure")
        entry = {
            "type": m.get("type"),
            "title": m.get("title"),
            "timeFrame": m.get("timeFrame"),
            "unit": unit,
            "paramType": m.get("paramType"),
            "dispersionType": m.get("dispersionType"),
            "groups": [],
        }
        for cls in (m.get("classes") or []):
            for cat in (cls.get("categories") or []):
                for meas in (cat.get("measurements") or []):
                    entry["groups"].append({
                        "group": group_title_map.get(meas.get("groupId"), meas.get("groupId")),
                        "value": meas.get("value"),
                        "spread": meas.get("spread"),
                        "lower": meas.get("lowerLimit"),
                        "upper": meas.get("upperLimit"),
                    })
        out.append(entry)
    return out or None


def _extract_participant_flow(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pf = (results or {}).get("participantFlowModule") or {}
    periods = pf.get("periods") or []
    groups = pf.get("groups") or []
    if not periods and not groups:
        return None

    gid2title = {g.get("id"): (g.get("title") or g.get("id")) for g in groups}
    compact_periods: List[Dict[str, Any]] = []
    for p in periods:
        pentry = {"title": p.get("title"), "milestones": [], "dropWithdraws": []}
        for ms in (p.get("milestones") or []):
            pentry["milestones"].append({
                "type": ms.get("type"),
                "achievements": [
                    {"group": gid2title.get(a.get("groupId"), a.get("groupId")), "n": a.get("numSubjects")}
                    for a in (ms.get("achievements") or [])
                ],
            })
        for dw in (p.get("dropWithdraws") or []):
            pentry["dropWithdraws"].append({
                "type": dw.get("type"),
                "reasons": [
                    {"group": gid2title.get(r.get("groupId"), r.get("groupId")), "n": r.get("numSubjects")}
                    for r in (dw.get("reasons") or [])
                ],
            })
        compact_periods.append(pentry)
    return {"periods": compact_periods} if compact_periods else None


def _extract_baseline(results: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    bl = (results or {}).get("baselineCharacteristicsModule") or {}
    if not bl:
        return None
    gid2title = {g.get("id"): (g.get("title") or g.get("id")) for g in (bl.get("groups") or [])}
    out: List[Dict[str, Any]] = []
    for m in (bl.get("measures") or []):
        entry = {
            "title": m.get("title"),
            "unit": m.get("unitOfMeasure"),
            "paramType": m.get("paramType"),
            "dispersionType": m.get("dispersionType"),
            "groups": [],
        }
        for cls in (m.get("classes") or []):
            for cat in (cls.get("categories") or []):
                for meas in (cat.get("measurements") or []):
                    entry["groups"].append({
                        "group": gid2title.get(meas.get("groupId"), meas.get("groupId")),
                        "value": meas.get("value"),
                        "spread": meas.get("spread"),
                    })
        out.append(entry)
    return out or None


def _extract_adverse_events(results: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    ae = (results or {}).get("adverseEventsModule") or {}
    egs = ae.get("eventGroups") or []
    if not egs:
        return None
    out: List[Dict[str, Any]] = []
    for g in egs:
        out.append({
            "group": g.get("title") or g.get("id"),
            "serious_affected": g.get("seriousNumAffected"),
            "serious_at_risk": g.get("seriousNumAtRisk"),
            "other_affected": g.get("otherNumAffected"),
            "other_at_risk": g.get("otherNumAtRisk"),
        })
    return out


# Field & site extractors
def _extract_trial_fields(study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    protocol = study.get("protocolSection", {}) or {}
    ident = protocol.get("identificationModule", {}) or {}
    status = protocol.get("statusModule", {}) or {}
    sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}
    outcomes = protocol.get("outcomesModule", {}) or {}
    design = protocol.get("designModule", {}) or {}

    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    phases_val = design.get("phases")
    if isinstance(phases_val, list):
        phase = "; ".join([p for p in phases_val if isinstance(p, str) and p.strip()])
    elif isinstance(phases_val, str):
        phase = phases_val.strip()
    else:
        phase = None

    planned_primary = [o.get("measure") for o in outcomes.get("primaryOutcomes", [])]
    planned_secondary = [o.get("measure") for o in outcomes.get("secondaryOutcomes", [])]

    results = study.get("resultsSection") or {}

    # Determine sponsor class and result flag
    sponsor_class = (sponsor.get("leadSponsor") or {}).get("class")
    has_results = bool(results)

    # Results (compact, JSON-ready)
    results_outcomes = _extract_outcome_measures(results)
    results_flow = _extract_participant_flow(results)
    results_baseline = _extract_baseline(results)
    results_ae = _extract_adverse_events(results)

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "overall_status": status.get("overallStatus"),
        "phase": phase,
        "sponsor": (sponsor.get("leadSponsor") or {}).get("name"),
        "sponsor_class": sponsor_class,
        "has_results": has_results,
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "primary_completion_date": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "completion_date": (status.get("completionDateStruct") or {}).get("date"),
        "first_posted_date": (status.get("studyFirstPostDateStruct") or {}).get("date"),
        "last_update_date": (status.get("lastUpdatePostDateStruct") or {}).get("date"),
        "planned_primary_outcomes": planned_primary,
        "planned_secondary_outcomes": planned_secondary,
        "results_outcome_measures": _jdump(results_outcomes),
        "results_participant_flow": _jdump(results_flow),
        "results_baseline": _jdump(results_baseline),
        "results_adverse_events": _jdump(results_ae),
    }


def _extract_sites(study: Dict[str, Any]) -> List[Dict[str, Any]]:
    protocol = study.get("protocolSection", {}) or {}
    clm = protocol.get("contactsLocationsModule") or {}
    lm = protocol.get("locationsModule") or {}
    locations = clm.get("locations") or lm.get("locations") or []
    if not isinstance(locations, list):
        return []

    rows: List[Dict[str, Any]] = []
    for loc in locations:
        facility = loc.get("facility")
        facility_name = None
        city = loc.get("city")
        state = loc.get("state")
        postal = loc.get("zip") or loc.get("postalCode")
        country = loc.get("country")
        site_status = loc.get("status") or loc.get("recruitmentStatus")

        if isinstance(facility, dict):
            facility_name = facility.get("name") or facility.get("facilityName")
            addr = facility.get("address") or {}
            city = city or addr.get("city")
            state = state or addr.get("state")
            postal = postal or addr.get("zip") or addr.get("postalCode")
            country = country or addr.get("country")
        elif isinstance(facility, str):
            facility_name = facility
        else:
            facility_name = loc.get("name") or loc.get("facilityName")

        rows.append({
            "facility": facility_name,
            "city": city,
            "state": state,
            "postal_code": postal,
            "country": country,
            "site_status": site_status,
        })
    return rows


# Main fetch (ONE CSV, UNIQUE nct_id) 
def fetch_breast_cancer_trials(
    start_year: Optional[int] = 2010,
    end_year: Optional[int] = 2025,
    max_records: int = 2_000_000,
    page_size: int = 200,
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    if not CLINICAL_TRIALS_BASE_URL:
        raise RuntimeError(
            "CLINICAL_TRIALS_BASE_URL is not set. Put it in .env as https://clinicaltrials.gov/api/v2/studies"
        )

    if save_csv_path is None:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        save_csv_path = os.path.join(RAW_DATA_DIR, "breast_cancer_trials_unique.csv")

    term = "breast cancer"
    trials: Dict[str, Dict[str, Any]] = {}
    trial_sites: Dict[str, List[Dict[str, Any]]] = {}
    nct_order: List[str] = []

    page_token: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"query.cond": term, "pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        data = _http_get_json(CLINICAL_TRIALS_BASE_URL, params=params)
        studies = data.get("studies", []) or []
        if not studies:
            break

        for study in studies:
            trial = _extract_trial_fields(study)
            if not trial:
                continue

            nct_id = trial["nct_id"]
            if nct_id not in trials:
                trials[nct_id] = trial
                trial_sites[nct_id] = []
                nct_order.append(nct_id)
                if len(trials) >= max_records:
                    break

            sites = _extract_sites(study)
            if sites:
                trial_sites[nct_id].extend(sites)

        if len(trials) >= max_records:
            break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    rows: List[Dict[str, Any]] = []
    for nct_id in nct_order:
        t = trials[nct_id]
        sites = trial_sites.get(nct_id, [])

        if sites:
            seen = set()
            deduped = []
            for s in sites:
                key = (s.get("facility"), s.get("city"), s.get("state"),
                       s.get("postal_code"), s.get("country"), s.get("site_status"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(s)
            sites = deduped

        site_count = len(sites)
        countries = sorted({s.get("country") for s in sites if s.get("country")})
        site_countries = "; ".join(countries) if countries else None

        if sites:
            items = []
            for s in sites:
                fac = s.get("facility") or ""
                city = s.get("city") or ""
                country = s.get("country") or ""
                parts = [p for p in [city, country] if p]
                loc = f"{fac} [{', '.join(parts)}]" if parts else fac
                items.append(loc)
            sites_compact = " | ".join(items)
        else:
            sites_compact = None

        row = {
            **t,
            "site_count": site_count,
            "site_countries": site_countries,
            "sites_compact": sites_compact,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_csv_path, index=False)

    if df.empty:
        return df

    df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    if start_year is not None:
        df = df[df["start_year"] >= int(start_year)]
    if end_year is not None:
        df = df[df["start_year"] <= int(end_year)]

    df = df.sort_values(["nct_id"]).drop_duplicates(subset=["nct_id"], keep="first")
    df = df.reset_index(drop=True)
    df.to_csv(save_csv_path, index=False)
    return df
