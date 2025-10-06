# utils/io.py
import os
import json
import datetime as dt
from typing import Any, Dict, Optional, List
import pandas as pd

# existing content (unchanged)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")


def ensure_dirs() -> None:
    """Create all standard project data folders if they don't exist."""
    for p in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR, LOGS_DIR]:
        os.makedirs(p, exist_ok=True)


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a filesystem-friendly UTC timestamp string."""
    return dt.datetime.utcnow().strftime(fmt)


def with_timestamp(path: str, suffix: Optional[str] = None) -> str:
    """
    Insert a timestamp (and optional suffix) before the file extension.
    e.g. /a/b/file.csv -> /a/b/file_20250101_120000_suffix.csv
    """
    base, ext = os.path.splitext(path)
    ts = timestamp()
    parts = [base, ts]
    if suffix:
        parts.append(suffix)
    return f"{'_'.join(parts)}{ext}"


# -SV / JSON helpers
def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    ensure_dirs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_parquet(df: pd.DataFrame, path: str) -> str:
    ensure_dirs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(obj: Any, path: str, indent: int = 2) -> str:
    ensure_dirs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Minimal run log
def append_run_log(event: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Append a JSON line to data/outputs/logs/run.log with timestamp + event + meta.
    """
    ensure_dirs()
    os.makedirs(LOGS_DIR, exist_ok=True)
    line = {
        "ts_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        "meta": meta or {},
    }
    log_path = os.path.join(LOGS_DIR, "run.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return log_path


# --- NEW helpers below (used by processing pipeline) ---

def read_csv_safely(path: str) -> pd.DataFrame:
    """Return empty DataFrame if file missing; otherwise normal read_csv."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def save_jsonl(records: List[Dict[str, Any]], path: str) -> str:
    """Write a list of dicts as one-JSON-per-line file."""
    ensure_dirs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def save_csv_parquet_dual(df: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
    """Write both CSV and Parquet side-by-side."""
    save_csv(df, csv_path, index=False)
    save_parquet(df, parquet_path)


def save_checkpoint_dual(df: pd.DataFrame, csv_ckpt: str, parquet_ckpt: str) -> None:
    """Write periodic checkpoints (CSV+Parquet)."""
    ensure_dirs()
    os.makedirs(os.path.dirname(csv_ckpt), exist_ok=True)
    df.to_csv(csv_ckpt, index=False)
    df.to_parquet(parquet_ckpt, index=False)
