import re
import time
import hashlib
from functools import wraps
from typing import Callable, Iterable, Optional


def normalize_ws(text: str) -> str:
    """Collapse consecutive whitespace and trim."""
    return re.sub(r"\s+", " ", (text or "").strip())


def clean_text_basic(text: str) -> str:
    """Lowercase + normalize whitespace (non-destructive)."""
    return normalize_ws((text or "").lower())


def slugify(text: str, max_len: int = 80) -> str:
    """
    Create a filesystem-friendly slug.
    """
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    if len(t) > max_len:
        t = t[:max_len].rstrip("-")
    return t or "untitled"


def sha1_of(text: str) -> str:
    """Deterministic hash for caching/ids."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def timer(fn: Callable):
    """Decorator: print runtime of a function."""
    @wraps(fn)
    def _wrap(*args, **kwargs):
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.time() - t0
            print(f"[TIMER] {fn.__name__}: {dt:.2f}s")
    return _wrap


def retry(
    attempts: int = 3,
    delay_sec: float = 1.0,
    backoff: float = 2.0,
    exceptions: Iterable[type] = (Exception,),
):
    """
    Decorator: retry function on exceptions with exponential backoff.
    """
    def _decorator(fn: Callable):
        @wraps(fn)
        def _wrap(*args, **kwargs):
            wait = delay_sec
            last_exc: Optional[Exception] = None
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if i >= attempts:
                        raise
                    print(f"[RETRY] {fn.__name__} failed (attempt {i}/{attempts}): {e}. Retrying in {wait:.1f}s")
                    time.sleep(wait)
                    wait *= backoff
            if last_exc:
                raise last_exc
        return _wrap
    return _decorator
