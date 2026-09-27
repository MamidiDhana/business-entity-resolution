"""Scalable text normalization for Business Entity Resolution.

Provides functions for business name, address, and country normalization.
Supports both scalar values and vectorized pandas Series for multi-million row processing.
"""

from __future__ import annotations

import re
import sys
from typing import Any
import unicodedata
import pandas as pd

# Ensure standard output can print Unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Common missing string representations
MISSING_STRINGS = frozenset({"", "nan", "null", "none", "<na>", "undefined", "n/a"})

# Fast Unicode translation table for punctuation and symbols (preserving letters, marks, digits)
# Characters with category starting with 'P' (Punctuation) or 'S' (Symbol) are replaced with space.
# We retain '&' separately so business name logic can convert '&' to ' and '.
PUNC_AND_SYMBOL_MAP: dict[int, str] = {
    ord(c): " "
    for c in (chr(i) for i in range(65536))
    if unicodedata.category(c).startswith(("P", "S")) and c != "&"
}

# Version of punctuation map that includes '&'
PUNC_AND_SYMBOL_MAP_WITH_AMP: dict[int, str] = {
    ord(c): " "
    for c in (chr(i) for i in range(65536))
    if unicodedata.category(c).startswith(("P", "S"))
}

# Compiled regex patterns for business name legal suffixes
LEGAL_SUFFIX_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bprivate\s+limited\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bpvt\s+limited\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bpublic\s+limited\b", re.IGNORECASE), "plc"),
    (re.compile(r"\bincorporated\b", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorporation\b", re.IGNORECASE), "corp"),
    (re.compile(r"\bcompany\b", re.IGNORECASE), "co"),
    (re.compile(r"\blimited\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\bl\s+l\s+c\b", re.IGNORECASE), "llc"),
    (re.compile(r"\bl\s+l\s+p\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bp\s+v\s+t\b", re.IGNORECASE), "pvt"),
    (re.compile(r"\bs\s+a\s+r\s+l\b", re.IGNORECASE), "sarl"),
    (re.compile(r"\bs\s+a\s+s\b", re.IGNORECASE), "sas"),
    (re.compile(r"\bs\s+a\b", re.IGNORECASE), "sa"),
    (re.compile(r"\bg\s+m\s+b\s+h\b", re.IGNORECASE), "gmbh"),
    (re.compile(r"\bp\s+l\s+c\b", re.IGNORECASE), "plc"),
]

# Compiled regex patterns for common address abbreviations
ADDRESS_ABBREVIATION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpost\s+office\s+box\b", re.IGNORECASE), "po box"),
    (re.compile(r"\bp\s*o\s*box\b", re.IGNORECASE), "po box"),
    (re.compile(r"\bp\s*o\s*b\b", re.IGNORECASE), "po box"),
    (re.compile(r"\bstreet\b", re.IGNORECASE), "st"),
    (re.compile(r"\broad\b", re.IGNORECASE), "rd"),
    (re.compile(r"\bavenue\b", re.IGNORECASE), "ave"),
    (re.compile(r"\bboulevard\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\bdrive\b", re.IGNORECASE), "dr"),
    (re.compile(r"\blane\b", re.IGNORECASE), "ln"),
    (re.compile(r"\bcourt\b", re.IGNORECASE), "ct"),
    (re.compile(r"\bsuite\b", re.IGNORECASE), "ste"),
    (re.compile(r"\bapartment\b", re.IGNORECASE), "apt"),
    (re.compile(r"\bfloor\b", re.IGNORECASE), "fl"),
    (re.compile(r"\bbuilding\b", re.IGNORECASE), "bldg"),
    (re.compile(r"\bhighway\b", re.IGNORECASE), "hwy"),
    (re.compile(r"\bparkway\b", re.IGNORECASE), "pkwy"),
]

# Multiple whitespace pattern
WHITESPACE_RE = re.compile(r"\s+")


def is_missing_value(val: Any) -> bool:
    """Check if a value represents missing or null data.
    
    Prevents conversion into misleading literal strings such as 'nan' or 'null'.
    """
    if val is None:
        return True
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in MISSING_STRINGS


def normalize_business_name(text: Any) -> str:
    """Normalize a single business name string.
    
    Steps:
    1. Check for missing/null value -> return empty string.
    2. Unicode normalization (NFKC) to standardize characters while preserving scripts.
    3. Lowercase.
    4. Replace '&' with ' and '.
    5. Replace punctuation and symbols with space (safely preserving all Unicode letters and digits).
    6. Normalize legal suffixes (e.g. 'private limited' -> 'pvt ltd', 'incorporated' -> 'inc').
    7. Collapse multiple spaces and strip.
    """
    if is_missing_value(text):
        return ""
    
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = t.replace("&", " and ")
    t = t.translate(PUNC_AND_SYMBOL_MAP)
    
    for pattern, replacement in LEGAL_SUFFIX_REPLACEMENTS:
        t = pattern.sub(replacement, t)
        
    return WHITESPACE_RE.sub(" ", t).strip()


def normalize_business_address(text: Any) -> str:
    """Normalize a single business address string.
    
    Steps:
    1. Check for missing/null value -> return empty string.
    2. Unicode normalization (NFKC).
    3. Lowercase.
    4. Replace punctuation and symbols with space (safely preserving numbers, building tokens, etc.).
    5. Normalize common justified address abbreviations (e.g. 'street' -> 'st', 'road' -> 'rd').
    6. Collapse multiple spaces and strip.
    """
    if is_missing_value(text):
        return ""
    
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = t.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
    
    for pattern, replacement in ADDRESS_ABBREVIATION_REPLACEMENTS:
        t = pattern.sub(replacement, t)
        
    return WHITESPACE_RE.sub(" ", t).strip()


def normalize_country(text: Any) -> str:
    """Normalize a single country string.
    
    Preserves all countries without hardcoding a restricted list.
    Cleans whitespace, lowercase, and safe punctuation (e.g. 'u.s.a.' -> 'usa', 'u.s.' -> 'us').
    """
    if is_missing_value(text):
        return ""
    
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"\b([a-z])\.([a-z])\.([a-z])\b", r"\1\2\3", t)
    t = re.sub(r"\b([a-z])\.([a-z])\b", r"\1\2", t)
    t = t.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
    return WHITESPACE_RE.sub(" ", t).strip()


# Vectorized Series implementations for multi-million row processing

def normalize_business_name_series(series: pd.Series) -> pd.Series:
    """Vectorized normalization of business names across a pandas Series."""
    if series.empty:
        return pd.Series(dtype=str)
        
    # Mask missing and corrupted literal strings as empty string
    s = series.fillna("").astype(str)
    s = s.mask(s.str.strip().str.lower().isin(MISSING_STRINGS), "")
    
    # NFKC Unicode normalize and lowercase
    s = s.str.normalize("NFKC").str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.translate(PUNC_AND_SYMBOL_MAP)
    
    for pattern, replacement in LEGAL_SUFFIX_REPLACEMENTS:
        s = s.str.replace(pattern.pattern, replacement, regex=True)
        
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_business_address_series(series: pd.Series) -> pd.Series:
    """Vectorized normalization of business addresses across a pandas Series."""
    if series.empty:
        return pd.Series(dtype=str)
        
    # Mask missing and corrupted literal strings as empty string
    s = series.fillna("").astype(str)
    s = s.mask(s.str.strip().str.lower().isin(MISSING_STRINGS), "")
    
    # NFKC Unicode normalize and lowercase
    s = s.str.normalize("NFKC").str.lower()
    s = s.str.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
    
    for pattern, replacement in ADDRESS_ABBREVIATION_REPLACEMENTS:
        s = s.str.replace(pattern.pattern, replacement, regex=True)
        
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_country_series(series: pd.Series) -> pd.Series:
    """Vectorized normalization of countries across a pandas Series."""
    if series.empty:
        return pd.Series(dtype=str)
        
    # Mask missing and corrupted literal strings as empty string
    s = series.fillna("").astype(str)
    s = s.mask(s.str.strip().str.lower().isin(MISSING_STRINGS), "")
    
    s = s.str.normalize("NFKC").str.lower()
    s = s.str.replace(r"\b([a-z])\.([a-z])\.([a-z])\b", r"\1\2\3", regex=True)
    s = s.str.replace(r"\b([a-z])\.([a-z])\b", r"\1\2", regex=True)
    s = s.str.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()
