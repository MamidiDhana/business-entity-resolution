"""Feature engineering pipeline for business entity resolution candidate pairs.

Extracts pairwise similarity, distance, lexical, structural, country, and data quality
features for every candidate pair (Source 1 <-> Target S2/S3).
Designed for high-throughput vectorized/batch execution with RapidFuzz and NumPy.
"""

from __future__ import annotations

import math
import re
import time
import tracemalloc
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Metadata / Identifier columns strictly separated from model features
IDENTIFIER_COLUMNS: list[str] = [
    "source1_entity_id",
    "source1_source",
    "target_entity_id",
    "target_source",
]

# Explicit list of numeric model features
FEATURE_COLUMNS: list[str] = [
    # --- Feature Group 1: Business Name ---
    "name_exact_match",
    "name_jaro_winkler",
    "name_edit_similarity",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_length_diff",
    "name_length_ratio",
    "name_token_count_diff",
    "name_shared_token_count",
    # --- Feature Group 2: Business Address ---
    "addr_exact_match",
    "addr_jaro_winkler",
    "addr_edit_similarity",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_length_diff",
    "addr_length_ratio",
    "addr_token_count_diff",
    "addr_shared_token_count",
    "addr_shared_digit_count",
    "addr_digit_match_ratio",
    # --- Feature Group 3: Country ---
    "country_exact_match",
    "country_mismatch",
    "country_missing_s1",
    "country_missing_target",
    # --- Feature Group 4: Data Quality ---
    "missing_name_s1",
    "missing_name_target",
    "missing_addr_s1",
    "missing_addr_target",
    "missing_country_either",
    "name_len_s1",
    "name_len_target",
    "addr_len_s1",
    "addr_len_target",
    # --- Feature Group 5: Target Source Indicator ---
    "target_source_is_s2",
    "target_source_is_s3",
    # --- Blocking Consensus (if available) ---
    "blocking_match_count",
]


def get_feature_names() -> list[str]:
    """Return the list of all predictive numeric feature names."""
    return list(FEATURE_COLUMNS)


def get_identifier_columns() -> list[str]:
    """Return the list of metadata identifier column names."""
    return list(IDENTIFIER_COLUMNS)


def _clean_str(val: Any) -> str:
    """Normalize input value to a clean string, avoiding literal 'nan'."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def compute_string_pair_features(
    s1: str,
    s2: str,
    is_address: bool = False,
) -> dict[str, float]:
    """Compute pairwise lexical, edit, and token similarity features for two strings.
    
    Optimized with fast-path shortcuts for exact matches and empty strings.
    """
    prefix = "addr_" if is_address else "name_"
    
    # Fast path 1: Both strings empty
    if not s1 and not s2:
        res = {
            f"{prefix}exact_match": 0.0,
            f"{prefix}jaro_winkler": 0.0,
            f"{prefix}edit_similarity": 0.0,
            f"{prefix}fuzz_ratio": 0.0,
            f"{prefix}token_sort_ratio": 0.0,
            f"{prefix}token_set_ratio": 0.0,
            f"{prefix}token_jaccard": 0.0,
            f"{prefix}token_overlap": 0.0,
            f"{prefix}length_diff": 0.0,
            f"{prefix}length_ratio": 0.0,
            f"{prefix}token_count_diff": 0.0,
            f"{prefix}shared_token_count": 0.0,
        }
        if is_address:
            res["addr_shared_digit_count"] = 0.0
            res["addr_digit_match_ratio"] = 1.0
        return res

    # Fast path 2: Exactly one string empty
    if not s1 or not s2:
        non_empty = s1 if s1 else s2
        toks = non_empty.split()
        res = {
            f"{prefix}exact_match": 0.0,
            f"{prefix}jaro_winkler": 0.0,
            f"{prefix}edit_similarity": 0.0,
            f"{prefix}fuzz_ratio": 0.0,
            f"{prefix}token_sort_ratio": 0.0,
            f"{prefix}token_set_ratio": 0.0,
            f"{prefix}token_jaccard": 0.0,
            f"{prefix}token_overlap": 0.0,
            f"{prefix}length_diff": float(len(non_empty)),
            f"{prefix}length_ratio": 0.0,
            f"{prefix}token_count_diff": float(len(toks)),
            f"{prefix}shared_token_count": 0.0,
        }
        if is_address:
            res["addr_shared_digit_count"] = 0.0
            res["addr_digit_match_ratio"] = 0.0
        return res

    # Fast path 3: Both strings identical
    if s1 == s2:
        tokens = s1.split()
        num_toks = len(tokens)
        res = {
            f"{prefix}exact_match": 1.0,
            f"{prefix}jaro_winkler": 1.0,
            f"{prefix}edit_similarity": 1.0,
            f"{prefix}fuzz_ratio": 1.0,
            f"{prefix}token_sort_ratio": 1.0,
            f"{prefix}token_set_ratio": 1.0,
            f"{prefix}token_jaccard": 1.0,
            f"{prefix}token_overlap": 1.0,
            f"{prefix}length_diff": 0.0,
            f"{prefix}length_ratio": 1.0,
            f"{prefix}token_count_diff": 0.0,
            f"{prefix}shared_token_count": float(num_toks),
        }
        if is_address:
            digits = [t for t in tokens if any(c.isdigit() for c in t)]
            res["addr_shared_digit_count"] = float(len(digits))
            res["addr_digit_match_ratio"] = 1.0
        return res

    # General path: Both strings non-empty and non-identical
    tokens1 = s1.split()
    tokens2 = s2.split()
    len1 = len(s1)
    len2 = len(s2)
    t_count1 = len(tokens1)
    t_count2 = len(tokens2)

    set1 = set(tokens1)
    set2 = set(tokens2)
    shared_tokens = set1 & set2
    shared_count = len(shared_tokens)
    union_count = len(set1 | set2)
    min_tokens = min(t_count1, t_count2)

    jaccard = (shared_count / union_count) if union_count > 0 else 0.0
    overlap = (shared_count / min_tokens) if min_tokens > 0 else 0.0
    len_diff = float(abs(len1 - len2))
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    t_count_diff = float(abs(t_count1 - t_count2))

    # RapidFuzz similarity metrics (all normalized to 0.0 - 1.0)
    jw = JaroWinkler.similarity(s1, s2)
    lev = Levenshtein.normalized_similarity(s1, s2)
    ratio = fuzz.ratio(s1, s2) / 100.0
    tok_sort = fuzz.token_sort_ratio(s1, s2) / 100.0
    tok_set = fuzz.token_set_ratio(s1, s2) / 100.0

    res = {
        f"{prefix}exact_match": 0.0,
        f"{prefix}jaro_winkler": float(jw),
        f"{prefix}edit_similarity": float(lev),
        f"{prefix}fuzz_ratio": float(ratio),
        f"{prefix}token_sort_ratio": float(tok_sort),
        f"{prefix}token_set_ratio": float(tok_set),
        f"{prefix}token_jaccard": float(jaccard),
        f"{prefix}token_overlap": float(overlap),
        f"{prefix}length_diff": len_diff,
        f"{prefix}length_ratio": float(len_ratio),
        f"{prefix}token_count_diff": t_count_diff,
        f"{prefix}shared_token_count": float(shared_count),
    }

    if is_address:
        digits1 = {t for t in tokens1 if any(c.isdigit() for c in t)}
        digits2 = {t for t in tokens2 if any(c.isdigit() for c in t)}
        shared_digits = len(digits1 & digits2)
        union_digits = len(digits1 | digits2)
        digit_ratio = (shared_digits / union_digits) if union_digits > 0 else 1.0

        res["addr_shared_digit_count"] = float(shared_digits)
        res["addr_digit_match_ratio"] = float(digit_ratio)

    return res


def compute_country_features(c1: str, c2: str) -> dict[str, float]:
    """Compute country match, mismatch, and missing indicator features."""
    c1_clean = _clean_str(c1).lower()
    c2_clean = _clean_str(c2).lower()
    
    missing_c1 = 1.0 if not c1_clean else 0.0
    missing_c2 = 1.0 if not c2_clean else 0.0
    
    if not c1_clean or not c2_clean:
        return {
            "country_exact_match": 0.0,
            "country_mismatch": 0.0,
            "country_missing_s1": missing_c1,
            "country_missing_target": missing_c2,
        }
    
    exact_match = 1.0 if c1_clean == c2_clean else 0.0
    mismatch = 1.0 if c1_clean != c2_clean else 0.0
    return {
        "country_exact_match": exact_match,
        "country_mismatch": mismatch,
        "country_missing_s1": 0.0,
        "country_missing_target": 0.0,
    }


def compute_data_quality_features(
    n1: str,
    n2: str,
    a1: str,
    a2: str,
    c1: str,
    c2: str,
) -> dict[str, float]:
    """Compute data quality indicators including missingness and record lengths."""
    return {
        "missing_name_s1": 1.0 if not n1 else 0.0,
        "missing_name_target": 1.0 if not n2 else 0.0,
        "missing_addr_s1": 1.0 if not a1 else 0.0,
        "missing_addr_target": 1.0 if not a2 else 0.0,
        "missing_country_either": 1.0 if (not c1 or not c2) else 0.0,
        "name_len_s1": float(len(n1)),
        "name_len_target": float(len(n2)),
        "addr_len_s1": float(len(a1)),
        "addr_len_target": float(len(a2)),
    }


def compute_pair_features_row(
    name1: Any,
    addr1: Any,
    country1: Any,
    name2: Any,
    addr2: Any,
    country2: Any,
    target_source: str,
    blocking_match_count: int | float = 1,
) -> dict[str, float]:
    """Compute all feature groups for a single candidate pair."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    a1 = _clean_str(addr1)
    a2 = _clean_str(addr2)
    c1 = _clean_str(country1)
    c2 = _clean_str(country2)

    feat_name = compute_string_pair_features(n1, n2, is_address=False)
    feat_addr = compute_string_pair_features(a1, a2, is_address=True)
    feat_country = compute_country_features(c1, c2)
    feat_quality = compute_data_quality_features(n1, n2, a1, a2, c1, c2)

    src = str(target_source).strip().upper()
    feat_source = {
        "target_source_is_s2": 1.0 if src == "S2" else 0.0,
        "target_source_is_s3": 1.0 if src == "S3" else 0.0,
        "blocking_match_count": float(blocking_match_count),
    }

    row_features: dict[str, float] = {}
    row_features.update(feat_name)
    row_features.update(feat_addr)
    row_features.update(feat_country)
    row_features.update(feat_quality)
    row_features.update(feat_source)
    return row_features


def build_pair_features_chunk(
    pairs_chunk: pd.DataFrame,
    s1_lookup: dict[str, tuple[str, str, str]] | pd.DataFrame | None = None,
    target_lookup: dict[str, tuple[str, str, str]] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate pairwise features for a candidate pair DataFrame chunk.
    
    Accepts:
      1. pairs_chunk with pre-populated entity fields:
         - name_s1 / business_name_s1
         - address_s1 / business_address_s1
         - country_s1
         - name_target / business_name_target
         - address_target / business_address_target
         - country_target
      2. OR pairs_chunk containing 'source1_entity_id' and 'target_entity_id',
         which are resolved using s1_lookup and target_lookup mappings.
         
    Returns:
      pd.DataFrame containing IDENTIFIER_COLUMNS followed by FEATURE_COLUMNS.
    """
    if pairs_chunk.empty:
        empty_cols = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
        df = pd.DataFrame(columns=empty_cols)
        for col in FEATURE_COLUMNS:
            df[col] = df[col].astype(np.float32)
        return df

    # Prepare lookup dicts if DataFrames are passed
    s1_dict: dict[str, tuple[str, str, str]] = {}
    if isinstance(s1_lookup, dict):
        s1_dict = s1_lookup
    elif isinstance(s1_lookup, pd.DataFrame):
        n_col = "name_normalized" if "name_normalized" in s1_lookup.columns else "business_name"
        a_col = "address_normalized" if "address_normalized" in s1_lookup.columns else "business_address"
        c_col = "country_normalized" if "country_normalized" in s1_lookup.columns else "country"
        for eid, n, a, c in zip(s1_lookup["entity_id"], s1_lookup[n_col], s1_lookup[a_col], s1_lookup[c_col]):
            s1_dict[str(eid)] = (_clean_str(n), _clean_str(a), _clean_str(c))

    target_dict: dict[str, tuple[str, str, str]] = {}
    if isinstance(target_lookup, dict):
        target_dict = target_lookup
    elif isinstance(target_lookup, pd.DataFrame):
        n_col = "name_normalized" if "name_normalized" in target_lookup.columns else "business_name"
        a_col = "address_normalized" if "address_normalized" in target_lookup.columns else "business_address"
        c_col = "country_normalized" if "country_normalized" in target_lookup.columns else "country"
        for eid, n, a, c in zip(target_lookup["entity_id"], target_lookup[n_col], target_lookup[a_col], target_lookup[c_col]):
            target_dict[str(eid)] = (_clean_str(n), _clean_str(a), _clean_str(c))

    # Resolve column names if columns exist directly in pairs_chunk
    has_inline_s1 = any(c in pairs_chunk.columns for c in ["business_name_s1", "name_s1"])
    has_inline_target = any(c in pairs_chunk.columns for c in ["business_name_target", "name_target"])

    s1_name_col = "business_name_s1" if "business_name_s1" in pairs_chunk.columns else "name_s1"
    s1_addr_col = "business_address_s1" if "business_address_s1" in pairs_chunk.columns else "address_s1"
    s1_country_col = "country_s1"

    t_name_col = "business_name_target" if "business_name_target" in pairs_chunk.columns else "name_target"
    t_addr_col = "business_address_target" if "business_address_target" in pairs_chunk.columns else "address_target"
    t_country_col = "country_target"

    # Pre-allocate column arrays
    n_rows = len(pairs_chunk)
    s1_ids = pairs_chunk["source1_entity_id"].astype(str).tolist()
    t_ids = pairs_chunk["target_entity_id"].astype(str).tolist()
    
    s1_sources = (
        pairs_chunk["source1_source"].tolist()
        if "source1_source" in pairs_chunk.columns
        else ["S1"] * n_rows
    )
    
    if "target_source" in pairs_chunk.columns:
        t_sources = pairs_chunk["target_source"].tolist()
    else:
        t_sources = ["S2" if tid.startswith("S2-") else "S3" for tid in t_ids]

    blocking_counts = (
        pairs_chunk["match_count"].fillna(1).astype(float).tolist()
        if "match_count" in pairs_chunk.columns
        else [1.0] * n_rows
    )

    feature_arrays: dict[str, np.ndarray] = {
        col: np.zeros(n_rows, dtype=np.float32) for col in FEATURE_COLUMNS
    }

    # Extract strings per row
    if has_inline_s1:
        names_s1 = pairs_chunk[s1_name_col].tolist()
        addrs_s1 = pairs_chunk[s1_addr_col].tolist()
        countries_s1 = pairs_chunk[s1_country_col].tolist() if s1_country_col in pairs_chunk.columns else [""] * n_rows
    else:
        names_s1, addrs_s1, countries_s1 = [], [], []
        for qid in s1_ids:
            item = s1_dict.get(qid, ("", "", ""))
            names_s1.append(item[0])
            addrs_s1.append(item[1])
            countries_s1.append(item[2])

    if has_inline_target:
        names_target = pairs_chunk[t_name_col].tolist()
        addrs_target = pairs_chunk[t_addr_col].tolist()
        countries_target = pairs_chunk[t_country_col].tolist() if t_country_col in pairs_chunk.columns else [""] * n_rows
    else:
        names_target, addrs_target, countries_target = [], [], []
        for tid in t_ids:
            item = target_dict.get(tid, ("", "", ""))
            names_target.append(item[0])
            addrs_target.append(item[1])
            countries_target.append(item[2])

    # Compute features row by row
    for i in range(n_rows):
        row_feat = compute_pair_features_row(
            name1=names_s1[i],
            addr1=addrs_s1[i],
            country1=countries_s1[i],
            name2=names_target[i],
            addr2=addrs_target[i],
            country2=countries_target[i],
            target_source=t_sources[i],
            blocking_match_count=blocking_counts[i],
        )
        for col in FEATURE_COLUMNS:
            feature_arrays[col][i] = row_feat.get(col, 0.0)

    # Assemble output DataFrame
    output_dict: dict[str, Any] = {
        "source1_entity_id": s1_ids,
        "source1_source": s1_sources,
        "target_entity_id": t_ids,
        "target_source": t_sources,
    }
    for col in FEATURE_COLUMNS:
        output_dict[col] = feature_arrays[col]

    return pd.DataFrame(output_dict)


def build_pair_features(
    pairs_df: pd.DataFrame,
    source1_df: pd.DataFrame | None = None,
    target_dfs: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build pairwise features for all candidate pairs in chunks.
    
    Memory-efficient: avoids full materialization and handles arbitrary millions of pairs.
    """
    if pairs_df.empty:
        return build_pair_features_chunk(pairs_df)

    # Prepare lookup mappings once
    s1_lookup: dict[str, tuple[str, str, str]] = {}
    if source1_df is not None:
        n_col = "name_normalized" if "name_normalized" in source1_df.columns else "business_name"
        a_col = "address_normalized" if "address_normalized" in source1_df.columns else "business_address"
        c_col = "country_normalized" if "country_normalized" in source1_df.columns else "country"
        for eid, n, a, c in zip(source1_df["entity_id"], source1_df[n_col], source1_df[a_col], source1_df[c_col]):
            s1_lookup[str(eid)] = (_clean_str(n), _clean_str(a), _clean_str(c))

    target_lookup: dict[str, tuple[str, str, str]] = {}
    if target_dfs is not None:
        dfs_to_index: list[pd.DataFrame] = []
        if isinstance(target_dfs, pd.DataFrame):
            dfs_to_index = [target_dfs]
        elif isinstance(target_dfs, dict):
            dfs_to_index = list(target_dfs.values())
        elif isinstance(target_dfs, Iterable):
            dfs_to_index = list(target_dfs)

        for t_df in dfs_to_index:
            n_col = "name_normalized" if "name_normalized" in t_df.columns else "business_name"
            a_col = "address_normalized" if "address_normalized" in t_df.columns else "business_address"
            c_col = "country_normalized" if "country_normalized" in t_df.columns else "country"
            for eid, n, a, c in zip(t_df["entity_id"], t_df[n_col], t_df[a_col], t_df[c_col]):
                target_lookup[str(eid)] = (_clean_str(n), _clean_str(a), _clean_str(c))

    # Process in chunks
    chunks: list[pd.DataFrame] = []
    n_pairs = len(pairs_df)
    
    if n_pairs <= chunksize:
        return build_pair_features_chunk(
            pairs_df,
            s1_lookup=s1_lookup,
            target_lookup=target_lookup,
        )

    for start_idx in range(0, n_pairs, chunksize):
        end_idx = min(start_idx + chunksize, n_pairs)
        sub_chunk = pairs_df.iloc[start_idx:end_idx]
        feat_chunk = build_pair_features_chunk(
            sub_chunk,
            s1_lookup=s1_lookup,
            target_lookup=target_lookup,
        )
        chunks.append(feat_chunk)

    return pd.concat(chunks, ignore_index=True)


def run_feature_engineering_smoke_test(
    num_pairs: int = 1_000,
) -> dict[str, Any]:
    """Execute a small smoke test on real training data to verify scale, types, and throughput."""
    print("=" * 80)
    print(f"RUNNING FEATURE ENGINEERING SMOKE TEST (Pairs = {num_pairs:,})")
    print("=" * 80)

    tracemalloc.start()
    t_start = time.perf_counter()

    # Load small slice of Source 1 and Target sources
    s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, nrows=200)
    s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, nrows=500)
    s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, nrows=500)

    # Build synthetic candidate pairs linking real S1 entities to real S2 / S3 entities
    pairs: list[dict[str, Any]] = []
    s1_ids = s1_df["entity_id"].tolist()
    s2_ids = s2_df["entity_id"].tolist()
    s3_ids = s3_df["entity_id"].tolist()

    idx = 0
    while len(pairs) < num_pairs:
        s1_id = s1_ids[idx % len(s1_ids)]
        if idx % 2 == 0:
            target_id = s2_ids[idx % len(s2_ids)]
            target_src = "S2"
        else:
            target_id = s3_ids[idx % len(s3_ids)]
            target_src = "S3"
        pairs.append({
            "source1_entity_id": s1_id,
            "source1_source": "S1",
            "target_entity_id": target_id,
            "target_source": target_src,
            "match_count": (idx % 3) + 1,
        })
        idx += 1

    pairs_df = pd.DataFrame(pairs)
    target_combined = pd.concat([s2_df, s3_df], ignore_index=True)

    # Generate pairwise features
    features_df = build_pair_features(
        pairs_df=pairs_df,
        source1_df=s1_df,
        target_dfs=target_combined,
        chunksize=500,
    )

    t_elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    print(f"Candidate Pairs Processed: {len(features_df):,}")
    print(f"Total Columns Generated:   {len(features_df.columns):,}")
    print(f"Feature Columns Count:     {len(FEATURE_COLUMNS):,}")
    print(f"Identifier Columns Count:  {len(IDENTIFIER_COLUMNS):,}")
    print(f"Total Runtime:             {t_elapsed:.2f} seconds")
    print(f"Pairs Throughput:          {len(features_df) / t_elapsed:.1f} pairs/sec")
    print(f"Peak Memory Usage:         {peak_mem_mb:.2f} MB")
    print("\nSample Feature Rows (First 3 rows, selected columns):")
    sample_cols = [
        "source1_entity_id", "target_entity_id", "target_source",
        "name_exact_match", "name_jaro_winkler", "name_token_jaccard",
        "addr_jaro_winkler", "country_exact_match", "target_source_is_s2",
    ]
    print(features_df[sample_cols].head(3).to_string(index=False))
    print("=" * 80)

    return {
        "num_pairs": len(features_df),
        "num_feature_columns": len(FEATURE_COLUMNS),
        "runtime_seconds": t_elapsed,
        "peak_memory_mb": peak_mem_mb,
    }


if __name__ == "__main__":
    run_feature_engineering_smoke_test(1_000)
