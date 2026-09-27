"""Scalable blocking and candidate generation for Business Entity Resolution.

Supports multiple complementary country-aware blocking strategies:
  1. Exact normalized business name
  2. Exact normalized address
  3. Approximate name (prefix words, compact alphanumeric prefix)
  4. Approximate address (prefix tokens, street number + distinctive word)
  5. Country-aware filtering across all signals

Designed for multi-million-row scale using streaming/chunked processing,
avoiding Cartesian products and billions of candidate pairs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time
import tracemalloc
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import PUNC_AND_SYMBOL_MAP_WITH_AMP
from src.preprocessing import preprocess_dataframe

# Ensure standard output can print Unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Stop words to exclude from single address word tokens
ADDRESS_STOP_WORDS = frozenset({
    "road", "street", "avenue", "drive", "boulevard", "lane", "court",
    "building", "floor", "suite", "unit", "post", "office", "box",
    "north", "south", "east", "west", "near", "opp", "opposite",
    "the", "and", "or", "in", "at", "to", "for", "st", "rd", "ave",
    "dr", "blvd", "ln", "ct", "ste", "apt", "fl", "bldg", "po", "pob",
    "nagar", "block", "flat", "house", "sector", "room", "shop", "colony",
    "vihar", "enclave", "plot", "no", "hno", "gala", "bazaar", "chowk", "market",
    # Administrative / City / State names that should not be used as unique street tokens
    "delhi", "mumbai", "bangalore", "chennai", "kolkata", "hyderabad", "pune",
    "ahmedabad", "jaipur", "lucknow", "surat", "kanpur", "nagpur", "patna",
    "indore", "thane", "bhopal", "visakhapatnam", "vadodara", "firozabad",
    "ludhiana", "agra", "siliguri", "nashik", "faridabad", "patiala", "meerut",
    "kalyan", "varanasi", "srinagar", "dhanbad", "jodhpur", "amritsar", "raipur",
    "allahabad", "coimbatore", "jabalpur", "gwalior", "vijayawada", "madurai",
    "guwahati", "chandigarh", "hubli", "dharwad", "amravati", "noida", "gurgaon",
    "gurugram", "orissa", "odisha", "karnataka", "maharashtra", "gujarat",
    "punjab", "haryana", "rajasthan", "tamil", "nadu", "kerala", "andhra",
    "pradesh", "telangana", "bengal", "bihar", "uttar", "madhya", "india",
})

GENERIC_NAME_TOKENS = frozenset({
    "the", "and", "inc", "corp", "llc", "ltd", "pvt", "co", "first",
    "national", "global", "international", "american", "united", "hotel",
    "restaurant", "store", "shop", "center", "centre", "service", "services",
    "solutions", "group", "holdings", "enterprises", "associates", "management",
    "consulting", "trading", "logistics", "company", "corporation", "limited",
    "mr", "ms", "mrs", "dr", "private", "public",
})

STRATEGY_WEIGHTS: dict[str, int] = {
    "exact_name": 100,
    "exact_addr": 80,
    "approx_name_first2": 30,
    "approx_name_sorted2": 25,
    "approx_addr_first3": 20,
    "approx_addr_num_word": 10,
    "approx_name_compact": 5,
}

WHITESPACE_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\b\d+\b")
WORDS_RE = re.compile(r"\b[^\d\s]{4,}\b")


@dataclass
class BlockingConfig:
    """Configuration parameters for candidate blocking."""
    max_candidates_per_query: int = 50
    chunksize: int = 500_000
    min_name_len: int = 3
    min_addr_len: int = 5
    compact_prefix_len: int = 8
    source_balanced: bool = False
    buffer_limit_per_source: int = 300


def extract_blocking_keys(
    name: Any,
    address: Any,
    country: Any,
    config: BlockingConfig | None = None,
) -> list[tuple[str, str]]:
    """Extract multi-signal blocking keys for a single entity record.
    
    All keys are strictly prefixed with country to enforce country-aware blocking.
    Optimized for multi-million-row throughput using native C string routines.
    
    Returns:
        List of (strategy_name, key_string) tuples.
    """
    if config is None:
        config = BlockingConfig()
        
    c = str(country).strip().lower() if country and country == country and country != "nan" else ""
    n = str(name).strip().lower() if name and name == name and name != "nan" else ""
    a = str(address).strip().lower() if address and address == address and address != "nan" else ""
    
    keys: list[tuple[str, str]] = []
    
    # 1. Exact Name Strategy
    if n and len(n) >= config.min_name_len:
        n_clean = n.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
        words = n_clean.split()
        if words:
            clean_n = " ".join(words)
            keys.append(("exact_name", f"{c}||{clean_n}"))
            
            # 2. Approximate Name: First 2 tokens
            if len(words) >= 2:
                keys.append(("approx_name_first2", f"{c}||{words[0]} {words[1]}"))
            elif len(words[0]) >= 4 and words[0] not in GENERIC_NAME_TOKENS:
                keys.append(("approx_name_first2", f"{c}||{words[0]}"))
                
            # 3. Approximate Name: Sorted 2 tokens (handles word reordering / M/s prefixes)
            sig = sorted([w for w in words if w not in GENERIC_NAME_TOKENS and len(w) >= 3])
            if len(sig) >= 2:
                keys.append(("approx_name_sorted2", f"{c}||{sig[0]} {sig[1]}"))
            elif len(sig) == 1 and len(sig[0]) >= 4:
                keys.append(("approx_name_sorted2", f"{c}||{sig[0]}"))
                
            # 4. Approximate Name: Compact alphanumeric prefix (handles joint words, domains)
            comp = "".join(words)
            if len(comp) >= config.compact_prefix_len:
                keys.append(("approx_name_compact", f"{c}||{comp[:config.compact_prefix_len]}"))
                
    # 5. Exact Address Strategy
    if a and len(a) >= config.min_addr_len:
        a_clean = a.translate(PUNC_AND_SYMBOL_MAP_WITH_AMP)
        tokens = a_clean.split()
        if tokens:
            clean_a = " ".join(tokens)
            keys.append(("exact_addr", f"{c}||{clean_a}"))
            
            # 6. Approximate Address: First 3 tokens (e.g. street number + street name)
            if len(tokens) >= 3:
                keys.append(("approx_addr_first3", f"{c}||{tokens[0]} {tokens[1]} {tokens[2]}"))
                
            # 7. Approximate Address: Number + Distinctive Word (catches typos in street name)
            nums: list[str] = []
            words_addr: list[str] = []
            for t in tokens:
                if t.isdigit() and len(nums) < 2:
                    nums.append(t)
                elif len(t) >= 4 and t not in ADDRESS_STOP_WORDS and len(words_addr) < 2 and not t.isdigit():
                    words_addr.append(t)
            for num in nums:
                for w in words_addr:
                    keys.append(("approx_addr_num_word", f"{c}||{num}_{w}"))
                    
    return keys


def build_query_blocking_index(
    query_df: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> tuple[dict[tuple[str, str], list[str]], set[str]]:
    """Build an inverted index mapping blocking keys to matching query entity IDs.
    
    Returns:
        tuple of (key_to_query_ids_dict, set_of_all_keys)
    """
    if config is None:
        config = BlockingConfig()
        
    key_to_qids: dict[tuple[str, str], list[str]] = defaultdict(list)
    
    # Check if preprocessed columns exist, otherwise fall back to raw columns
    name_col = "name_normalized" if "name_normalized" in query_df.columns else "business_name"
    addr_col = "address_normalized" if "address_normalized" in query_df.columns else "business_address"
    country_col = "country_normalized" if "country_normalized" in query_df.columns else "country"
    
    for qid, n, a, c in zip(
        query_df["entity_id"],
        query_df[name_col],
        query_df[addr_col],
        query_df[country_col],
    ):
        record_keys = extract_blocking_keys(n, a, c, config=config)
        for strategy, key_val in record_keys:
            key_to_qids[(strategy, key_val)].append(qid)
            
    all_keys = set(key_to_qids.keys())
    return key_to_qids, all_keys


def match_target_chunk(
    chunk_df: pd.DataFrame,
    key_to_qids: dict[tuple[str, str], list[str]],
    target_source: str,
    config: BlockingConfig | None = None,
) -> list[tuple[str, str, str, str, str]]:
    """Match a chunk of target entities against the query inverted index.
    
    Returns:
        List of (source1_entity_id, "S1", target_entity_id, target_source, strategy) tuples.
    """
    if config is None:
        config = BlockingConfig()
        
    name_col = "name_normalized" if "name_normalized" in chunk_df.columns else "business_name"
    addr_col = "address_normalized" if "address_normalized" in chunk_df.columns else "business_address"
    country_col = "country_normalized" if "country_normalized" in chunk_df.columns else "country"
    
    matches: list[tuple[str, str, str, str, str]] = []
    
    for tid, n, a, c in zip(
        chunk_df["entity_id"],
        chunk_df[name_col],
        chunk_df[addr_col],
        chunk_df[country_col],
    ):
        t_keys = extract_blocking_keys(n, a, c, config=config)
        for strat, kval in t_keys:
            qids = key_to_qids.get((strat, kval))
            if qids:
                for qid in qids:
                    matches.append((qid, "S1", tid, target_source, strat))
                    
    return matches


def finalize_candidates_from_source_maps(
    cands_by_source: dict[str, dict[str, dict[str, set[str]]]],
    max_candidates_per_query: int = 50,
    source_balanced: bool = False,
) -> pd.DataFrame:
    """Aggregate, deduplicate, rank, and truncate candidate pairs per query entity.
    
    Supports:
      - Global ranking (highest strategy consensus score across all sources)
      - Source-balanced ranking (dedicated quotas per target source with automatic overflow)
    """
    s2_map = cands_by_source.get("S2", {})
    s3_map = cands_by_source.get("S3", {})
    all_qids = sorted(set(s2_map.keys()).union(s3_map.keys()))
    
    qids_list: list[str] = []
    tids_list: list[str] = []
    tsrc_list: list[str] = []
    strats_list: list[str] = []
    counts_list: list[int] = []
    
    quota_s2 = max_candidates_per_query // 2 if source_balanced else max_candidates_per_query
    quota_s3 = (max_candidates_per_query - quota_s2) if source_balanced else max_candidates_per_query
    
    for qid in all_qids:
        s2_dict = s2_map.get(qid, {})
        s3_dict = s3_map.get(qid, {})
        
        s2_items: list[tuple[str, str, int, str, int]] = []
        for tid, strats in s2_dict.items():
            sc = sum(STRATEGY_WEIGHTS.get(s, 1) for s in strats)
            s2_items.append((tid, "S2", len(strats), ",".join(sorted(strats)), sc))
        s2_items.sort(key=lambda item: (-item[4], -item[2], item[0]))
        
        s3_items: list[tuple[str, str, int, str, int]] = []
        for tid, strats in s3_dict.items():
            sc = sum(STRATEGY_WEIGHTS.get(s, 1) for s in strats)
            s3_items.append((tid, "S3", len(strats), ",".join(sorted(strats)), sc))
        s3_items.sort(key=lambda item: (-item[4], -item[2], item[0]))
        
        if source_balanced:
            s2_selected = s2_items[:quota_s2]
            s3_selected = s3_items[:quota_s3]
            
            # Overflow unused quota from one source to the other
            if len(s2_selected) < quota_s2:
                extra = quota_s2 - len(s2_selected)
                s3_selected = s3_items[:quota_s3 + extra]
            elif len(s3_selected) < quota_s3:
                extra = quota_s3 - len(s3_selected)
                s2_selected = s2_items[:quota_s2 + extra]
                
            combined = s2_selected + s3_selected
            combined.sort(key=lambda item: (-item[4], -item[2], item[0]))
        else:
            combined = s2_items + s3_items
            combined.sort(key=lambda item: (-item[4], -item[2], item[0]))
            combined = combined[:max_candidates_per_query]
            
        for tid, tsrc, m_count, m_strats, _score in combined:
            qids_list.append(qid)
            tids_list.append(tid)
            tsrc_list.append(tsrc)
            strats_list.append(m_strats)
            counts_list.append(m_count)
            
    if not qids_list:
        return pd.DataFrame(columns=[
            "source1_entity_id", "source1_source",
            "target_entity_id", "target_source",
            "matched_strategies", "match_count",
        ])
    return pd.DataFrame({
        "source1_entity_id": qids_list,
        "source1_source": "S1",
        "target_entity_id": tids_list,
        "target_source": tsrc_list,
        "matched_strategies": strats_list,
        "match_count": counts_list,
    })


def finalize_candidates(
    raw_matches: list[tuple[str, str, str, str, str]],
    max_candidates_per_query: int = 50,
    source_balanced: bool = False,
) -> pd.DataFrame:
    """Aggregate, deduplicate, rank, and truncate candidate pairs per query entity.
    
    Guarantees:
      - Every pair is S1 -> S2 or S1 -> S3
      - No S1 -> S1 pairs
      - No S2 -> S3 pairs
      - Unique (source1_entity_id, target_entity_id) pairs
    """
    if not raw_matches:
        return pd.DataFrame(columns=[
            "source1_entity_id", "source1_source",
            "target_entity_id", "target_source",
            "matched_strategies", "match_count",
        ])
        
    cands_by_source: dict[str, dict[str, dict[str, set[str]]]] = {
        "S2": defaultdict(lambda: defaultdict(set)),
        "S3": defaultdict(lambda: defaultdict(set)),
    }
    for qid, qsrc, tid, tsrc, strat in raw_matches:
        if not qid.startswith("S1-") or not (tid.startswith("S2-") or tid.startswith("S3-")):
            continue
        if qid == tid:
            continue
        cands_by_source[tsrc][qid][tid].add(strat)
        
    return finalize_candidates_from_source_maps(
        cands_by_source=cands_by_source,
        max_candidates_per_query=max_candidates_per_query,
        source_balanced=source_balanced,
    )


def generate_candidates(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_source: str = "S2",
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Generate candidate pairs in memory between query_df and target_df.
    
    Used for unit testing and small-batch in-memory processing.
    """
    if config is None:
        config = BlockingConfig()
        
    key_to_qids, _ = build_query_blocking_index(query_df, config=config)
    raw_matches = match_target_chunk(target_df, key_to_qids, target_source=target_source, config=config)
    return finalize_candidates(
        raw_matches,
        max_candidates_per_query=config.max_candidates_per_query,
        source_balanced=config.source_balanced,
    )


def collect_candidates_by_source_chunked(
    query_df: pd.DataFrame,
    target_sources: list[tuple[Path | str, str]],
    config: BlockingConfig | None = None,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    """Stream target TSVs and collect raw matching candidate maps separated by target source.
    
    Each source accumulates into its own isolated dictionary so that earlier streamed
    sources never crowd out or displace later streamed sources.
    
    Returns:
        dict: {"S2": {qid: {tid: set_of_strategies}}, "S3": {qid: {tid: set_of_strategies}}}
    """
    if config is None:
        config = BlockingConfig()
        
    print(f"Building query blocking index for {len(query_df):,} Source 1 records...", flush=True)
    key_to_qids, all_keys = build_query_blocking_index(query_df, config=config)
    print(f"Indexed {len(all_keys):,} distinct blocking keys across strategies.", flush=True)
    
    cands_by_source: dict[str, dict[str, dict[str, set[str]]]] = {
        "S2": defaultdict(lambda: defaultdict(set)),
        "S3": defaultdict(lambda: defaultdict(set)),
    }
    
    name_col = "name_normalized" if "name_normalized" in query_df.columns else "business_name"
    addr_col = "address_normalized" if "address_normalized" in query_df.columns else "business_address"
    country_col = "country_normalized" if "country_normalized" in query_df.columns else "country"
    
    for file_path, source_tag in target_sources:
        path = Path(file_path)
        if not path.is_file():
            print(f"Warning: Target file not found: {path}", flush=True)
            continue
            
        print(f"\nStreaming {path.name} (Source {source_tag}) in chunks of {config.chunksize:,}...", flush=True)
        t0 = time.perf_counter()
        total_target_rows = 0
        chunk_num = 0
        
        target_cands = cands_by_source[source_tag]
        
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=config.chunksize):
            chunk_num += 1
            total_target_rows += len(chunk)
            t_chunk_start = time.perf_counter()
            
            c_name_col = "name_normalized" if "name_normalized" in chunk.columns else "business_name"
            c_addr_col = "address_normalized" if "address_normalized" in chunk.columns else "business_address"
            c_country_col = "country_normalized" if "country_normalized" in chunk.columns else "country"
            
            tids = chunk["entity_id"].tolist()
            names = chunk[c_name_col].tolist()
            addrs = chunk[c_addr_col].tolist()
            countries = chunk[c_country_col].tolist()
            
            for tid, n, a, c in zip(tids, names, addrs, countries):
                t_keys = extract_blocking_keys(n, a, c, config=config)
                for strat, kval in t_keys:
                    qids = key_to_qids.get((strat, kval))
                    if qids:
                        for qid in qids:
                            sc = target_cands[qid]
                            if tid in sc:
                                sc[tid].add(strat)
                            elif len(sc) < config.buffer_limit_per_source or strat in {"exact_name", "exact_addr"}:
                                sc[tid] = {strat}
                                
            chunk_duration = time.perf_counter() - t_chunk_start
            print(f"  Chunk {chunk_num:2d}: {total_target_rows:,} rows processed (chunk took {chunk_duration:.1f}s)", flush=True)
            
        elapsed = time.perf_counter() - t0
        print(f"  Completed {path.name}: {total_target_rows:,} rows in {elapsed:.2f}s", flush=True)
        
    return cands_by_source


def generate_candidates_chunked(
    query_df: pd.DataFrame,
    target_sources: list[tuple[Path | str, str]],
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Generate candidates by streaming through large target source TSVs in bounded chunks.
    
    Args:
        query_df: Preprocessed or raw DataFrame of Source 1 queries.
        target_sources: List of (file_path, source_tag) tuples, e.g. [('.../train_source2.tsv', 'S2')].
        config: Blocking configuration.
        
    Returns:
        Deduplicated, ranked, and truncated candidate DataFrame.
    """
    if config is None:
        config = BlockingConfig()
        
    cands_by_source = collect_candidates_by_source_chunked(
        query_df=query_df,
        target_sources=target_sources,
        config=config,
    )
    
    print("\nFinalizing, deduplicating, and ranking candidates...", flush=True)
    candidates_df = finalize_candidates_from_source_maps(
        cands_by_source=cands_by_source,
        max_candidates_per_query=config.max_candidates_per_query,
        source_balanced=config.source_balanced,
    )
    print(f"Generated {len(candidates_df):,} unique candidate pairs for {candidates_df['source1_entity_id'].nunique():,} queries.", flush=True)
    return candidates_df


def evaluate_candidate_recall(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    query_ids: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate candidate recall against the ground truth.
    
    Accounts for MULTIPLE TRUE MATCHES per query entity.
    Evaluates every true target reference individually.
    
    Formula:
        candidate_recall = (total found true target references) / (total true target references)
    """
    if query_ids is not None:
        query_set = set(query_ids)
        gt_filtered = ground_truth_df[ground_truth_df["source1_entity_id"].isin(query_set)]
    else:
        gt_filtered = ground_truth_df
        query_set = set(gt_filtered["source1_entity_id"])
        
    total_queries = len(query_set)
    
    # Map each query ID to its ground truth set of target IDs
    true_targets_by_query: dict[str, set[str]] = {}
    total_true_refs = 0
    s2_true_refs = 0
    s3_true_refs = 0
    queries_with_zero_matches = 0
    
    for qid, m in zip(gt_filtered["source1_entity_id"], gt_filtered["matched_entity_ids"]):
        if pd.isna(m) or not str(m).strip():
            true_targets_by_query[qid] = set()
            queries_with_zero_matches += 1
        else:
            targets = {t.strip() for t in str(m).split(",") if t.strip()}
            true_targets_by_query[qid] = targets
            total_true_refs += len(targets)
            for t in targets:
                if t.startswith("S2-"):
                    s2_true_refs += 1
                elif t.startswith("S3-"):
                    s3_true_refs += 1
                    
    # Map generated candidates to query IDs
    candidates_by_query: dict[str, set[str]] = defaultdict(set)
    if not candidates_df.empty:
        for qid, tid in zip(candidates_df["source1_entity_id"], candidates_df["target_entity_id"]):
            candidates_by_query[qid].add(tid)
            
    # Compute recall statistics across all true target references
    found_total = 0
    found_s2 = 0
    found_s3 = 0
    
    for qid, targets in true_targets_by_query.items():
        cands = candidates_by_query.get(qid, set())
        for t in targets:
            if t in cands:
                found_total += 1
                if t.startswith("S2-"):
                    found_s2 += 1
                elif t.startswith("S3-"):
                    found_s3 += 1
                    
    overall_recall = (found_total / total_true_refs) if total_true_refs > 0 else 1.0
    s2_recall = (found_s2 / s2_true_refs) if s2_true_refs > 0 else 1.0
    s3_recall = (found_s3 / s3_true_refs) if s3_true_refs > 0 else 1.0
    lost_matches = total_true_refs - found_total
    lost_pct = (lost_matches / total_true_refs) if total_true_refs > 0 else 0.0
    
    # Candidate counts per query
    cand_counts = [len(candidates_by_query.get(qid, set())) for qid in query_set]
    avg_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
    med_cands = float(np.median(cand_counts)) if cand_counts else 0.0
    max_cands = int(np.max(cand_counts)) if cand_counts else 0
    min_cands = int(np.min(cand_counts)) if cand_counts else 0
    zero_cands_count = sum(1 for c in cand_counts if c == 0)
    zero_cands_pct = (zero_cands_count / total_queries) if total_queries > 0 else 0.0
    
    report: dict[str, Any] = {
        "total_queries": total_queries,
        "queries_with_zero_true_matches": queries_with_zero_matches,
        "total_true_target_references": total_true_refs,
        "found_true_target_references": found_total,
        "overall_candidate_recall": overall_recall,
        "s2_true_references": s2_true_refs,
        "s2_found_references": found_s2,
        "s2_candidate_recall": s2_recall,
        "s3_true_references": s3_true_refs,
        "s3_found_references": found_s3,
        "s3_candidate_recall": s3_recall,
        "lost_matches_count": lost_matches,
        "lost_matches_percentage": lost_pct,
        "average_candidates_per_query": avg_cands,
        "median_candidates_per_query": med_cands,
        "max_candidates_per_query": max_cands,
        "min_candidates_per_query": min_cands,
        "queries_with_zero_candidates": zero_cands_count,
        "queries_with_zero_candidates_percentage": zero_cands_pct,
    }
    return report


def run_sample_blocking_benchmark(
    sample_size: int = 10_000,
    max_candidates_per_query: int = 50,
) -> dict[str, Any]:
    """Execute the reproducible sample development benchmark for candidate generation.
    
    Takes a 10,000 Source 1 query sample, generates candidates against the target datasets,
    evaluates recall against ground truth, and reports runtime and memory usage.
    """
    print("=" * 80)
    print(f"RUNNING BLOCKING SAMPLE BENCHMARK (Sample size = {sample_size:,})")
    print("=" * 80)
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # 1. Load reproducible query sample
    s1_path = TRAIN_DIR / "train_source1.tsv"
    print(f"Loading {sample_size:,} Source 1 query entities from {s1_path.name}...")
    query_df = pd.read_csv(s1_path, sep="\t", dtype=str, nrows=sample_size)
    query_ids = set(query_df["entity_id"])
    
    # 2. Configure target sources
    target_sources = [
        (TRAIN_DIR / "train_source2.tsv", "S2"),
        (TRAIN_DIR / "train_source3.tsv", "S3"),
    ]
    
    config = BlockingConfig(
        max_candidates_per_query=max_candidates_per_query,
        chunksize=500_000,
    )
    
    # 3. Generate candidates
    candidates_df = generate_candidates_chunked(
        query_df=query_df,
        target_sources=target_sources,
        config=config,
    )
    
    # 4. Evaluate against ground truth
    print("\nLoading ground truth to evaluate recall...", flush=True)
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    gt_rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=250_000):
        matched = chunk[chunk["source1_entity_id"].isin(query_ids)]
        if not matched.empty:
            gt_rows.append(matched)
    gt_df = pd.concat(gt_rows, ignore_index=True) if gt_rows else pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    
    eval_results = evaluate_candidate_recall(
        candidates_df=candidates_df,
        ground_truth_df=gt_df,
        query_ids=query_ids,
    )
    
    total_elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    eval_results["runtime_seconds"] = total_elapsed
    eval_results["peak_memory_mb"] = peak_mem_mb
    eval_results["total_candidate_pairs"] = len(candidates_df)
    
    print("\n" + "=" * 80, flush=True)
    print("BLOCKING EVALUATION REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"Sample Size:                 {sample_size:,} Source 1 entities", flush=True)
    print(f"Total True Targets:          {eval_results['total_true_target_references']:,}", flush=True)
    print(f"Candidate Recall (Overall):  {eval_results['overall_candidate_recall']:.2%}", flush=True)
    print(f"Source 2 Recall:             {eval_results['s2_candidate_recall']:.2%} ({eval_results['s2_found_references']:,}/{eval_results['s2_true_references']:,})", flush=True)
    print(f"Source 3 Recall:             {eval_results['s3_candidate_recall']:.2%} ({eval_results['s3_found_references']:,}/{eval_results['s3_true_references']:,})", flush=True)
    print(f"True Matches Lost:           {eval_results['lost_matches_count']:,} ({eval_results['lost_matches_percentage']:.2%})", flush=True)
    print(f"Total Candidate Pairs:       {eval_results['total_candidate_pairs']:,}", flush=True)
    print(f"Average Candidates per S1:   {eval_results['average_candidates_per_query']:.2f}", flush=True)
    print(f"Median Candidates per S1:    {eval_results['median_candidates_per_query']:.1f}", flush=True)
    print(f"Maximum Candidates per S1:   {eval_results['max_candidates_per_query']}", flush=True)
    print(f"Queries with 0 Candidates:   {eval_results['queries_with_zero_candidates']} ({eval_results['queries_with_zero_candidates_percentage']:.2%})", flush=True)
    print(f"Total Benchmark Runtime:     {total_elapsed:.2f} seconds", flush=True)
    print(f"Peak Memory Usage:           {peak_mem_mb:.2f} MB", flush=True)
    print("=" * 80, flush=True)
    
    return eval_results


def run_multi_cap_blocking_experiments(
    sample_size: int = 10_000,
    caps: list[int] | None = None,
) -> dict[str, Any]:
    """Execute multi-cap evaluations (50, 75, 100, 150, 200) under Global and Source-Balanced ranking.
    
    Streams the 10.3M target rows ONCE, then evaluates all caps in memory.
    """
    if caps is None:
        caps = [50, 75, 100, 150, 200]
        
    print("=" * 80, flush=True)
    print(f"RUNNING MULTI-CAP BLOCKING EXPERIMENTS (Sample size = {sample_size:,})", flush=True)
    print("=" * 80, flush=True)
    
    tracemalloc.start()
    t_start = time.perf_counter()
    
    # 1. Load reproducible query sample
    s1_path = TRAIN_DIR / "train_source1.tsv"
    print(f"Loading {sample_size:,} Source 1 query entities from {s1_path.name}...", flush=True)
    query_df = pd.read_csv(s1_path, sep="\t", dtype=str, nrows=sample_size)
    query_ids = set(query_df["entity_id"])
    
    # 2. Configure target sources
    target_sources = [
        (TRAIN_DIR / "train_source2.tsv", "S2"),
        (TRAIN_DIR / "train_source3.tsv", "S3"),
    ]
    
    max_cap = max(caps)
    config = BlockingConfig(
        max_candidates_per_query=max_cap,
        buffer_limit_per_source=max(300, max_cap + 50),
        chunksize=500_000,
    )
    
    # 3. Stream once to collect candidates
    cands_by_source = collect_candidates_by_source_chunked(
        query_df=query_df,
        target_sources=target_sources,
        config=config,
    )
    streaming_elapsed = time.perf_counter() - t_start
    print(f"\nStreaming complete in {streaming_elapsed:.2f}s.", flush=True)
    
    # 4. Load ground truth once
    print("\nLoading ground truth for sample queries...", flush=True)
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    gt_rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=250_000):
        matched = chunk[chunk["source1_entity_id"].isin(query_ids)]
        if not matched.empty:
            gt_rows.append(matched)
    gt_df = pd.concat(gt_rows, ignore_index=True) if gt_rows else pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    
    # 5. Evaluate all caps
    experiment_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    
    modes = [("Global", False), ("Source-Balanced", True)]
    
    for mode_name, is_balanced in modes:
        print(f"\nEvaluating Mode: {mode_name}...", flush=True)
        for cap in caps:
            t_eval_start = time.perf_counter()
            cands_df = finalize_candidates_from_source_maps(
                cands_by_source=cands_by_source,
                max_candidates_per_query=cap,
                source_balanced=is_balanced,
            )
            eval_res = evaluate_candidate_recall(
                candidates_df=cands_df,
                ground_truth_df=gt_df,
                query_ids=query_ids,
            )
            eval_duration = time.perf_counter() - t_eval_start
            
            exp_key = f"{mode_name.lower().replace('-', '_')}_cap_{cap}"
            eval_res["mode"] = mode_name
            eval_res["cap"] = cap
            eval_res["total_candidate_pairs"] = len(cands_df)
            eval_res["eval_duration_seconds"] = eval_duration
            experiment_results[exp_key] = eval_res
            
            summary_rows.append({
                "Mode": mode_name,
                "Cap": cap,
                "Recall": f"{eval_res['overall_candidate_recall']:.2%}",
                "S2 Recall": f"{eval_res['s2_candidate_recall']:.2%}",
                "S3 Recall": f"{eval_res['s3_candidate_recall']:.2%}",
                "Total Candidates": len(cands_df),
                "Avg Cands": f"{eval_res['average_candidates_per_query']:.2f}",
                "Med Cands": f"{eval_res['median_candidates_per_query']:.1f}",
                "Max Cands": eval_res['max_candidates_per_query'],
                "Lost Matches": eval_res['lost_matches_count'],
            })
            
    total_elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    print("\n" + "=" * 100, flush=True)
    print("MULTI-CAP CANDIDATE BLOCKING OPTIMIZATION COMPARISON TABLE", flush=True)
    print("=" * 100, flush=True)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False), flush=True)
    print("=" * 100, flush=True)
    print(f"Total Experiment Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)", flush=True)
    print(f"Peak Memory Usage:        {peak_mem_mb:.2f} MB", flush=True)
    print("=" * 100, flush=True)
    
    experiment_results["summary_table"] = summary_rows
    experiment_results["total_runtime_seconds"] = total_elapsed
    experiment_results["peak_memory_mb"] = peak_mem_mb
    return experiment_results


if __name__ == "__main__":
    run_multi_cap_blocking_experiments(sample_size=10_000, caps=[50, 75, 100, 150, 200])
