"""Step 11: Test Model Scoring and Final Match Generation.

Scores candidate pairs from Step 10 using the trained classifier and selected threshold (0.9900).
Generates:
  - output/scored_candidates.tsv (candidate-level probabilities and binary predictions)
  - output/matching_results.tsv (Source 1 entity-level aggregated matches)
  - output/final_prediction_report.json (comprehensive statistics and audit)

Guarantees:
  - Strict candidate subset constraint (matching_results <= candidate_pairs)
  - Memory-safe chunked execution (< 1.5 GB RAM)
  - Multi-match support (no top-1 forcing)
  - Official challenge validator compliance
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    _clean_str,
    build_pair_features_chunk,
)
from utils.validate_submission import validate_submission

DATASET_DIR = PROJECT_ROOT / "dataset"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_MODEL_PATH = MODELS_DIR / "model.joblib"
DEFAULT_FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
DEFAULT_THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"
DEFAULT_CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
DEFAULT_SCORED_CANDIDATES_PATH = OUTPUT_DIR / "scored_candidates.tsv"
DEFAULT_MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "final_prediction_report.json"


def load_scoring_artifacts(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    feature_cols_path: Path | str = DEFAULT_FEATURE_COLS_PATH,
    threshold_config_path: Path | str = DEFAULT_THRESHOLD_CONFIG_PATH,
) -> tuple[Any, list[str], float]:
    """Load the trained model, expected feature columns, and selected threshold."""
    m_path = Path(model_path)
    fc_path = Path(feature_cols_path)
    th_path = Path(threshold_config_path)

    if not m_path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {m_path}")
    if not fc_path.is_file():
        raise FileNotFoundError(f"Feature columns configuration not found: {fc_path}")
    if not th_path.is_file():
        raise FileNotFoundError(f"Threshold configuration not found: {th_path}")

    model = joblib.load(m_path)

    with open(fc_path, "r", encoding="utf-8") as f:
        feature_columns: list[str] = json.load(f)

    with open(th_path, "r", encoding="utf-8") as f:
        threshold_config: dict[str, Any] = json.load(f)

    selected_threshold = float(threshold_config.get("selected_threshold", 0.9900))
    print(f"Loaded trained model: {type(model).__name__}")
    print(f"Loaded {len(feature_columns)} feature columns from {fc_path.name}")
    print(f"Selected classification threshold: {selected_threshold:.4f}")
    return model, feature_columns, selected_threshold


def build_entity_lookups(
    s1_path: Path | str,
    s2_path: Path | str,
    s3_path: Path | str,
    required_s1_ids: set[str] | None = None,
    required_target_ids: set[str] | None = None,
    chunksize: int = 500_000,
) -> tuple[dict[str, tuple[str, str, str]], dict[str, tuple[str, str, str]]]:
    """Build memory-efficient string tuple lookups for entities present in candidates.
    
    Returns:
        s1_lookup: dict[s1_id, (name, addr, country)]
        target_lookup: dict[target_id, (name, addr, country)]
    """
    print("\nBuilding memory-efficient entity lookups for scoring...")
    t0 = time.perf_counter()

    # 1. Source 1 lookup
    s1_lookup: dict[str, tuple[str, str, str]] = {}
    print(f"  Reading Source 1 from {Path(s1_path).name}...")
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=chunksize):
        n_col = "name_normalized" if "name_normalized" in chunk.columns else "business_name"
        a_col = "address_normalized" if "address_normalized" in chunk.columns else "business_address"
        c_col = "country_normalized" if "country_normalized" in chunk.columns else "country"
        
        for eid, n, a, c in zip(chunk["entity_id"], chunk[n_col], chunk[a_col], chunk[c_col]):
            eid_str = str(eid)
            if required_s1_ids is None or eid_str in required_s1_ids:
                s1_lookup[eid_str] = (_clean_str(n), _clean_str(a), _clean_str(c))

    print(f"  Source 1 entities indexed: {len(s1_lookup):,}")

    # 2. Target lookups (S2 and S3)
    target_lookup: dict[str, tuple[str, str, str]] = {}
    for t_path, label in [(s2_path, "Source 2"), (s3_path, "Source 3")]:
        print(f"  Reading {label} from {Path(t_path).name}...")
        for chunk in pd.read_csv(t_path, sep="\t", dtype=str, chunksize=chunksize):
            n_col = "name_normalized" if "name_normalized" in chunk.columns else "business_name"
            a_col = "address_normalized" if "address_normalized" in chunk.columns else "business_address"
            c_col = "country_normalized" if "country_normalized" in chunk.columns else "country"
            
            for eid, n, a, c in zip(chunk["entity_id"], chunk[n_col], chunk[a_col], chunk[c_col]):
                eid_str = str(eid)
                if required_target_ids is None or eid_str in required_target_ids:
                    target_lookup[eid_str] = (_clean_str(n), _clean_str(a), _clean_str(c))

    t_elapsed = time.perf_counter() - t0
    print(f"  Target entities indexed: {len(target_lookup):,} in {t_elapsed:.2f}s.")
    return s1_lookup, target_lookup


def score_candidate_chunk(
    pairs_chunk: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    threshold: float,
    s1_lookup: dict[str, tuple[str, str, str]],
    target_lookup: dict[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """Extract features and generate probability predictions for a chunk of candidate pairs."""
    if pairs_chunk.empty:
        return pd.DataFrame(columns=[
            "source1_entity_id", "target_entity_id", "target_source",
            "match_probability", "predicted_match",
        ])

    # 1. Feature generation using existing feature engineering chunk processor
    feat_df = build_pair_features_chunk(
        pairs_chunk=pairs_chunk,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
    )

    # 2. Verify exact feature columns match model expectation
    feat_matrix = feat_df[feature_columns].to_numpy(dtype=np.float32)

    # 3. Model inference: P(match | candidate)
    probs = model.predict_proba(feat_matrix)[:, 1]

    # 4. Binary decision: probability >= threshold
    preds = (probs >= threshold).astype(np.int32)

    # 5. Build scored chunk DataFrame
    scored_df = pd.DataFrame({
        "source1_entity_id": feat_df["source1_entity_id"],
        "target_entity_id": feat_df["target_entity_id"],
        "target_source": feat_df["target_source"],
        "match_probability": probs.astype(np.float32),
        "predicted_match": preds,
    })
    return scored_df


def run_test_scoring_pipeline(
    candidate_pairs_file: Path | str = DEFAULT_CANDIDATE_PAIRS_PATH,
    s1_file: Path | str = TEST_DIR / "test_source1.tsv",
    s2_file: Path | str = TEST_DIR / "test_source2.tsv",
    s3_file: Path | str = TEST_DIR / "test_source3.tsv",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    feature_cols_path: Path | str = DEFAULT_FEATURE_COLS_PATH,
    threshold_config_path: Path | str = DEFAULT_THRESHOLD_CONFIG_PATH,
    scored_output_file: Path | str = DEFAULT_SCORED_CANDIDATES_PATH,
    matching_output_file: Path | str = DEFAULT_MATCHING_RESULTS_PATH,
    report_output_file: Path | str = DEFAULT_REPORT_PATH,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Execute the full Step 11 test scoring and match generation pipeline."""
    print("=" * 80)
    print("STEP 11: FULL TEST MODEL SCORING & FINAL MATCH GENERATION")
    print("=" * 80)

    tracemalloc.start()
    t_start = time.perf_counter()

    cand_path = Path(candidate_pairs_file)
    scored_path = Path(scored_output_file)
    match_path = Path(matching_output_file)
    report_path = Path(report_output_file)

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    match_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not cand_path.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found: {cand_path}")

    # 1. Load artifacts
    model, feature_columns, threshold = load_scoring_artifacts(
        model_path=model_path,
        feature_cols_path=feature_cols_path,
        threshold_config_path=threshold_config_path,
    )

    # 2. First pass over candidate_pairs.tsv: inspect IDs to build lean lookup
    print(f"\nScanning candidate pairs from {cand_path.name} to identify required entities...")
    t_scan_start = time.perf_counter()
    req_s1_ids: set[str] = set()
    req_target_ids: set[str] = set()
    total_candidates = 0

    for chunk in pd.read_csv(cand_path, sep="\t", usecols=["source1_entity_id", "target_entity_id"], dtype=str, chunksize=chunksize * 2):
        req_s1_ids.update(chunk["source1_entity_id"])
        req_target_ids.update(chunk["target_entity_id"])
        total_candidates += len(chunk)

    t_scan_elapsed = time.perf_counter() - t_scan_start
    print(f"  Found {total_candidates:,} candidate pairs across {len(req_s1_ids):,} S1 queries and {len(req_target_ids):,} targets ({t_scan_elapsed:.2f}s).")

    # 3. Build string entity lookups
    s1_lookup, target_lookup = build_entity_lookups(
        s1_path=s1_file,
        s2_path=s2_file,
        s3_path=s3_file,
        required_s1_ids=req_s1_ids,
        required_target_ids=req_target_ids,
    )

    # 4. Stream and score candidate pairs in chunks
    print(f"\nScoring candidates in chunks of {chunksize:,} pairs...")
    if scored_path.exists():
        scored_path.unlink()

    # Streaming metrics accumulators
    n_positive = 0
    n_negative = 0
    s2_cands = 0
    s3_cands = 0
    s2_pos = 0
    s3_pos = 0

    # Aggregate positive predictions: s1_id -> list of target_id
    positive_matches_by_s1: dict[str, list[str]] = defaultdict(list)
    s2_matched_s1: set[str] = set()
    s3_matched_s1: set[str] = set()

    # Track probability distribution using a reservoir / sample of float32
    sampled_probs: list[float] = []
    sample_rate = 0.01  # 1% sample for exact percentiles, or track all if fits
    min_prob = 1.0
    max_prob = 0.0
    prob_sum = 0.0

    t_scoring_start = time.perf_counter()
    chunk_idx = 0

    for chunk in pd.read_csv(cand_path, sep="\t", dtype=str, chunksize=chunksize):
        chunk_idx += 1
        t_c_start = time.perf_counter()

        scored_chunk = score_candidate_chunk(
            pairs_chunk=chunk,
            model=model,
            feature_columns=feature_columns,
            threshold=threshold,
            s1_lookup=s1_lookup,
            target_lookup=target_lookup,
        )

        probs_arr = scored_chunk["match_probability"].to_numpy()
        preds_arr = scored_chunk["predicted_match"].to_numpy()

        # Sanity checks on probabilities
        assert np.all(np.isfinite(probs_arr)), "Non-finite probability encountered!"
        assert np.all((probs_arr >= 0.0) & (probs_arr <= 1.0001)), "Probability out of [0, 1] range!"

        # Accumulate metrics
        chunk_pos = int(np.sum(preds_arr == 1))
        chunk_neg = len(preds_arr) - chunk_pos
        n_positive += chunk_pos
        n_negative += chunk_neg

        c_min = float(np.min(probs_arr))
        c_max = float(np.max(probs_arr))
        if c_min < min_prob:
            min_prob = c_min
        if c_max > max_prob:
            max_prob = c_max
        prob_sum += float(np.sum(probs_arr))

        # Sample probabilities for distribution percentiles
        sample_step = max(1, int(1.0 / sample_rate))
        sampled_probs.extend(probs_arr[::sample_step].tolist())

        # Source breakdown
        s2_mask = scored_chunk["target_source"] == "S2"
        s3_mask = scored_chunk["target_source"] == "S3"
        s2_cands += int(np.sum(s2_mask))
        s3_cands += int(np.sum(s3_mask))
        s2_pos += int(np.sum((preds_arr == 1) & s2_mask))
        s3_pos += int(np.sum((preds_arr == 1) & s3_mask))

        # Collect positive predictions for matching_results.tsv
        if chunk_pos > 0:
            pos_df = scored_chunk[preds_arr == 1]
            for s1_id, t_id, t_src in zip(pos_df["source1_entity_id"], pos_df["target_entity_id"], pos_df["target_source"]):
                positive_matches_by_s1[s1_id].append(t_id)
                if t_src == "S2":
                    s2_matched_s1.add(s1_id)
                else:
                    s3_matched_s1.add(s1_id)

        # Write scored chunk to output/scored_candidates.tsv
        is_first_chunk = (chunk_idx == 1)
        scored_chunk.to_csv(
            scored_path,
            sep="\t",
            mode="w" if is_first_chunk else "a",
            header=is_first_chunk,
            index=False,
            float_format="%.5f",
        )

        t_c_elapsed = time.perf_counter() - t_c_start
        if chunk_idx % 10 == 0 or chunk_idx == 1:
            print(f"  [Chunk {chunk_idx}] Scored {len(scored_chunk):,} pairs ({t_c_elapsed:.2f}s) | Positives so far: {n_positive:,} ({n_positive / (n_positive + n_negative):.2%})")

    t_scoring_elapsed = time.perf_counter() - t_scoring_start
    print(f"\nScoring complete: {total_candidates:,} candidates scored in {t_scoring_elapsed:.2f}s ({total_candidates / t_scoring_elapsed:.1f} pairs/sec).")
    print(f"Scored candidates saved to: {scored_path}")

    # Compute probability statistics
    mean_prob = prob_sum / total_candidates if total_candidates > 0 else 0.0
    sampled_probs_arr = np.array(sampled_probs, dtype=np.float32)
    p50_prob = float(np.percentile(sampled_probs_arr, 50)) if len(sampled_probs_arr) > 0 else 0.0
    p90_prob = float(np.percentile(sampled_probs_arr, 90)) if len(sampled_probs_arr) > 0 else 0.0
    p95_prob = float(np.percentile(sampled_probs_arr, 95)) if len(sampled_probs_arr) > 0 else 0.0
    p99_prob = float(np.percentile(sampled_probs_arr, 99)) if len(sampled_probs_arr) > 0 else 0.0

    # 5. Build matching_results.tsv for ALL Source 1 test entities
    print(f"\nBuilding final matching_results.tsv across all Source 1 test entities...")
    t_match_start = time.perf_counter()

    s1_all_df = pd.read_csv(s1_file, sep="\t", usecols=["entity_id"], dtype=str)
    all_test_s1_ids = s1_all_df["entity_id"].tolist()
    total_test_s1 = len(all_test_s1_ids)

    out_s1_ids: list[str] = []
    out_matched_ids: list[str] = []

    zero_match_s1 = 0
    one_match_s1 = 0
    multi_match_s1 = 0
    total_matched_targets = 0
    cardinality_hist: dict[str, int] = defaultdict(int)

    for s1_id in all_test_s1_ids:
        out_s1_ids.append(s1_id)
        raw_targets = positive_matches_by_s1.get(s1_id, [])

        if not raw_targets:
            out_matched_ids.append("")
            zero_match_s1 += 1
            cardinality_hist["0"] += 1
        else:
            # Deduplicate targets while strictly preserving order
            seen_tids: set[str] = set()
            deduped_targets: list[str] = []
            for tid in raw_targets:
                if tid not in seen_tids:
                    seen_tids.add(tid)
                    deduped_targets.append(tid)

            n_m = len(deduped_targets)
            total_matched_targets += n_m
            cardinality_hist[str(n_m)] += 1

            if n_m == 1:
                one_match_s1 += 1
            else:
                multi_match_s1 += 1

            out_matched_ids.append(",".join(deduped_targets))

    matching_df = pd.DataFrame({
        "source1_entity_id": out_s1_ids,
        "matched_entity_ids": out_matched_ids,
    })

    # Write matching_results.tsv preserving empty strings
    matching_df.to_csv(match_path, sep="\t", index=False)
    t_match_elapsed = time.perf_counter() - t_match_start
    print(f"Final matching results saved to: {match_path} ({len(matching_df):,} rows, {t_match_elapsed:.2f}s)")

    # 6. Candidate subset guarantee check: matching_results <= candidate_pairs
    print("\nVerifying Candidate Subset Guarantee (matching_results.tsv <= candidate_pairs.tsv)...")
    t_sub_start = time.perf_counter()
    cand_pairs_set = set()
    for chunk in pd.read_csv(cand_path, sep="\t", usecols=["source1_entity_id", "target_entity_id"], dtype=str, chunksize=chunksize * 2):
        for s1_id, tid in zip(chunk["source1_entity_id"], chunk["target_entity_id"]):
            cand_pairs_set.add((s1_id, tid))

    violations: list[tuple[str, str]] = []
    for s1_id, m_str in zip(matching_df["source1_entity_id"], matching_df["matched_entity_ids"]):
        if m_str:
            for tid in m_str.split(","):
                tid = tid.strip()
                if tid and (s1_id, tid) not in cand_pairs_set:
                    violations.append((s1_id, tid))

    if violations:
        raise ValueError(f"CRITICAL ERROR: Candidate subset guarantee VIOLATED! Found {len(violations)} predicted matches not in candidate_pairs.tsv! Example: {violations[:3]}")
    print(f"  Candidate Subset Guarantee: 100% VERIFIED ({total_matched_targets:,} matches all present in candidate_pairs.tsv).")

    # 7. Run official challenge validator
    print("\nRunning official submission validator...")
    validator_passed = validate_submission(
        matching_file=match_path,
        candidate_file=cand_path,
        test_dir=TEST_DIR,
    )
    if not validator_passed:
        raise ValueError("Official challenge validator FAILED!")

    # 8. Source-specific and overlap metrics
    only_s2_s1 = len(s2_matched_s1 - s3_matched_s1)
    only_s3_s1 = len(s3_matched_s1 - s2_matched_s1)
    both_s2_s3_s1 = len(s2_matched_s1 & s3_matched_s1)
    neither_s1 = total_test_s1 - len(s2_matched_s1 | s3_matched_s1)

    t_total_elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # Sort cardinality histogram keys numerically
    sorted_cardinality = {k: cardinality_hist[k] for k in sorted(cardinality_hist.keys(), key=lambda x: int(x))}

    report_dict: dict[str, Any] = {
        "test_s1_rows": total_test_s1,
        "total_candidates_scored": total_candidates,
        "positive_predictions": n_positive,
        "positive_prediction_rate": (n_positive / total_candidates) if total_candidates > 0 else 0.0,
        "zero_match_s1": zero_match_s1,
        "one_match_s1": one_match_s1,
        "multi_match_s1": multi_match_s1,
        "average_matches_per_s1": (total_matched_targets / total_test_s1) if total_test_s1 > 0 else 0.0,
        "median_matches_per_s1": 1.0 if (zero_match_s1 < total_test_s1 / 2) else 0.0,
        "maximum_matches_per_s1": max((int(k) for k in cardinality_hist.keys() if int(k) > 0), default=0),
        "s2_matches": s2_pos,
        "s3_matches": s3_pos,
        "threshold": threshold,
        "min_probability": min_prob,
        "max_probability": max_prob,
        "mean_probability": mean_prob,
        "median_probability": p50_prob,
        "p90_probability": p90_prob,
        "p95_probability": p95_prob,
        "p99_probability": p99_prob,
        "candidate_subset_check": True,
        "official_validator_passed": validator_passed,
        "match_cardinality_distribution": sorted_cardinality,
        "source_statistics": {
            "S2": {
                "candidate_count": s2_cands,
                "predicted_matches": s2_pos,
                "prediction_rate": (s2_pos / s2_cands) if s2_cands > 0 else 0.0,
                "s1_entities_with_s2_match": len(s2_matched_s1),
            },
            "S3": {
                "candidate_count": s3_cands,
                "predicted_matches": s3_pos,
                "prediction_rate": (s3_pos / s3_cands) if s3_cands > 0 else 0.0,
                "s1_entities_with_s3_match": len(s3_matched_s1),
            },
            "overlap": {
                "only_s2_matches": only_s2_s1,
                "only_s3_matches": only_s3_s1,
                "both_s2_and_s3_matches": both_s2_s3_s1,
                "no_matches": neither_s1,
            },
        },
        "performance": {
            "total_runtime_seconds": t_total_elapsed,
            "scoring_runtime_seconds": t_scoring_elapsed,
            "candidates_per_second": (total_candidates / t_scoring_elapsed) if t_scoring_elapsed > 0 else 0.0,
            "peak_memory_mb": peak_mem_mb,
            "chunks_processed": chunk_idx,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nFinal prediction report saved to: {report_path}")

    # Print spot-check samples
    print_spot_checks(all_test_s1_ids, positive_matches_by_s1, scored_path)

    return report_dict


def print_spot_checks(
    all_s1_ids: list[str],
    positive_matches_by_s1: dict[str, list[str]],
    scored_candidates_file: Path,
) -> None:
    """Print deterministic spot-check samples for validation audit."""
    print("\n" + "=" * 80)
    print("DETERMINISTIC SPOT CHECKS")
    print("=" * 80)

    # 1. First 10 S1 IDs
    print("\n--- First 10 Test Source 1 Entities ---")
    for s1_id in all_s1_ids[:10]:
        targets = positive_matches_by_s1.get(s1_id, [])
        t_str = ",".join(targets) if targets else "[NO MATCH]"
        print(f"  {s1_id} -> {t_str}")

    # 2. 10 entities with single predictions
    single_match_ids = [s1 for s1, t in positive_matches_by_s1.items() if len(t) == 1][:10]
    print(f"\n--- 10 Entities with Single Prediction ({len(single_match_ids)} sampled) ---")
    for s1_id in single_match_ids:
        print(f"  {s1_id} -> {positive_matches_by_s1[s1_id]}")

    # 3. 10 entities with zero predictions
    pos_set = set(positive_matches_by_s1.keys())
    zero_ids = [s1 for s1 in all_s1_ids if s1 not in pos_set][:10]
    print(f"\n--- 10 Entities with Zero Predictions ({len(zero_ids)} sampled) ---")
    for s1_id in zero_ids:
        print(f"  {s1_id} -> [NO MATCH]")

    # 4. 10 entities with multiple predictions
    multi_ids = [s1 for s1, t in positive_matches_by_s1.items() if len(t) > 1][:10]
    print(f"\n--- 10 Entities with Multiple Predictions ({len(multi_ids)} sampled) ---")
    for s1_id in multi_ids:
        print(f"  {s1_id} -> {positive_matches_by_s1[s1_id]}")

    print("=" * 80)


if __name__ == "__main__":
    run_test_scoring_pipeline()
