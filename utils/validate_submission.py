#!/usr/bin/env python3
"""Official challenge submission validator for Business Entity Resolution.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def validate_submission(
    matching_file: Path | str,
    candidate_file: Path | str,
    test_dir: Path | str,
) -> bool:
    matching_path = Path(matching_file)
    candidate_path = Path(candidate_file)
    test_path = Path(test_dir)

    print("=" * 80)
    print("OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 80)
    print(f"Matching Results: {matching_path}")
    print(f"Candidate Pairs:  {candidate_path}")
    print(f"Test Directory:   {test_path}")
    print("-" * 80)

    # 1. File existence
    if not matching_path.is_file():
        print(f"ERROR: Matching results file not found: {matching_path}", file=sys.stderr)
        return False
    if not candidate_path.is_file():
        print(f"ERROR: Candidate pairs file not found: {candidate_path}", file=sys.stderr)
        return False
    
    s1_path = test_path / "test_source1.tsv"
    s2_path = test_path / "test_source2.tsv"
    s3_path = test_path / "test_source3.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.is_file():
            print(f"ERROR: Test source file not found: {p}", file=sys.stderr)
            return False

    # 2. Load ground-truth valid test IDs
    print("Loading valid test entity IDs...")
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)
    valid_s1_ids = set(s1_df["entity_id"].dropna())
    total_test_s1 = len(valid_s1_ids)

    s2_df = pd.read_csv(s2_path, sep="\t", usecols=["entity_id"], dtype=str)
    valid_s2_ids = set(s2_df["entity_id"].dropna())

    s3_df = pd.read_csv(s3_path, sep="\t", usecols=["entity_id"], dtype=str)
    valid_s3_ids = set(s3_df["entity_id"].dropna())
    valid_target_ids = valid_s2_ids | valid_s3_ids

    print(f"  Valid Test Source 1 entities: {total_test_s1:,}")
    print(f"  Valid Test Source 2 entities: {len(valid_s2_ids):,}")
    print(f"  Valid Test Source 3 entities: {len(valid_s3_ids):,}")
    print(f"  Total Valid Target entities:  {len(valid_target_ids):,}")

    # 3. Validate Matching Results TSV
    print("\nValidating matching_results.tsv...")
    match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    
    req_match_cols = ["source1_entity_id", "matched_entity_ids"]
    for col in req_match_cols:
        if col not in match_df.columns:
            print(f"ERROR: Matching results missing required column: {col}", file=sys.stderr)
            return False

    # Check A: Row count == test_source1 rows
    n_match_rows = len(match_df)
    if n_match_rows != total_test_s1:
        print(f"ERROR: Matching rows ({n_match_rows:,}) does not match test S1 rows ({total_test_s1:,})!", file=sys.stderr)
        return False
    print(f"  Row count matches test S1 count: {n_match_rows:,}")

    # Check B: All S1 IDs unique
    match_s1_ids = match_df["source1_entity_id"].tolist()
    unique_match_s1 = set(match_s1_ids)
    if len(unique_match_s1) != n_match_rows:
        n_dups = n_match_rows - len(unique_match_s1)
        print(f"ERROR: Found {n_dups:,} duplicate Source 1 IDs in matching results!", file=sys.stderr)
        return False
    print("  All Source 1 IDs are strictly unique.")

    # Check C: All S1 IDs belong to test_source1
    if unique_match_s1 != valid_s1_ids:
        missing_s1 = valid_s1_ids - unique_match_s1
        extra_s1 = unique_match_s1 - valid_s1_ids
        if missing_s1:
            print(f"ERROR: {len(missing_s1)} test S1 IDs missing from matching results!", file=sys.stderr)
        if extra_s1:
            print(f"ERROR: {len(extra_s1)} unknown S1 IDs present in matching results!", file=sys.stderr)
        return False
    print("  All Source 1 IDs match test_source1 exactly (100% 1-to-1 coverage).")

    # Check D, E, F: Target validity, no intra-S1 dups, collect predicted pairs
    print("  Checking matches against target entities...")
    total_matched_targets = 0
    zero_match_count = 0
    single_match_count = 0
    multi_match_count = 0

    invalid_target_ids: list[str] = []
    intra_s1_duplicates: list[tuple[str, str]] = []
    s1_in_target_violations: list[tuple[str, str]] = []
    needed_candidate_pairs: set[tuple[str, str]] = set()

    for s1_id, matched_str in zip(match_df["source1_entity_id"], match_df["matched_entity_ids"]):
        matched_str = matched_str.strip()
        if not matched_str:
            zero_match_count += 1
            continue

        target_ids = [t.strip() for t in matched_str.split(",") if t.strip()]
        if not target_ids:
            zero_match_count += 1
            continue

        if len(target_ids) == 1:
            single_match_count += 1
        else:
            multi_match_count += 1

        seen_targets_in_row = set()
        for tid in target_ids:
            total_matched_targets += 1

            if tid.startswith("S1-") or tid in valid_s1_ids:
                s1_in_target_violations.append((s1_id, tid))

            if tid not in valid_target_ids:
                invalid_target_ids.append(tid)

            if tid in seen_targets_in_row:
                intra_s1_duplicates.append((s1_id, tid))
            seen_targets_in_row.add(tid)

            needed_candidate_pairs.add((s1_id, tid))

    del match_df

    if s1_in_target_violations:
        print(f"ERROR: Found {len(s1_in_target_violations)} Source 1 IDs inside matched_entity_ids! (e.g. {s1_in_target_violations[:3]})", file=sys.stderr)
        return False

    if invalid_target_ids:
        print(f"ERROR: Found {len(invalid_target_ids)} invalid target IDs not in test_source2 or test_source3! (e.g. {invalid_target_ids[:3]})", file=sys.stderr)
        return False

    if intra_s1_duplicates:
        print(f"ERROR: Found {len(intra_s1_duplicates)} duplicate target IDs within single S1 records! (e.g. {intra_s1_duplicates[:3]})", file=sys.stderr)
        return False

    # 4. Validate Candidate Pairs File Schema, Integrity, and Subset Guarantee (streamed)
    print("\nValidating candidate_pairs.tsv in bounded chunks...")
    req_cand_cols = ["source1_entity_id", "target_entity_id", "target_source"]
    total_candidates = 0
    duplicate_candidates = 0
    invalid_cand_s1_count = 0
    invalid_cand_target_count = 0
    last_pair = None
    is_first_chunk = True

    for chunk in pd.read_csv(candidate_path, sep="\t", dtype=str, chunksize=500_000):
        if is_first_chunk:
            for col in req_cand_cols:
                if col not in chunk.columns:
                    print(f"ERROR: Candidate pairs file missing column: {col}", file=sys.stderr)
                    return False
            is_first_chunk = False

        total_candidates += len(chunk)
        for s1_id, tid in zip(chunk["source1_entity_id"], chunk["target_entity_id"]):
            pair = (s1_id, tid)
            if pair == last_pair:
                duplicate_candidates += 1
            last_pair = pair

            if s1_id not in valid_s1_ids:
                invalid_cand_s1_count += 1
            if tid not in valid_target_ids:
                invalid_cand_target_count += 1

        needed_candidate_pairs.difference_update(zip(chunk["source1_entity_id"], chunk["target_entity_id"]))

    print(f"  Candidate rows: {total_candidates:,}")

    if duplicate_candidates > 0:
        print(f"ERROR: Found {duplicate_candidates:,} duplicate candidate pairs!", file=sys.stderr)
        return False

    if invalid_cand_s1_count > 0:
        print(f"ERROR: Candidate pairs contain {invalid_cand_s1_count:,} unknown Source 1 IDs!", file=sys.stderr)
        return False

    if invalid_cand_target_count > 0:
        print(f"ERROR: Candidate pairs contain {invalid_cand_target_count:,} unknown Target IDs!", file=sys.stderr)
        return False

    if needed_candidate_pairs:
        print(f"CRITICAL ERROR: Candidate subset guarantee VIOLATED! {len(needed_candidate_pairs)} predicted matches not in candidate_pairs.tsv! (e.g. {list(needed_candidate_pairs)[:3]})", file=sys.stderr)
        return False

    print("\n" + "=" * 80)
    print("SUBMISSION VALIDATION PASSED!")
    print("=" * 80)
    print(f"  Test Source 1 Records:    {n_match_rows:,}")
    print(f"  Zero-Match S1 Count:       {zero_match_count:,} ({zero_match_count / n_match_rows:.2%})")
    print(f"  Single-Match S1 Count:     {single_match_count:,} ({single_match_count / n_match_rows:.2%})")
    print(f"  Multi-Match S1 Count:      {multi_match_count:,} ({multi_match_count / n_match_rows:.2%})")
    print(f"  Total Matched Targets:     {total_matched_targets:,}")
    print(f"  Average Matches per S1:    {total_matched_targets / n_match_rows:.4f}")
    print("  Candidate Subset Guarantee: PASSED (100% of matches are in candidate_pairs.tsv)")
    print("=" * 80)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Official challenge submission validator.")
    parser.add_argument("--matching", type=str, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test dataset directory")
    args = parser.parse_args()

    success = validate_submission(
        matching_file=args.matching,
        candidate_file=args.candidate,
        test_dir=args.test_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
