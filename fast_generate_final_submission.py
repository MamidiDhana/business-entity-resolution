"""Fast, memory-safe generator for output/matching_results.tsv and output/candidate_pairs.tsv.

1. Streams all completed batch files from output/tmp/ directly into output/candidate_pairs.tsv.
2. Extracts high-confidence matches (top match per target source) for matching_results.tsv.
3. Formats output/matching_results.tsv for all 1,732,544 test Source 1 queries.
4. Audits compliance using utils/validate_submission.py.
"""

from __future__ import annotations

import glob
import sys
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TMP_DIR = OUTPUT_DIR / "tmp"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


def main():
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FAST FINAL SUBMISSION GENERATION")
    print("=" * 80)

    # 1. Identify all completed candidate batch files in output/tmp/
    batch_files = sorted(
        glob.glob(str(TMP_DIR / "candidate_pairs_*_batch_*.tsv")),
        key=lambda x: (
            Path(x).stem.split("_")[2],
            int(Path(x).stem.split("_")[-1])
        )
    )
    print(f"Found {len(batch_files)} completed candidate batch files in {TMP_DIR.name}/.")

    # 2. Stream candidates into candidate_pairs.tsv while extracting matches
    print("\nStreaming candidates into output/candidate_pairs.tsv and extracting top matches...")
    t0 = time.perf_counter()
    
    if CANDIDATE_FILE.exists():
        CANDIDATE_FILE.unlink()

    s1_matches: dict[str, list[str]] = defaultdict(list)
    total_candidates_written = 0
    is_first = True

    for b_idx, bf in enumerate(batch_files, 1):
        # Read batch
        df = pd.read_csv(
            bf,
            sep="\t",
            usecols=["source1_entity_id", "target_entity_id", "target_source", "match_count"],
            dtype={"source1_entity_id": str, "target_entity_id": str, "target_source": str, "match_count": int},
        )
        if df.empty:
            continue

        # Sort within batch by s1, match_count desc to find best match per target source
        df.sort_values(
            by=["source1_entity_id", "match_count"],
            ascending=[True, False],
            inplace=True,
        )

        # For matching results: pick top 1 target per source (S2 and S3)
        top_per_source = df.groupby(["source1_entity_id", "target_source"], as_index=False).first()
        for s1, tid in zip(top_per_source["source1_entity_id"], top_per_source["target_entity_id"]):
            s1_matches[s1].append(tid)

        # Ensure batch is sorted deterministically for candidate_pairs.tsv
        df.sort_values(
            by=["source1_entity_id", "target_source", "target_entity_id"],
            ascending=[True, True, True],
            inplace=True,
        )

        # Write to candidate_pairs.tsv (streaming to keep memory bounded)
        out_cols = ["source1_entity_id", "target_entity_id", "target_source"]
        df[out_cols].to_csv(
            CANDIDATE_FILE,
            sep="\t",
            mode="w" if is_first else "a",
            header=is_first,
            index=False,
        )
        is_first = False
        total_candidates_written += len(df)

        del df, top_per_source
        if b_idx % 5 == 0 or b_idx == len(batch_files):
            print(f"  Processed {b_idx}/{len(batch_files)} batches ({total_candidates_written:,} candidates written)...")

    print(f"Candidate pairs saved: {total_candidates_written:,} rows in {time.perf_counter() - t0:.1f}s.")
    print(f"Queries with matches extracted: {len(s1_matches):,}.")

    # 3. Build matching_results.tsv across ALL 1,732,544 test Source 1 entities
    print("\nBuilding output/matching_results.tsv across all Source 1 entities...")
    t_match = time.perf_counter()
    s1_df = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", usecols=["entity_id"], dtype=str)
    all_s1_ids = s1_df["entity_id"].tolist()
    total_s1 = len(all_s1_ids)

    out_s1: list[str] = []
    out_matches: list[str] = []
    match_count = 0
    zero_count = 0

    for s1 in all_s1_ids:
        out_s1.append(s1)
        m_list = s1_matches.get(s1, [])
        if m_list:
            seen = set()
            deduped = [m for m in m_list if not (m in seen or seen.add(m))]
            out_matches.append(",".join(deduped))
            match_count += 1
        else:
            out_matches.append("")
            zero_count += 1

    match_df = pd.DataFrame({
        "source1_entity_id": out_s1,
        "matched_entity_ids": out_matches,
    })
    match_df.to_csv(MATCHING_FILE, sep="\t", index=False)
    del match_df, out_s1, out_matches, s1_matches, s1_df
    print(f"Saved {total_s1:,} rows to {MATCHING_FILE} in {time.perf_counter() - t_match:.1f}s.")
    print(f"  Queries with matches:    {match_count:,} ({match_count / total_s1:.1%})")
    print(f"  Queries without matches: {zero_count:,} ({zero_count / total_s1:.1%})")

    # 4. Run official submission validator
    print("\nRunning official submission validator...")
    from utils.validate_submission import validate_submission
    valid = validate_submission(
        matching_file=MATCHING_FILE,
        candidate_file=CANDIDATE_FILE,
        test_dir=TEST_DIR,
    )
    
    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    if valid:
        print(f"SUCCESS: Submission generated and validated in {elapsed:.1f}s!")
        print(f"Matching Results:  {MATCHING_FILE} ({MATCHING_FILE.stat().st_size:,} bytes)")
        print(f"Candidate Pairs:   {CANDIDATE_FILE} ({CANDIDATE_FILE.stat().st_size:,} bytes)")
    else:
        print(f"ERROR: Validation failed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
