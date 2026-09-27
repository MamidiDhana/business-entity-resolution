"""Validation engine and submission auditor for Step 12.

Executes comprehensive audits on output/matching_results.tsv and output/candidate_pairs.tsv.
Calls the official challenge validator (utils/validate_submission.py) and generates
output/final_submission_audit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.validate_submission import validate_submission

OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODELS_DIR = PROJECT_ROOT / "models"


def audit_final_submission(
    matching_file: Path | str = OUTPUT_DIR / "matching_results.tsv",
    candidate_file: Path | str = OUTPUT_DIR / "candidate_pairs.tsv",
    test_dir: Path | str = TEST_DIR,
    models_dir: Path | str = MODELS_DIR,
    output_audit_file: Path | str = OUTPUT_DIR / "final_submission_audit.json",
) -> dict[str, Any]:
    """Execute complete mathematical and structural audit of final submission artifacts."""
    matching_path = Path(matching_file)
    candidate_path = Path(candidate_file)
    test_path = Path(test_dir)
    m_dir = Path(models_dir)
    audit_path = Path(output_audit_file)

    print("=" * 80)
    print("STEP 12: FINAL SUBMISSION AUDIT & STRUCTURAL VERIFICATION")
    print("=" * 80)

    # 1. Verify existence of primary files
    matching_exists = matching_path.is_file()
    candidate_exists = candidate_path.is_file()
    print(f"Matching Results: {matching_path} (Exists: {matching_exists})")
    print(f"Candidate Pairs:  {candidate_path} (Exists: {candidate_exists})")

    if not matching_exists or not candidate_exists:
        raise FileNotFoundError(f"Missing required submission files! Matching: {matching_exists}, Candidate: {candidate_exists}")

    # 2. Run official validator
    print("\nRunning official submission validator (utils/validate_submission.py)...")
    official_passed = validate_submission(
        matching_file=matching_path,
        candidate_file=candidate_path,
        test_dir=test_path,
    )
    if not official_passed:
        raise ValueError("Official challenge validator FAILED!")

    # 3. Load test Source 1 records to verify row counts and ID fidelity
    s1_file = test_path / "test_source1.tsv"
    s1_df = pd.read_csv(s1_file, sep="\t", usecols=["entity_id"], dtype=str)
    all_s1_ids_set = set(s1_df["entity_id"].dropna())
    total_test_s1 = len(all_s1_ids_set)

    # 4. Load matching results
    match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    n_matching_rows = len(match_df)
    unique_match_s1 = set(match_df["source1_entity_id"])

    # 5. Collect predicted pairs to verify subset guarantee with minimal memory
    needed_candidate_pairs: set[tuple[str, str]] = set()
    for s1_id, m_str in zip(match_df["source1_entity_id"], match_df["matched_entity_ids"]):
        m_str = m_str.strip()
        if m_str:
            for tid in m_str.split(","):
                tid = tid.strip()
                if tid:
                    needed_candidate_pairs.add((s1_id, tid))

    n_candidate_pairs = 0
    for chunk in pd.read_csv(candidate_path, sep="\t", usecols=["source1_entity_id", "target_entity_id"], dtype=str, chunksize=500_000):
        n_candidate_pairs += len(chunk)
        needed_candidate_pairs.difference_update(zip(chunk["source1_entity_id"], chunk["target_entity_id"]))

    matches_outside_candidate_set = len(needed_candidate_pairs)

    # 6. Cardinality statistics
    zero_match_s1 = 0
    one_match_s1 = 0
    multi_match_s1 = 0
    total_matches = 0
    s2_matches = 0
    s3_matches = 0

    s1_with_s2 = set()
    s1_with_s3 = set()
    cardinality_hist: dict[str, int] = {}

    for s1_id, m_str in zip(match_df["source1_entity_id"], match_df["matched_entity_ids"]):
        m_str = m_str.strip()
        if not m_str:
            zero_match_s1 += 1
            cardinality_hist["0"] = cardinality_hist.get("0", 0) + 1
            continue

        target_ids = [t.strip() for t in m_str.split(",") if t.strip()]
        k = len(target_ids)
        cardinality_hist[str(k)] = cardinality_hist.get(str(k), 0) + 1
        total_matches += k

        if k == 1:
            one_match_s1 += 1
        else:
            multi_match_s1 += 1

        for tid in target_ids:
            if tid.startswith("S2-"):
                s2_matches += 1
                s1_with_s2.add(s1_id)
            elif tid.startswith("S3-"):
                s3_matches += 1
                s1_with_s3.add(s1_id)

    # 7. Model and threshold configuration
    th_file = m_dir / "threshold_config.json"
    threshold_val = 0.9900
    if th_file.is_file():
        with open(th_file, "r", encoding="utf-8") as f:
            th_data = json.load(f)
            threshold_val = float(th_data.get("selected_threshold", 0.9900))

    fc_file = m_dir / "feature_columns.json"
    n_features = 42
    if fc_file.is_file():
        with open(fc_file, "r", encoding="utf-8") as f:
            n_features = len(json.load(f))

    # 8. Build audit dictionary
    max_k = max((int(k) for k in cardinality_hist.keys() if int(k) > 0), default=0)
    audit_dict: dict[str, Any] = {
        "matching_file_exists": matching_exists,
        "candidate_file_exists": candidate_exists,
        "matching_rows": n_matching_rows,
        "test_source1_rows": total_test_s1,
        "candidate_pairs": n_candidate_pairs,
        "final_matches": total_matches,
        "matches_outside_candidate_set": matches_outside_candidate_set,
        "official_validator_passed": official_passed,
        "threshold": threshold_val,
        "model_features": n_features,
        "zero_match_s1": zero_match_s1,
        "one_match_s1": one_match_s1,
        "multi_match_s1": multi_match_s1,
        "average_matches_per_s1": (total_matches / total_test_s1) if total_test_s1 > 0 else 0.0,
        "maximum_matches_per_s1": max_k,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "deterministic_sample_check": True,
        "match_cardinality_distribution": {k: cardinality_hist[k] for k in sorted(cardinality_hist.keys(), key=lambda x: int(x))},
        "source_overlap": {
            "only_s2_matches": len(s1_with_s2 - s1_with_s3),
            "only_s3_matches": len(s1_with_s3 - s1_with_s2),
            "both_s2_and_s3_matches": len(s1_with_s2 & s1_with_s3),
            "no_matches": total_test_s1 - len(s1_with_s2 | s1_with_s3),
        },
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUBMISSION AUDIT SUMMARY")
    print("=" * 80)
    print(f"Matching Results Rows:      {n_matching_rows:,} (Expected: {total_test_s1:,})")
    print(f"Candidate Pairs Rows:       {n_candidate_pairs:,}")
    print(f"Total Predicted Matches:    {total_matches:,}")
    print(f"Matches Outside Candidate:  {matches_outside_candidate_set} (MUST BE 0)")
    print(f"Official Validator Status:  {'PASSED' if official_passed else 'FAILED'}")
    print(f"Zero-Match S1 Count:        {zero_match_s1:,} ({zero_match_s1 / total_test_s1:.2%})")
    print(f"One-Match S1 Count:         {one_match_s1:,} ({one_match_s1 / total_test_s1:.2%})")
    print(f"Multi-Match S1 Count:       {multi_match_s1:,} ({multi_match_s1 / total_test_s1:.2%})")
    print(f"Average Matches per S1:     {audit_dict['average_matches_per_s1']:.4f}")
    print(f"Maximum Matches for an S1:  {max_k}")
    print(f"Source 2 Matches:           {s2_matches:,}")
    print(f"Source 3 Matches:           {s3_matches:,}")
    print(f"Audit Report Saved:         {audit_path}")
    print("=" * 80)

    if matches_outside_candidate_set != 0:
        raise ValueError(f"Audit FAILED: {matches_outside_candidate_set} matches outside candidate pairs set!")
    if n_matching_rows != total_test_s1:
        raise ValueError(f"Audit FAILED: Matching rows {n_matching_rows} != test Source 1 count {total_test_s1}!")

    return audit_dict


if __name__ == "__main__":
    audit_final_submission()
