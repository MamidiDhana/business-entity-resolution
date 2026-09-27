"""End-to-End Orchestrator for Step 10 & Step 11.

Executes:
  1. Step 10: Test-Set Candidate Generation Pipeline -> output/candidate_pairs.tsv
  2. Step 11: Test Model Scoring & Match Generation -> output/matching_results.tsv, output/scored_candidates.tsv
  3. Validation & Reporting -> output/candidate_generation_report.json, output/final_prediction_report.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.test_candidate_generation import run_full_test_candidate_generation_pipeline
from src.test_scoring import run_test_scoring_pipeline
from src.validate_final_output import audit_final_submission


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 80)
    print("STARTING INTEGRATED STEP 10, STEP 11 & STEP 12 PIPELINE EXECUTION")
    print("=" * 80)

    # --- STEP 10: Candidate Generation ---
    print("\n>>> LAUNCHING STEP 10: CANDIDATE GENERATION...")
    t10_start = time.perf_counter()
    cand_report = run_full_test_candidate_generation_pipeline(max_query_batch_size=50_000)
    t10_elapsed = time.perf_counter() - t10_start
    print(f"\n>>> STEP 10 COMPLETED IN {t10_elapsed:.2f}s ({t10_elapsed / 60:.2f} min).")

    # --- STEP 11: Model Scoring & Matching Results ---
    print("\n>>> LAUNCHING STEP 11: MODEL SCORING & FINAL MATCH GENERATION...")
    t11_start = time.perf_counter()
    scoring_report = run_test_scoring_pipeline(chunksize=100_000)
    t11_elapsed = time.perf_counter() - t11_start
    print(f"\n>>> STEP 11 COMPLETED IN {t11_elapsed:.2f}s ({t11_elapsed / 60:.2f} min).")

    # --- STEP 12: Final Submission Audit & Official Validation ---
    print("\n>>> LAUNCHING STEP 12: FINAL SUBMISSION AUDIT...")
    t12_start = time.perf_counter()
    audit_report = audit_final_submission()
    t12_elapsed = time.perf_counter() - t12_start
    print(f"\n>>> STEP 12 COMPLETED IN {t12_elapsed:.2f}s ({t12_elapsed / 60:.2f} min).")

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL PIPELINES COMPLETED SUCCESSFULLY IN {t_total:.2f}s ({t_total / 60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
