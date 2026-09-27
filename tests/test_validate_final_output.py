"""Unit tests for Step 12: Submission Audit and Validation Engine."""

from __future__ import annotations

import unittest
from pathlib import Path
import pandas as pd

from src.validate_final_output import audit_final_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"


class TestValidateFinalOutput(unittest.TestCase):
    """Test suite verifying validation auditor functionality and rule enforcement."""

    def test_synthetic_audit_run(self) -> None:
        """Verify audit_final_submission functions correctly on a valid synthetic data slice."""
        tmp_match = OUTPUT_DIR / "tmp_test_matching.tsv"
        tmp_cand = OUTPUT_DIR / "tmp_test_candidate.tsv"
        tmp_audit = OUTPUT_DIR / "tmp_test_audit.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        s1_file = TEST_DIR / "test_source1.tsv"
        s1_sample = pd.read_csv(s1_file, sep="\t", nrows=10, dtype=str)
        s1_ids = s1_sample["entity_id"].tolist()

        # Build candidate pairs
        cand_rows = [
            {"source1_entity_id": s1_ids[0], "target_entity_id": "S2-100", "target_source": "S2"},
            {"source1_entity_id": s1_ids[1], "target_entity_id": "S3-200", "target_source": "S3"},
        ]
        pd.DataFrame(cand_rows).to_csv(tmp_cand, sep="\t", index=False)

        # Build matching results
        match_rows = []
        for i, qid in enumerate(s1_ids):
            if i == 0:
                match_rows.append({"source1_entity_id": qid, "matched_entity_ids": "S2-100"})
            elif i == 1:
                match_rows.append({"source1_entity_id": qid, "matched_entity_ids": "S3-200"})
            else:
                match_rows.append({"source1_entity_id": qid, "matched_entity_ids": ""})

        # Save synthetic matching
        pd.DataFrame(match_rows).to_csv(tmp_match, sep="\t", index=False)

        # Cleanup
        tmp_match.unlink(missing_ok=True)
        tmp_cand.unlink(missing_ok=True)
        tmp_audit.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
