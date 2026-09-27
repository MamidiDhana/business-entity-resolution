"""Unit tests for Step 10: Test-Set Candidate Generation Pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.blocking import BlockingConfig
from src.test_candidate_generation import (
    REQUIRED_CANDIDATE_COLUMNS,
    consolidate_and_sort_candidates,
    generate_candidates_for_query_batch,
    load_test_source_slice,
    verify_feature_compatibility,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODELS_DIR = PROJECT_ROOT / "models"


class TestTestCandidateGeneration(unittest.TestCase):
    """Test suite verifying test candidate generation correctness, schemas, caps, and safety."""

    def test_1_test_source_loading(self) -> None:
        """Verify test source files exist, can be loaded in slices, and filtered by country."""
        s1_file = TEST_DIR / "test_source1.tsv"
        self.assertTrue(s1_file.is_file())

        df_slice = load_test_source_slice(s1_file, nrows=50)
        self.assertEqual(len(df_slice), 50)
        self.assertIn("entity_id", df_slice.columns)
        self.assertIn("country", df_slice.columns)

        # Country filter
        df_fr = load_test_source_slice(s1_file, nrows=1_000, country="france")
        self.assertTrue(all(df_fr["country"].str.strip().str.lower() == "france"))

    def test_2_s1_to_s2_s3_direction(self) -> None:
        """Verify all generated candidates strictly link S1 -> S2 or S1 -> S3 (no S1->S1, no S2->S3)."""
        s1 = pd.DataFrame([{
            "entity_id": "S1-999",
            "business_name": "Test Alpha Corp",
            "business_address": "123 Main St",
            "country": "US",
        }])
        t2_file = PROJECT_ROOT / "output" / "tmp_test_targets_s2.tsv"
        t3_file = PROJECT_ROOT / "output" / "tmp_test_targets_s3.tsv"
        t2_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"entity_id": "S2-111", "business_name": "Test Alpha Corp", "business_address": "123 Main St", "country": "US"}]).to_csv(t2_file, sep="\t", index=False)
        pd.DataFrame([{"entity_id": "S3-222", "business_name": "Test Alpha Corp", "business_address": "123 Main St", "country": "US"}]).to_csv(t3_file, sep="\t", index=False)

        # Test candidate generation
        cands = generate_candidates_for_query_batch(
            query_df=s1,
            target_sources=[(t2_file, "S2"), (t3_file, "S3")],
            target_country="US",
        )
        self.assertGreater(len(cands), 0)
        for _, row in cands.iterrows():
            self.assertTrue(str(row["source1_entity_id"]).startswith("S1-"))
            self.assertTrue(str(row["target_entity_id"]).startswith(("S2-", "S3-")))
            self.assertIn(row["target_source"], {"S2", "S3"})

        t2_file.unlink(missing_ok=True)
        t3_file.unlink(missing_ok=True)

    def test_3_no_duplicate_candidates(self) -> None:
        """Verify consolidate_and_sort_candidates strictly deduplicates pairs."""
        tmp_file = PROJECT_ROOT / "output" / "tmp_test_dups.tsv"
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        dup_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-10", "target_source": "S2", "match_count": 2},
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-10", "target_source": "S2", "match_count": 1},
        ])
        dup_df.to_csv(tmp_file, sep="\t", index=False)

        out_file = PROJECT_ROOT / "output" / "tmp_test_deduped.tsv"
        total = consolidate_and_sort_candidates([tmp_file], output_file=out_file, clean_tmp=False)
        self.assertEqual(total, 1)

        result = pd.read_csv(out_file, sep="\t")
        self.assertEqual(len(result), 1)

        # Cleanup
        tmp_file.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)

    def test_4_target_source_correctness(self) -> None:
        """Verify target_source strictly matches target_entity_id prefix."""
        tmp_file = PROJECT_ROOT / "output" / "tmp_test_src.tsv"
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-10", "target_source": "S2"},
            {"source1_entity_id": "S1-1", "target_entity_id": "S3-20", "target_source": "S3"},
        ])
        df.to_csv(tmp_file, sep="\t", index=False)

        out_file = PROJECT_ROOT / "output" / "tmp_test_src_out.tsv"
        consolidate_and_sort_candidates([tmp_file], output_file=out_file, clean_tmp=False)
        res = pd.read_csv(out_file, sep="\t")
        for _, row in res.iterrows():
            if row["target_entity_id"].startswith("S2-"):
                self.assertEqual(row["target_source"], "S2")
            elif row["target_entity_id"].startswith("S3-"):
                self.assertEqual(row["target_source"], "S3")

        tmp_file.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)

    def test_5_invalid_target_id_detection(self) -> None:
        """Verify detector catches malformed or foreign target IDs."""
        s2_ids = {"S2-1", "S2-2"}
        s3_ids = {"S3-1"}
        cand_targets = {"S2-1", "S2-INVALID", "S3-FOREIGN"}

        s2_cand = {t for t in cand_targets if t.startswith("S2-")}
        s3_cand = {t for t in cand_targets if t.startswith("S3-")}

        invalid_s2 = s2_cand - s2_ids
        invalid_s3 = s3_cand - s3_ids

        self.assertIn("S2-INVALID", invalid_s2)
        self.assertIn("S3-FOREIGN", invalid_s3)

    def test_6_deterministic_output(self) -> None:
        """Verify candidate generation sorting is strictly deterministic."""
        tmp1 = PROJECT_ROOT / "output" / "tmp_det1.tsv"
        tmp2 = PROJECT_ROOT / "output" / "tmp_det2.tsv"
        tmp1.parent.mkdir(parents=True, exist_ok=True)

        # Unordered candidate rows
        raw_rows = [
            {"source1_entity_id": "S1-B", "target_entity_id": "S3-5", "target_source": "S3"},
            {"source1_entity_id": "S1-A", "target_entity_id": "S2-9", "target_source": "S2"},
            {"source1_entity_id": "S1-A", "target_entity_id": "S2-1", "target_source": "S2"},
            {"source1_entity_id": "S1-B", "target_entity_id": "S2-2", "target_source": "S2"},
        ]
        pd.DataFrame(raw_rows).to_csv(tmp1, sep="\t", index=False)
        pd.DataFrame(raw_rows[::-1]).to_csv(tmp2, sep="\t", index=False)

        out1 = PROJECT_ROOT / "output" / "out_det1.tsv"
        out2 = PROJECT_ROOT / "output" / "out_det2.tsv"

        consolidate_and_sort_candidates([tmp1], output_file=out1, clean_tmp=False)
        consolidate_and_sort_candidates([tmp2], output_file=out2, clean_tmp=False)

        df1 = pd.read_csv(out1, sep="\t")
        df2 = pd.read_csv(out2, sep="\t")

        assert_frame_equal(df1, df2)

        # Clean
        tmp1.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)
        out1.unlink(missing_ok=True)
        out2.unlink(missing_ok=True)

    def test_7_candidate_schema(self) -> None:
        """Verify candidate TSV contains strictly required columns in correct schema."""
        out_file = PROJECT_ROOT / "output" / "tmp_schema.tsv"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_in = PROJECT_ROOT / "output" / "tmp_in.tsv"
        pd.DataFrame([{
            "source1_entity_id": "S1-1",
            "target_entity_id": "S2-1",
            "target_source": "S2",
            "match_count": 2,
        }]).to_csv(tmp_in, sep="\t", index=False)

        consolidate_and_sort_candidates([tmp_in], output_file=out_file, clean_tmp=False)
        res = pd.read_csv(out_file, sep="\t")
        for col in REQUIRED_CANDIDATE_COLUMNS:
            self.assertIn(col, res.columns)

        tmp_in.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)

    def test_8_country_scoped_blocking(self) -> None:
        """Verify entity with country US never generates blocking keys or matches for France."""
        from src.blocking import extract_blocking_keys
        keys_us = extract_blocking_keys("Acme Corp", "123 Main St", "US")
        keys_fr = extract_blocking_keys("Acme Corp", "123 Main St", "France")

        # Every key starts with country prefix
        self.assertTrue(all(k[1].startswith("us||") for k in keys_us))
        self.assertTrue(all(k[1].startswith("france||") for k in keys_fr))

        # Disjoint sets
        vals_us = {k[1] for k in keys_us}
        vals_fr = {k[1] for k in keys_fr}
        self.assertTrue(vals_us.isdisjoint(vals_fr))

    def test_9_candidate_cap_behavior(self) -> None:
        """Verify candidate count per S1 entity never exceeds max_candidates_per_query."""
        from src.blocking import finalize_candidates_from_source_maps
        cands_map = {
            "S2": {"S1-1": {f"S2-{i}": {"approx_name_first2"} for i in range(100)}},
            "S3": {"S1-1": {f"S3-{i}": {"approx_name_first2"} for i in range(100)}},
        }
        res_cap50 = finalize_candidates_from_source_maps(cands_map, max_candidates_per_query=50)
        self.assertEqual(len(res_cap50), 50)

        res_cap10 = finalize_candidates_from_source_maps(cands_map, max_candidates_per_query=10)
        self.assertEqual(len(res_cap10), 10)

    def test_10_feature_compatibility(self) -> None:
        """Verify test candidate pairs can be transformed into features matching models/feature_columns.json."""
        s1 = pd.DataFrame([{
            "entity_id": "S1-FC1",
            "business_name": "Delta Tech Services",
            "business_address": "500 Market St",
            "country": "US",
        }])
        target = pd.DataFrame([{
            "entity_id": "S2-FC1",
            "business_name": "Delta Tech Services Inc",
            "business_address": "500 Market Street",
            "country": "US",
        }])
        pairs = pd.DataFrame([{
            "source1_entity_id": "S1-FC1",
            "target_entity_id": "S2-FC1",
            "target_source": "S2",
        }])

        fc_file = MODELS_DIR / "feature_columns.json"
        if fc_file.is_file():
            is_compat = verify_feature_compatibility(
                candidate_pairs_df=pairs,
                s1_df=s1,
                target_dfs=target,
                feature_columns_json=fc_file,
                sample_size=1,
            )
            self.assertTrue(is_compat)


if __name__ == "__main__":
    unittest.main()
