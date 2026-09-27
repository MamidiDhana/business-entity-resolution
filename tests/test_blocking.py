"""Unit tests for blocking and candidate generation using small synthetic DataFrames."""

from __future__ import annotations

import unittest
import pandas as pd

from src.blocking import (
    BlockingConfig,
    evaluate_candidate_recall,
    extract_blocking_keys,
    finalize_candidates,
    generate_candidates,
)


class TestBlocking(unittest.TestCase):
    """Test suite verifying all candidate generation rules, validity, and evaluation."""

    def setUpself(self) -> None:
        pass

    def test_exact_name_match(self) -> None:
        """Verify that entities with identical business names under same country are matched."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-100",
            "business_name": "Apex Summit Solutions",
            "business_address": "123 North St",
            "country": "US",
        }])
        target_df = pd.DataFrame([{
            "entity_id": "S2-200",
            "business_name": "Apex Summit Solutions",
            "business_address": "999 Completely Different Ave",
            "country": "US",
        }])

        candidates = generate_candidates(query_df, target_df, target_source="S2")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["source1_entity_id"], "S1-100")
        self.assertEqual(candidates.iloc[0]["target_entity_id"], "S2-200")
        self.assertEqual(candidates.iloc[0]["target_source"], "S2")
        self.assertIn("exact_name", candidates.iloc[0]["matched_strategies"])

    def test_exact_address_match(self) -> None:
        """Verify that entities with identical address under same country are matched even with different names."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-101",
            "business_name": "Acme Holdings",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US",
        }])
        target_df = pd.DataFrame([{
            "entity_id": "S3-301",
            "business_name": "Drexkor Global",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US",
        }])

        candidates = generate_candidates(query_df, target_df, target_source="S3")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["source1_entity_id"], "S1-101")
        self.assertEqual(candidates.iloc[0]["target_entity_id"], "S3-301")
        self.assertEqual(candidates.iloc[0]["target_source"], "S3")
        self.assertIn("exact_addr", candidates.iloc[0]["matched_strategies"])

    def test_country_aware_blocking(self) -> None:
        """Verify that identical names/addresses in different countries are NEVER matched."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-102",
            "business_name": "National Health Clinic",
            "business_address": "100 Main Street",
            "country": "US",
        }])
        target_df = pd.DataFrame([{
            "entity_id": "S2-202",
            "business_name": "National Health Clinic",
            "business_address": "100 Main Street",
            "country": "India",  # Different country
        }])

        candidates = generate_candidates(query_df, target_df, target_source="S2")
        self.assertTrue(candidates.empty, "Entities in different countries should not be paired")

    def test_approximate_name_candidate(self) -> None:
        """Verify candidate generation for name prefix variations and compact forms."""
        # Query with legal suffix and target without suffix or with domain suffix
        query_df = pd.DataFrame([{
            "entity_id": "S1-103",
            "business_name": "Maure Williams Colombier Inc",
            "business_address": "PO Box 1",
            "country": "US",
        }])
        target_df = pd.DataFrame([
            {
                "entity_id": "S2-203",
                "business_name": "Maure Williams Colombier",  # Missing 'Inc'
                "business_address": "Different Road",
                "country": "US",
            },
            {
                "entity_id": "S3-303",
                "business_name": "maurewilliamscolombier.com",  # Domain format
                "business_address": "Other Street",
                "country": "US",
            },
        ])

        cands_s2 = generate_candidates(query_df, target_df.iloc[[0]], target_source="S2")
        self.assertEqual(len(cands_s2), 1)
        self.assertEqual(cands_s2.iloc[0]["target_entity_id"], "S2-203")

        cands_s3 = generate_candidates(query_df, target_df.iloc[[1]], target_source="S3")
        self.assertEqual(len(cands_s3), 1)
        self.assertEqual(cands_s3.iloc[0]["target_entity_id"], "S3-303")

    def test_approximate_address_candidate(self) -> None:
        """Verify address prefix and number+word matching."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-104",
            "business_name": "Alpha Beta",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US",
        }])
        target_df = pd.DataFrame([{
            "entity_id": "S3-304",
            "business_name": "Completely Different Name",
            "business_address": "85 Wanye Avenue, Ticonderoga Township, NY",  # Typo in street, matches on 85_ticonderoga
            "country": "US",
        }])

        candidates = generate_candidates(query_df, target_df, target_source="S3")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["target_entity_id"], "S3-304")

    def test_multiple_true_matches(self) -> None:
        """Verify a single Source 1 entity can match multiple targets across S2 and S3."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-105",
            "business_name": "Raj Investments LLP",
            "business_address": "6 CIT Colony, Mylapore, Chennai",
            "country": "India",
        }])
        target_s2 = pd.DataFrame([
            {
                "entity_id": "S2-205A",
                "business_name": "Raj Investments LLP",
                "business_address": "6 CIT Colony, Mylapore, Chennai",
                "country": "India",
            },
            {
                "entity_id": "S2-205B",
                "business_name": "Raj Investments",
                "business_address": "6 CIT Colony, Mylapore, Chennai",
                "country": "India",
            },
        ])
        target_s3 = pd.DataFrame([
            {
                "entity_id": "S3-305A",
                "business_name": "Raj Investments",
                "business_address": "6 CIT Colony, Mylapore, Chennai",
                "country": "India",
            },
        ])

        cands_s2 = generate_candidates(query_df, target_s2, target_source="S2")
        cands_s3 = generate_candidates(query_df, target_s3, target_source="S3")
        all_cands = pd.concat([cands_s2, cands_s3], ignore_index=True)

        target_ids = set(all_cands["target_entity_id"])
        self.assertEqual(target_ids, {"S2-205A", "S2-205B", "S3-305A"})
        self.assertEqual(len(all_cands), 3)

    def test_zero_match_source1_entity(self) -> None:
        """Verify that a query entity with no matching targets produces 0 candidates gracefully."""
        query_df = pd.DataFrame([{
            "entity_id": "S1-999",
            "business_name": "Unique Never Seen Enterprise",
            "business_address": "99999 Nowhere Land Road",
            "country": "US",
        }])
        target_df = pd.DataFrame([{
            "entity_id": "S2-888",
            "business_name": "Ordinary Bakeries",
            "business_address": "123 Baker Street",
            "country": "US",
        }])

        candidates = generate_candidates(query_df, target_df, target_source="S2")
        self.assertTrue(candidates.empty)

    def test_duplicate_candidate_removal(self) -> None:
        """Verify that a target matching multiple strategies appears only once per query."""
        raw_matches = [
            ("S1-1", "S1", "S2-2", "S2", "exact_name"),
            ("S1-1", "S1", "S2-2", "S2", "exact_addr"),
            ("S1-1", "S1", "S2-2", "S2", "approx_name_first2"),
        ]
        final = finalize_candidates(raw_matches)
        self.assertEqual(len(final), 1)
        self.assertEqual(final.iloc[0]["match_count"], 3)
        self.assertIn("exact_name", final.iloc[0]["matched_strategies"])
        self.assertIn("exact_addr", final.iloc[0]["matched_strategies"])

    def test_s1_to_s2_and_s1_to_s3_validity(self) -> None:
        """Verify candidate pairs strictly follow S1 -> S2 and S1 -> S3 conventions."""
        raw_matches = [
            ("S1-1", "S1", "S2-2", "S2", "exact_name"),
            ("S1-1", "S1", "S3-3", "S3", "exact_addr"),
            ("S2-1", "S2", "S3-3", "S3", "exact_name"),  # Invalid: S2 -> S3
            ("S1-1", "S1", "S1-2", "S1", "exact_name"),  # Invalid: S1 -> S1
        ]
        final = finalize_candidates(raw_matches)
        self.assertEqual(len(final), 2)
        for _, row in final.iterrows():
            self.assertEqual(row["source1_source"], "S1")
            self.assertTrue(row["source1_entity_id"].startswith("S1-"))
            self.assertIn(row["target_source"], {"S2", "S3"})
            self.assertTrue(row["target_entity_id"].startswith(("S2-", "S3-")))

    def test_no_s1_to_s1_pairs(self) -> None:
        """Verify S1 entities are never paired with other S1 entities."""
        raw_matches = [
            ("S1-100", "S1", "S1-101", "S1", "exact_name"),
            ("S1-100", "S1", "S1-100", "S1", "exact_addr"),
        ]
        final = finalize_candidates(raw_matches)
        self.assertTrue(final.empty)

    def test_candidate_recall_calculation(self) -> None:
        """Verify recall evaluation accounting for multiple true target matches."""
        ground_truth = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S2-11,S3-20"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-30,S3-40"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": None},  # 0 true matches
        ])
        # Suppose candidates found S2-10, S3-20 (from S1-1), S2-30 (from S1-2), plus a false candidate S2-99
        candidates = pd.DataFrame([
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-10"},
            {"source1_entity_id": "S1-1", "target_entity_id": "S3-20"},
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-99"},  # false candidate
            {"source1_entity_id": "S1-2", "target_entity_id": "S2-30"},
        ])

        eval_res = evaluate_candidate_recall(candidates, ground_truth)

        # Total true: 3 (from S1-1) + 2 (from S1-2) = 5
        self.assertEqual(eval_res["total_true_target_references"], 5)
        # S2 true: S2-10, S2-11, S2-30 = 3
        self.assertEqual(eval_res["s2_true_references"], 3)
        # S3 true: S3-20, S3-40 = 2
        self.assertEqual(eval_res["s3_true_references"], 2)

        # Found true: S2-10, S3-20, S2-30 = 3
        self.assertEqual(eval_res["found_true_target_references"], 3)
        self.assertEqual(eval_res["s2_found_references"], 2)
        self.assertEqual(eval_res["s3_found_references"], 1)

        # Recalls: overall = 3/5 = 0.60, s2 = 2/3 = 0.6667, s3 = 1/2 = 0.50
        self.assertAlmostEqual(eval_res["overall_candidate_recall"], 3 / 5)
        self.assertAlmostEqual(eval_res["s2_candidate_recall"], 2 / 3)
        self.assertAlmostEqual(eval_res["s3_candidate_recall"], 1 / 2)
        self.assertEqual(eval_res["lost_matches_count"], 2)

    def test_deterministic_output(self) -> None:
        """Verify candidate generation is strictly deterministic across runs."""
        query_df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Delta Corp", "business_address": "12 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Echo Ltd", "business_address": "99 Oak Ave", "country": "US"},
        ])
        target_df = pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Delta Corp", "business_address": "12 Main St", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Delta Corp", "business_address": "55 Other St", "country": "US"},
            {"entity_id": "S2-3", "business_name": "Echo Ltd", "business_address": "99 Oak Ave", "country": "US"},
        ])

        run1 = generate_candidates(query_df, target_df, target_source="S2")
        run2 = generate_candidates(query_df, target_df, target_source="S2")
        pd.testing.assert_frame_equal(run1, run2)


if __name__ == "__main__":
    unittest.main()
