"""Unit tests for Step 7: Feature Engineering pipeline."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_engineering import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_pair_features,
    build_pair_features_chunk,
    compute_country_features,
    compute_data_quality_features,
    compute_pair_features_row,
    compute_string_pair_features,
    get_feature_names,
    get_identifier_columns,
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite verifying pairwise feature generation correctness, edge cases, and safety."""

    def test_1_identical_names_and_addresses(self) -> None:
        """Verify feature outputs when names and addresses are exactly identical."""
        pairs = pd.DataFrame([{
            "source1_entity_id": "S1-1",
            "source1_source": "S1",
            "target_entity_id": "S2-1",
            "target_source": "S2",
            "business_name_s1": "Acme Global Solutions",
            "business_address_s1": "100 Market St Suite 500",
            "country_s1": "US",
            "business_name_target": "Acme Global Solutions",
            "business_address_target": "100 Market St Suite 500",
            "country_target": "US",
        }])
        feat_df = build_pair_features_chunk(pairs)

        self.assertEqual(feat_df["name_exact_match"].iloc[0], 1.0)
        self.assertEqual(feat_df["name_jaro_winkler"].iloc[0], 1.0)
        self.assertEqual(feat_df["name_edit_similarity"].iloc[0], 1.0)
        self.assertEqual(feat_df["name_token_jaccard"].iloc[0], 1.0)
        self.assertEqual(feat_df["name_length_diff"].iloc[0], 0.0)
        self.assertEqual(feat_df["name_length_ratio"].iloc[0], 1.0)

        self.assertEqual(feat_df["addr_exact_match"].iloc[0], 1.0)
        self.assertEqual(feat_df["addr_jaro_winkler"].iloc[0], 1.0)
        self.assertEqual(feat_df["addr_digit_match_ratio"].iloc[0], 1.0)
        self.assertEqual(feat_df["country_exact_match"].iloc[0], 1.0)

    def test_2_similar_names(self) -> None:
        """Verify similarity features for fuzzy/similar names."""
        feat = compute_string_pair_features("Acme Logistics International", "Akme Logistics Inter", is_address=False)
        self.assertEqual(feat["name_exact_match"], 0.0)
        self.assertGreater(feat["name_jaro_winkler"], 0.80)
        self.assertGreater(feat["name_token_sort_ratio"], 0.70)
        self.assertGreater(feat["name_token_overlap"], 0.3)

    def test_3_different_names(self) -> None:
        """Verify similarity features for completely different names."""
        feat = compute_string_pair_features("Apple Inc", "Zebra Technologies", is_address=False)
        self.assertEqual(feat["name_exact_match"], 0.0)
        self.assertLess(feat["name_jaro_winkler"], 0.5)
        self.assertEqual(feat["name_token_jaccard"], 0.0)
        self.assertEqual(feat["name_shared_token_count"], 0.0)

    def test_4_similar_addresses(self) -> None:
        """Verify address features for similar addresses with matching digits."""
        feat = compute_string_pair_features("123 Main Street Suite 400", "123 Main St Ste 400", is_address=True)
        self.assertEqual(feat["addr_exact_match"], 0.0)
        self.assertGreater(feat["addr_jaro_winkler"], 0.80)
        self.assertGreaterEqual(feat["addr_shared_digit_count"], 2.0)
        self.assertEqual(feat["addr_digit_match_ratio"], 1.0)

    def test_5_different_addresses(self) -> None:
        """Verify address features for completely different addresses."""
        feat = compute_string_pair_features("100 Market St San Francisco CA", "999 Broad Way New York NY", is_address=True)
        self.assertEqual(feat["addr_exact_match"], 0.0)
        self.assertEqual(feat["addr_token_jaccard"], 0.0)
        self.assertEqual(feat["addr_shared_digit_count"], 0.0)
        self.assertEqual(feat["addr_digit_match_ratio"], 0.0)

    def test_6_exact_country_match(self) -> None:
        """Verify country features when countries match exactly."""
        feat = compute_country_features("US", "US")
        self.assertEqual(feat["country_exact_match"], 1.0)
        self.assertEqual(feat["country_mismatch"], 0.0)
        self.assertEqual(feat["country_missing_s1"], 0.0)
        self.assertEqual(feat["country_missing_target"], 0.0)

    def test_7_country_mismatch(self) -> None:
        """Verify country features when countries disagree."""
        feat = compute_country_features("US", "CA")
        self.assertEqual(feat["country_exact_match"], 0.0)
        self.assertEqual(feat["country_mismatch"], 1.0)

    def test_8_missing_name(self) -> None:
        """Verify safe handling of missing names without 'nan' literal string leaks."""
        for val in [None, np.nan, "nan", ""]:
            feat = compute_string_pair_features(str(val) if val is not None else "", "Acme Corp", is_address=False)
            self.assertEqual(feat["name_exact_match"], 0.0)
            self.assertEqual(feat["name_jaro_winkler"], 0.0)
            self.assertEqual(feat["name_shared_token_count"], 0.0)

        qual = compute_data_quality_features("", "Acme Corp", "Addr 1", "Addr 2", "US", "US")
        self.assertEqual(qual["missing_name_s1"], 1.0)
        self.assertEqual(qual["missing_name_target"], 0.0)

    def test_9_missing_address(self) -> None:
        """Verify safe handling of missing addresses."""
        feat = compute_string_pair_features("100 Main St", "", is_address=True)
        self.assertEqual(feat["addr_exact_match"], 0.0)
        self.assertEqual(feat["addr_jaro_winkler"], 0.0)
        self.assertEqual(feat["addr_shared_digit_count"], 0.0)

        qual = compute_data_quality_features("Name 1", "Name 2", "Addr 1", "", "US", "US")
        self.assertEqual(qual["missing_addr_target"], 1.0)

    def test_10_unicode_non_ascii_text(self) -> None:
        """Verify proper feature extraction on Unicode / Cyrillic / non-ASCII text."""
        pairs = pd.DataFrame([{
            "source1_entity_id": "S1-U1",
            "source1_source": "S1",
            "target_entity_id": "S2-U1",
            "target_source": "S2",
            "business_name_s1": "ООО Вектор Инноваций",
            "business_address_s1": "ул Ленина д 15",
            "country_s1": "RU",
            "business_name_target": "ООО Вектор Инноваций",
            "business_address_target": "ул Ленина д 15",
            "country_target": "RU",
        }])
        feat_df = build_pair_features_chunk(pairs)
        self.assertEqual(feat_df["name_exact_match"].iloc[0], 1.0)
        self.assertEqual(feat_df["addr_exact_match"].iloc[0], 1.0)
        self.assertFalse(feat_df.isna().any().any())

    def test_11_multiple_candidates_for_one_source1(self) -> None:
        """Verify independent pairwise feature computation for 1-to-many candidate pairs."""
        pairs = pd.DataFrame([
            {
                "source1_entity_id": "S1-100",
                "source1_source": "S1",
                "target_entity_id": "S2-101",
                "target_source": "S2",
                "business_name_s1": "Beta Health",
                "business_address_s1": "456 Oak Ave",
                "country_s1": "GB",
                "business_name_target": "Beta Health UK",
                "business_address_target": "456 Oak Ave London",
                "country_target": "GB",
            },
            {
                "source1_entity_id": "S1-100",
                "source1_source": "S1",
                "target_entity_id": "S3-999",
                "target_source": "S3",
                "business_name_s1": "Beta Health",
                "business_address_s1": "456 Oak Ave",
                "country_s1": "GB",
                "business_name_target": "Omni Global Logistics",
                "business_address_target": "100 Pine St",
                "country_target": "US",
            },
        ])
        feat_df = build_pair_features_chunk(pairs)
        self.assertEqual(len(feat_df), 2)
        # First candidate is high similarity
        self.assertGreater(feat_df["name_jaro_winkler"].iloc[0], 0.8)
        self.assertEqual(feat_df["country_exact_match"].iloc[0], 1.0)
        # Second candidate is low similarity / non-match
        self.assertEqual(feat_df["name_token_jaccard"].iloc[1], 0.0)
        self.assertLess(feat_df["name_edit_similarity"].iloc[1], 0.4)
        self.assertLess(feat_df["name_jaro_winkler"].iloc[1], 0.6)
        self.assertEqual(feat_df["country_exact_match"].iloc[1], 0.0)

    def test_12_s2_and_s3_target_sources(self) -> None:
        """Verify that target source indicator features (S2 vs S3) are accurately encoded."""
        pairs = pd.DataFrame([
            {
                "source1_entity_id": "S1-1",
                "source1_source": "S1",
                "target_entity_id": "S2-10",
                "target_source": "S2",
                "business_name_s1": "Test Co",
                "business_address_s1": "Test Addr",
                "country_s1": "US",
                "business_name_target": "Test Co",
                "business_address_target": "Test Addr",
                "country_target": "US",
            },
            {
                "source1_entity_id": "S1-1",
                "source1_source": "S1",
                "target_entity_id": "S3-20",
                "target_source": "S3",
                "business_name_s1": "Test Co",
                "business_address_s1": "Test Addr",
                "country_s1": "US",
                "business_name_target": "Test Co",
                "business_address_target": "Test Addr",
                "country_target": "US",
            },
        ])
        feat_df = build_pair_features_chunk(pairs)
        self.assertEqual(feat_df["target_source_is_s2"].iloc[0], 1.0)
        self.assertEqual(feat_df["target_source_is_s3"].iloc[0], 0.0)
        self.assertEqual(feat_df["target_source_is_s2"].iloc[1], 0.0)
        self.assertEqual(feat_df["target_source_is_s3"].iloc[1], 1.0)

    def test_13_no_entity_id_used_as_model_feature(self) -> None:
        """Verify entity IDs are preserved solely as metadata and excluded from model features."""
        feature_names = get_feature_names()
        id_columns = get_identifier_columns()

        for col in ["source1_entity_id", "target_entity_id", "entity_id"]:
            self.assertNotIn(col, feature_names)

        self.assertIn("source1_entity_id", id_columns)
        self.assertIn("target_entity_id", id_columns)

    def test_14_no_ground_truth_label_used_as_feature(self) -> None:
        """Verify no ground truth or match label is included in feature names to prevent target leakage."""
        feature_names = get_feature_names()
        for forbidden in ["matched_entity_ids", "is_match", "label", "target", "ground_truth"]:
            self.assertNotIn(forbidden, feature_names)

    def test_15_deterministic_output(self) -> None:
        """Verify that feature generation is strictly deterministic across repeated runs."""
        pairs = pd.DataFrame([
            {
                "source1_entity_id": "S1-1",
                "source1_source": "S1",
                "target_entity_id": "S2-1",
                "target_source": "S2",
                "business_name_s1": "Alpha Delta Corp",
                "business_address_s1": "789 Highway 101",
                "country_s1": "CA",
                "business_name_target": "Alpha Delta Inc",
                "business_address_target": "789 Hwy 101",
                "country_target": "CA",
            }
        ])
        feat1 = build_pair_features_chunk(pairs)
        feat2 = build_pair_features_chunk(pairs)
        assert_frame_equal(feat1, feat2)


if __name__ == "__main__":
    unittest.main()
