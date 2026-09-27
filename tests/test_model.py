"""Unit tests for Step 8: Model Training, Validation, and Multi-Match Metrics."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from src.feature_engineering import FEATURE_COLUMNS, IDENTIFIER_COLUMNS
from src.model import (
    EntityResolutionModel,
    ModelConfig,
    add_labels_to_candidate_pairs,
    build_ground_truth_map,
    calculate_f0_5,
    compute_classification_metrics,
    compute_multi_match_statistics,
    split_candidates_by_group,
)


def _create_synthetic_training_data(n_entities: int = 20, pairs_per_entity: int = 5) -> pd.DataFrame:
    """Helper creating a small synthetic candidate DataFrame with 42 feature columns."""
    rows = []
    rng = np.random.default_rng(42)

    for i in range(n_entities):
        s1_id = f"S1-{i}"
        for j in range(pairs_per_entity):
            target_id = f"S2-{i * 10 + j}" if j % 2 == 0 else f"S3-{i * 10 + j}"
            tsrc = "S2" if target_id.startswith("S2-") else "S3"
            
            # Make j == 0 a strong match and j > 0 weaker matches
            is_match = 1 if j == 0 else 0
            
            row = {
                "source1_entity_id": s1_id,
                "source1_source": "S1",
                "target_entity_id": target_id,
                "target_source": tsrc,
                "label": is_match,
            }
            # Fill 42 features: high for match, low for non-match
            for feat in FEATURE_COLUMNS:
                if is_match:
                    row[feat] = float(rng.uniform(0.75, 1.0))
                else:
                    row[feat] = float(rng.uniform(0.0, 0.4))
            rows.append(row)

    return pd.DataFrame(rows)


class TestEntityResolutionModel(unittest.TestCase):
    """Test suite verifying multi-match labels, group-aware split, model fitting, and F0.5 metrics."""

    def test_1_label_construction_with_zero_matches(self) -> None:
        """Verify candidate pairs receive label=0 when Source 1 has zero ground-truth matches."""
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-EMPTY1", "matched_entity_ids": ""},
            {"source1_entity_id": "S1-EMPTY2", "matched_entity_ids": np.nan},
        ])
        gt_map = build_ground_truth_map(gt_df)
        self.assertEqual(len(gt_map["S1-EMPTY1"]), 0)
        self.assertEqual(len(gt_map["S1-EMPTY2"]), 0)

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-EMPTY1", "target_entity_id": "S2-100"},
            {"source1_entity_id": "S1-EMPTY2", "target_entity_id": "S3-200"},
        ])
        labeled = add_labels_to_candidate_pairs(pairs, gt_map)
        self.assertEqual(labeled["label"].tolist(), [0, 0])

    def test_2_label_construction_with_one_match(self) -> None:
        """Verify candidate pairs are labeled 1 for exact match and 0 for non-match."""
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-SINGLE", "matched_entity_ids": "S2-TRUE"},
        ])
        gt_map = build_ground_truth_map(gt_df)
        self.assertEqual(gt_map["S1-SINGLE"], {"S2-TRUE"})

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-SINGLE", "target_entity_id": "S2-TRUE"},
            {"source1_entity_id": "S1-SINGLE", "target_entity_id": "S2-FALSE"},
        ])
        labeled = add_labels_to_candidate_pairs(pairs, gt_map)
        self.assertEqual(labeled["label"].tolist(), [1, 0])

    def test_3_label_construction_with_multiple_matches(self) -> None:
        """Verify candidate pairs for multi-match Source 1 entities are correctly labeled."""
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-MULTI", "matched_entity_ids": "S2-10, S3-20, S2-30"},
        ])
        gt_map = build_ground_truth_map(gt_df)
        self.assertEqual(gt_map["S1-MULTI"], {"S2-10", "S3-20", "S2-30"})

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-MULTI", "target_entity_id": "S2-10"},
            {"source1_entity_id": "S1-MULTI", "target_entity_id": "S3-20"},
            {"source1_entity_id": "S1-MULTI", "target_entity_id": "S2-30"},
            {"source1_entity_id": "S1-MULTI", "target_entity_id": "S2-99"},
        ])
        labeled = add_labels_to_candidate_pairs(pairs, gt_map)
        self.assertEqual(labeled["label"].tolist(), [1, 1, 1, 0])

    def test_4_correct_positive_negative_labels(self) -> None:
        """Verify binary labels accurately reflect multiple query entities simultaneously."""
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-A", "matched_entity_ids": "S2-1, S3-2"},
            {"source1_entity_id": "S1-B", "matched_entity_ids": "S2-9"},
        ])
        gt_map = build_ground_truth_map(gt_df)

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-A", "target_entity_id": "S2-1"},
            {"source1_entity_id": "S1-A", "target_entity_id": "S3-2"},
            {"source1_entity_id": "S1-A", "target_entity_id": "S3-99"},
            {"source1_entity_id": "S1-B", "target_entity_id": "S2-9"},
            {"source1_entity_id": "S1-B", "target_entity_id": "S2-1"},
        ])
        labeled = add_labels_to_candidate_pairs(pairs, gt_map)
        self.assertEqual(labeled["label"].tolist(), [1, 1, 0, 1, 0])

    def test_5_feature_columns_exclude_identifiers(self) -> None:
        """Verify predictive feature columns strictly exclude identifier and label columns."""
        model = EntityResolutionModel()
        for id_col in IDENTIFIER_COLUMNS + ["label", "matched_entity_ids"]:
            self.assertNotIn(id_col, model.feature_columns)
        self.assertEqual(len(model.feature_columns), len(FEATURE_COLUMNS))

    def test_6_group_aware_split(self) -> None:
        """Verify group-aware split places all candidate pairs of a Source 1 into either train or val."""
        df = _create_synthetic_training_data(n_entities=30, pairs_per_entity=4)
        train_df, val_df = split_candidates_by_group(df, val_ratio=0.3, random_state=42)

        train_s1 = set(train_df["source1_entity_id"])
        val_s1 = set(val_df["source1_entity_id"])

        # Strictly disjoint
        self.assertTrue(train_s1.isdisjoint(val_s1))
        # Complete coverage
        self.assertEqual(len(train_df) + len(val_df), len(df))
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)

    def test_7_model_training(self) -> None:
        """Verify model trains successfully on synthetic feature dataset."""
        train_df = _create_synthetic_training_data(n_entities=20, pairs_per_entity=4)
        model = EntityResolutionModel(config=ModelConfig(max_iter=20, random_state=42))
        model.fit(train_df)
        self.assertTrue(model.is_fitted)

    def test_8_probability_prediction(self) -> None:
        """Verify model predicts calibrated probabilities bounded in [0.0, 1.0]."""
        train_df = _create_synthetic_training_data(n_entities=25, pairs_per_entity=4)
        val_df = _create_synthetic_training_data(n_entities=10, pairs_per_entity=4)

        model = EntityResolutionModel(config=ModelConfig(max_iter=30, random_state=42))
        model.fit(train_df)

        probs = model.predict_proba(val_df)
        self.assertEqual(len(probs), len(val_df))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Check positive synthetic pairs receive higher probability than negatives
        pos_probs = probs[val_df["label"] == 1]
        neg_probs = probs[val_df["label"] == 0]
        self.assertGreater(float(np.mean(pos_probs)), float(np.mean(neg_probs)))

    def test_9_f0_5_calculation(self) -> None:
        """Verify F0.5 score matches exact challenge definition."""
        # Perfect precision and recall
        self.assertAlmostEqual(calculate_f0_5(1.0, 1.0), 1.0)
        # Zero cases
        self.assertEqual(calculate_f0_5(0.0, 1.0), 0.0)
        self.assertEqual(calculate_f0_5(1.0, 0.0), 0.0)
        self.assertEqual(calculate_f0_5(0.0, 0.0), 0.0)

        # Numerical check: P = 0.8, R = 0.5
        # F0.5 = (1.25 * 0.8 * 0.5) / (0.25 * 0.8 + 0.5) = 0.5 / 0.7 = 0.7142857...
        expected = (1.25 * 0.8 * 0.5) / (0.25 * 0.8 + 0.5)
        self.assertAlmostEqual(calculate_f0_5(0.8, 0.5), expected)

    def test_10_deterministic_reproducible_training(self) -> None:
        """Verify that training with the same random seed produces strictly identical probabilities."""
        df = _create_synthetic_training_data(n_entities=20, pairs_per_entity=4)
        m1 = EntityResolutionModel(config=ModelConfig(max_iter=25, random_state=42)).fit(df)
        m2 = EntityResolutionModel(config=ModelConfig(max_iter=25, random_state=42)).fit(df)

        p1 = m1.predict_proba(df)
        p2 = m2.predict_proba(df)
        assert_array_almost_equal(p1, p2, decimal=6)


if __name__ == "__main__":
    unittest.main()
