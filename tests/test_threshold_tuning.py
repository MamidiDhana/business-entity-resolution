"""Unit tests for Step 9: Probability Threshold Optimization and Calibration."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.threshold_tuning import (
    calculate_f0_5,
    evaluate_multi_match_distribution,
    evaluate_threshold,
    select_optimal_threshold,
    sweep_thresholds,
)


class TestThresholdTuning(unittest.TestCase):
    """Test suite verifying threshold evaluation, F0.5 optimization, and multi-match calibration."""

    def test_1_f0_5_formula(self) -> None:
        """Verify strict adherence to the challenge F0.5 definition: (1.25 * P * R) / (0.25 * P + R)."""
        # Perfect precision and recall
        self.assertAlmostEqual(calculate_f0_5(1.0, 1.0), 1.0)

        # Zero precision or recall
        self.assertEqual(calculate_f0_5(0.0, 1.0), 0.0)
        self.assertEqual(calculate_f0_5(1.0, 0.0), 0.0)
        self.assertEqual(calculate_f0_5(0.0, 0.0), 0.0)

        # Numerical check: P = 0.8, R = 0.6
        # num = 1.25 * 0.8 * 0.6 = 0.6
        # den = 0.25 * 0.8 + 0.6 = 0.2 + 0.6 = 0.8
        # F0.5 = 0.6 / 0.8 = 0.75
        self.assertAlmostEqual(calculate_f0_5(0.8, 0.6), 0.75)

    def test_2_threshold_conversion(self) -> None:
        """Verify binary prediction conversion: prob >= threshold -> 1 else 0."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1"] * 5,
            "source1_source": ["S1"] * 5,
            "target_entity_id": [f"S2-{i}" for i in range(5)],
            "target_source": ["S2"] * 5,
            "label": [0, 0, 1, 1, 1],
            "prob": [0.10, 0.49, 0.50, 0.51, 0.90],
        })
        res = evaluate_threshold(df, threshold=0.50)
        self.assertEqual(res["predicted_positives"], 3)
        self.assertEqual(res["true_positives"], 3)
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["true_negatives"], 2)
        self.assertEqual(res["false_negatives"], 0)

    def test_3_precision_calculation(self) -> None:
        """Verify precision computation including empty prediction handling."""
        # TP = 4, FP = 1 -> Precision = 4 / 5 = 0.8
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1"] * 5,
            "source1_source": ["S1"] * 5,
            "target_entity_id": [f"S2-{i}" for i in range(5)],
            "target_source": ["S2"] * 5,
            "label": [1, 1, 1, 1, 0],
            "prob": [0.9, 0.8, 0.7, 0.6, 0.55],
        })
        res = evaluate_threshold(df, threshold=0.50)
        self.assertAlmostEqual(res["precision"], 0.8)

        # No positive predictions -> Precision = 0.0 (no division by zero)
        res_high = evaluate_threshold(df, threshold=0.99)
        self.assertEqual(res_high["precision"], 0.0)

    def test_4_recall_calculation(self) -> None:
        """Verify recall computation across different detection rates."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1"] * 4,
            "source1_source": ["S1"] * 4,
            "target_entity_id": [f"S2-{i}" for i in range(4)],
            "target_source": ["S2"] * 4,
            "label": [1, 1, 1, 1],
            "prob": [0.9, 0.7, 0.3, 0.2],
        })
        # At threshold 0.5: 2 detected out of 4 true matches -> Recall = 0.5
        res = evaluate_threshold(df, threshold=0.50)
        self.assertAlmostEqual(res["recall"], 0.5)

        # At threshold 0.1: all 4 detected -> Recall = 1.0
        res_low = evaluate_threshold(df, threshold=0.10)
        self.assertAlmostEqual(res_low["recall"], 1.0)

    def test_5_threshold_sweep(self) -> None:
        """Verify coarse and fine threshold sweeps return a structured, non-empty DataFrame."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-A", "S1-A", "S1-B", "S1-B"],
            "source1_source": ["S1"] * 4,
            "target_entity_id": ["S2-1", "S3-1", "S2-2", "S3-2"],
            "target_source": ["S2", "S3", "S2", "S3"],
            "label": [1, 0, 1, 0],
            "prob": [0.92, 0.45, 0.88, 0.15],
        })
        sweep_df = sweep_thresholds(df, coarse_grid=[0.2, 0.5, 0.8])
        self.assertFalse(sweep_df.empty)
        self.assertIn("threshold", sweep_df.columns)
        self.assertIn("f0_5", sweep_df.columns)
        self.assertIn("precision", sweep_df.columns)
        self.assertIn("recall", sweep_df.columns)
        # Thresholds should be sorted in ascending order
        self.assertTrue(sweep_df["threshold"].is_monotonic_increasing)

    def test_6_best_threshold_selection(self) -> None:
        """Verify optimal threshold selection and precision-conservative tie-breaking."""
        # Synthetic results where threshold 0.70 has highest F0.5
        results_df = pd.DataFrame([
            {"threshold": 0.40, "precision": 0.70, "recall": 0.90, "f0_5": 0.73},
            {"threshold": 0.60, "precision": 0.85, "recall": 0.85, "f0_5": 0.85},
            {"threshold": 0.70, "precision": 0.92, "recall": 0.80, "f0_5": 0.89},
            {"threshold": 0.80, "precision": 0.95, "recall": 0.60, "f0_5": 0.84},
        ])
        best = select_optimal_threshold(results_df)
        self.assertEqual(best["threshold"], 0.70)

        # Tie-breaker test: identical F0.5 -> should pick higher precision / threshold
        tied_df = pd.DataFrame([
            {"threshold": 0.50, "precision": 0.80, "recall": 0.80, "f0_5": 0.80},
            {"threshold": 0.60, "precision": 0.85, "recall": 0.75, "f0_5": 0.80},
        ])
        best_tied = select_optimal_threshold(tied_df)
        self.assertEqual(best_tied["threshold"], 0.60)
        self.assertEqual(best_tied["precision"], 0.85)

    def test_7_probability_boundary_handling(self) -> None:
        """Verify proper handling at extreme probability thresholds (0.0 and 1.0)."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "source1_source": ["S1", "S1"],
            "target_entity_id": ["S2-1", "S3-1"],
            "target_source": ["S2", "S3"],
            "label": [1, 0],
            "prob": [1.0, 0.0],
        })
        # At threshold 0.0: all predicted as positive
        res_0 = evaluate_threshold(df, threshold=0.0)
        self.assertEqual(res_0["predicted_positives"], 2)

        # At threshold 1.0: only exact 1.0 predicted as positive
        res_1 = evaluate_threshold(df, threshold=1.0)
        self.assertEqual(res_1["predicted_positives"], 1)
        self.assertEqual(res_1["precision"], 1.0)

    def test_8_zero_predicted_matches(self) -> None:
        """Verify behavior when no candidate exceeds the threshold."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "source1_source": ["S1", "S1"],
            "target_entity_id": ["S2-1", "S3-1"],
            "target_source": ["S2", "S3"],
            "label": [1, 0],
            "prob": [0.40, 0.10],
        })
        res = evaluate_threshold(df, threshold=0.80)
        self.assertEqual(res["predicted_positives"], 0)
        self.assertEqual(res["true_positives"], 0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["f0_5"], 0.0)

        mm = evaluate_multi_match_distribution(df, threshold=0.80)
        self.assertEqual(mm["predicted_distribution"]["zero_matches_count"], 2)
        self.assertEqual(mm["predicted_distribution"]["one_match_count"], 0)

    def test_9_multiple_predicted_matches_per_source1(self) -> None:
        """Verify multi-match tracking when a Source 1 entity has multiple predicted targets."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-MULTI", "S1-MULTI", "S1-MULTI", "S1-SINGLE"],
            "source1_source": ["S1"] * 4,
            "target_entity_id": ["S2-A", "S3-B", "S2-C", "S2-D"],
            "target_source": ["S2", "S3", "S2", "S2"],
            "label": [1, 1, 1, 1],
            "prob": [0.95, 0.90, 0.85, 0.92],
        })
        mm = evaluate_multi_match_distribution(df, threshold=0.80)
        p_dist = mm["predicted_distribution"]
        self.assertEqual(mm["total_source1_entities"], 2)
        self.assertEqual(p_dist["multi_matches_count"], 1)  # S1-MULTI has 3 matches
        self.assertEqual(p_dist["one_match_count"], 1)    # S1-SINGLE has 1 match
        self.assertEqual(p_dist["max_matches_per_s1"], 3)

    def test_10_deterministic_results(self) -> None:
        """Verify that threshold evaluation and selection are strictly deterministic."""
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "source1_source": ["S1", "S1", "S1"],
            "target_entity_id": ["S2-1", "S3-2", "S2-3"],
            "target_source": ["S2", "S3", "S2"],
            "label": [1, 0, 1],
            "prob": [0.85, 0.20, 0.75],
        })
        sweep1 = sweep_thresholds(df, coarse_grid=[0.3, 0.5, 0.7])
        sweep2 = sweep_thresholds(df, coarse_grid=[0.3, 0.5, 0.7])
        assert_frame_equal(sweep1, sweep2)


if __name__ == "__main__":
    unittest.main()
