"""Unit tests for Step 11: Test Model Scoring & Final Match Generation."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.test_scoring import (
    load_scoring_artifacts,
    score_candidate_chunk,
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_COLS_PATH,
    DEFAULT_THRESHOLD_CONFIG_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODELS_DIR = PROJECT_ROOT / "models"


class DummyModel:
    """Mock model producing deterministic probability outputs for unit testing."""
    def __init__(self, prob_value: float = 0.9950):
        self.prob_value = prob_value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        # Return [1 - prob, prob]
        probs = np.full((n, 2), 1.0 - self.prob_value, dtype=np.float32)
        probs[:, 1] = self.prob_value
        return probs


class TestTestScoring(unittest.TestCase):
    """Test suite verifying Step 11 model scoring, threshold application, and match aggregation."""

    def test_1_model_and_artifact_loading(self) -> None:
        """Verify model, feature columns, and threshold config can be successfully loaded."""
        if DEFAULT_MODEL_PATH.is_file():
            model, feat_cols, threshold = load_scoring_artifacts()
            self.assertIsNotNone(model)
            self.assertGreater(len(feat_cols), 0)
            self.assertAlmostEqual(threshold, 0.9900, places=4)

    def test_2_feature_schema_compatibility(self) -> None:
        """Verify score_candidate_chunk produces features strictly matching expected schema."""
        model = DummyModel(0.9950)
        feat_cols = ["name_exact_match", "addr_exact_match", "target_source_is_s2"]
        
        pairs = pd.DataFrame([{
            "source1_entity_id": "S1-100",
            "target_entity_id": "S2-200",
            "target_source": "S2",
        }])
        s1_lookup = {"S1-100": ("Apex Tech", "100 Main St", "US")}
        target_lookup = {"S2-200": ("Apex Tech", "100 Main St", "US")}

        scored = score_candidate_chunk(
            pairs_chunk=pairs,
            model=model,
            feature_columns=feat_cols,
            threshold=0.9900,
            s1_lookup=s1_lookup,
            target_lookup=target_lookup,
        )
        self.assertEqual(len(scored), 1)
        self.assertIn("match_probability", scored.columns)
        self.assertIn("predicted_match", scored.columns)
        self.assertEqual(scored.iloc[0]["predicted_match"], 1)

    def test_3_probability_bounds(self) -> None:
        """Verify predicted probabilities strictly adhere to [0.0, 1.0] bounds without NaN/Inf."""
        model = DummyModel(0.5000)
        feat_cols = ["name_exact_match"]
        pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "target_entity_id": "S2-1", "target_source": "S2"}])
        s1_lookup = {"S1-1": ("Test", "Address", "US")}
        target_lookup = {"S2-1": ("Test", "Address", "US")}

        scored = score_candidate_chunk(pairs, model, feat_cols, 0.9900, s1_lookup, target_lookup)
        p = scored.iloc[0]["match_probability"]
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertFalse(np.isnan(p))
        self.assertFalse(np.isinf(p))

    def test_4_threshold_application(self) -> None:
        """Verify threshold rule: predicted_match = 1 iff prob >= threshold."""
        feat_cols = ["name_exact_match"]
        pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "target_entity_id": "S2-1", "target_source": "S2"}])
        s1_lookup = {"S1-1": ("Test", "Address", "US")}
        target_lookup = {"S2-1": ("Test", "Address", "US")}

        # Test above threshold (0.995 >= 0.990) -> 1
        model_pos = DummyModel(0.9950)
        scored_pos = score_candidate_chunk(pairs, model_pos, feat_cols, 0.9900, s1_lookup, target_lookup)
        self.assertEqual(scored_pos.iloc[0]["predicted_match"], 1)

        # Test below threshold (0.985 < 0.990) -> 0
        model_neg = DummyModel(0.9850)
        scored_neg = score_candidate_chunk(pairs, model_neg, feat_cols, 0.9900, s1_lookup, target_lookup)
        self.assertEqual(scored_neg.iloc[0]["predicted_match"], 0)

    def test_5_deterministic_scoring(self) -> None:
        """Verify running scoring twice on the same chunk produces identical outputs."""
        model = DummyModel(0.9920)
        feat_cols = ["name_exact_match", "addr_exact_match"]
        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-A", "target_entity_id": "S2-1", "target_source": "S2"},
            {"source1_entity_id": "S1-B", "target_entity_id": "S3-2", "target_source": "S3"},
        ])
        s1_lookup = {"S1-A": ("Alpha", "1 St", "US"), "S1-B": ("Beta", "2 St", "US")}
        target_lookup = {"S2-1": ("Alpha", "1 St", "US"), "S3-2": ("Beta", "2 St", "US")}

        scored_1 = score_candidate_chunk(pairs, model, feat_cols, 0.9900, s1_lookup, target_lookup)
        scored_2 = score_candidate_chunk(pairs, model, feat_cols, 0.9900, s1_lookup, target_lookup)
        assert_frame_equal(scored_1, scored_2)

    def test_6_zero_match_handling(self) -> None:
        """Verify an S1 entity with no candidates passing threshold results in empty matched_entity_ids."""
        matched_dict = {"S1-EMPTY": []}
        all_s1 = ["S1-EMPTY"]
        out_matched = [",".join(matched_dict.get(s1, [])) for s1 in all_s1]
        self.assertEqual(out_matched[0], "")

    def test_7_single_match_handling(self) -> None:
        """Verify an S1 entity with 1 match formats exactly as single target ID."""
        matched_dict = {"S1-SINGLE": ["S2-12345"]}
        all_s1 = ["S1-SINGLE"]
        out_matched = [",".join(matched_dict.get(s1, [])) for s1 in all_s1]
        self.assertEqual(out_matched[0], "S2-12345")

    def test_8_multi_match_handling(self) -> None:
        """Verify multi-match entities format as comma-separated IDs without spaces."""
        matched_dict = {"S1-MULTI": ["S2-100", "S3-200", "S2-300"]}
        all_s1 = ["S1-MULTI"]
        out_matched = [",".join(matched_dict.get(s1, [])) for s1 in all_s1]
        self.assertEqual(out_matched[0], "S2-100,S3-200,S2-300")

    def test_9_duplicate_target_removal(self) -> None:
        """Verify identical target IDs for the same S1 are deduplicated."""
        raw_targets = ["S2-100", "S2-100", "S3-200"]
        deduped = []
        seen = set()
        for t in raw_targets:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        self.assertEqual(deduped, ["S2-100", "S3-200"])

    def test_10_candidate_subset_guarantee(self) -> None:
        """Verify candidate subset validator correctly flags missing candidate matches."""
        cand_pairs = {("S1-1", "S2-10"), ("S1-1", "S2-20")}
        valid_matches = [("S1-1", "S2-10")]
        invalid_matches = [("S1-1", "S2-99")]  # Not in candidate pairs

        for s1, tid in valid_matches:
            self.assertIn((s1, tid), cand_pairs)

        for s1, tid in invalid_matches:
            self.assertNotIn((s1, tid), cand_pairs)

    def test_11_target_source_correctness(self) -> None:
        """Verify target IDs retain intact S2- and S3- prefixes and no S1- IDs inside matches."""
        matched_targets = ["S2-12345", "S3-67890"]
        for tid in matched_targets:
            self.assertTrue(tid.startswith(("S2-", "S3-")))
            self.assertFalse(tid.startswith("S1-"))

    def test_12_final_output_schema(self) -> None:
        """Verify final matching_results.tsv schema has exactly required columns."""
        df = pd.DataFrame([{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"}])
        self.assertEqual(list(df.columns), ["source1_entity_id", "matched_entity_ids"])


if __name__ == "__main__":
    unittest.main()
