"""Threshold optimization and multi-match calibration for entity resolution.

Evaluates probability thresholds to maximize the challenge F0.5 objective:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Performs coarse and fine sweeps, source breakdown (S2 vs S3), multi-match profiling,
and precision-conservative tie-breaking.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
REQUIRED_PRED_COLUMNS = {
    "source1_entity_id",
    "source1_source",
    "target_entity_id",
    "target_source",
    "label",
    "prob",
}


def calculate_f0_5(precision: float, recall: float) -> float:
    """Calculate the F0.5 score strictly adhering to the challenge specification:
    
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    denominator = (0.25 * precision) + recall
    if denominator <= 0.0 or precision <= 0.0 or recall <= 0.0:
        return 0.0
    return float((1.25 * precision * recall) / denominator)


def load_and_validate_predictions(file_path: Path | str) -> pd.DataFrame:
    """Load validation predictions TSV and strictly validate data integrity and schemas."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Validation predictions file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # Validate required columns
    missing = REQUIRED_PRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Validation predictions missing required columns: {missing}")

    # Validate binary labels
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid label values found: {unique_labels}. Expected subset of {{0, 1}}.")

    # Validate probabilities in [0.0, 1.0]
    if df["prob"].isna().any():
        raise ValueError("Null/NaN values detected in prediction probabilities.")

    probs = df["prob"].to_numpy()
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(f"Probabilities out of bounds [0.0, 1.0]: min={probs.min()}, max={probs.max()}")

    return df


def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Compute classification and segmentation metrics for a specific probability threshold."""
    labels = df["label"].to_numpy(dtype=int)
    probs = df["prob"].to_numpy(dtype=float)
    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fn = int(np.sum((labels == 1) & (preds == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f0_5 = calculate_f0_5(prec, rec)

    predicted_pos = tp + fp
    predicted_rate = float(predicted_pos / len(df)) if len(df) > 0 else 0.0

    # Source 2 segmented metrics
    s2_mask = (df["target_source"] == "S2").to_numpy()
    if np.any(s2_mask):
        s2_tp = int(np.sum((labels[s2_mask] == 1) & (preds[s2_mask] == 1)))
        s2_fp = int(np.sum((labels[s2_mask] == 0) & (preds[s2_mask] == 1)))
        s2_fn = int(np.sum((labels[s2_mask] == 1) & (preds[s2_mask] == 0)))
        s2_prec = float(s2_tp / (s2_tp + s2_fp)) if (s2_tp + s2_fp) > 0 else 0.0
        s2_rec = float(s2_tp / (s2_tp + s2_fn)) if (s2_tp + s2_fn) > 0 else 0.0
        s2_f0_5 = calculate_f0_5(s2_prec, s2_rec)
    else:
        s2_prec, s2_rec, s2_f0_5 = 0.0, 0.0, 0.0

    # Source 3 segmented metrics
    s3_mask = (df["target_source"] == "S3").to_numpy()
    if np.any(s3_mask):
        s3_tp = int(np.sum((labels[s3_mask] == 1) & (preds[s3_mask] == 1)))
        s3_fp = int(np.sum((labels[s3_mask] == 0) & (preds[s3_mask] == 1)))
        s3_fn = int(np.sum((labels[s3_mask] == 1) & (preds[s3_mask] == 0)))
        s3_prec = float(s3_tp / (s3_tp + s3_fp)) if (s3_tp + s3_fp) > 0 else 0.0
        s3_rec = float(s3_tp / (s3_tp + s3_fn)) if (s3_tp + s3_fn) > 0 else 0.0
        s3_f0_5 = calculate_f0_5(s3_prec, s3_rec)
    else:
        s3_prec, s3_rec, s3_f0_5 = 0.0, 0.0, 0.0

    return {
        "threshold": round(threshold, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": prec,
        "recall": rec,
        "f0_5": f0_5,
        "predicted_positives": predicted_pos,
        "predicted_positive_rate": predicted_rate,
        "s2_precision": s2_prec,
        "s2_recall": s2_rec,
        "s2_f0_5": s2_f0_5,
        "s3_precision": s3_prec,
        "s3_recall": s3_rec,
        "s3_f0_5": s3_f0_5,
    }


def sweep_thresholds(
    df: pd.DataFrame,
    coarse_grid: list[float] | None = None,
    fine_step: float = 0.01,
) -> pd.DataFrame:
    """Perform a coarse threshold sweep followed by fine grid search around the peak F0.5 region."""
    if coarse_grid is None:
        coarse_grid = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]

    evaluated: dict[float, dict[str, Any]] = {}

    # 1. Coarse evaluation
    for th in coarse_grid:
        evaluated[round(th, 4)] = evaluate_threshold(df, th)

    # 2. Identify peak coarse threshold
    best_coarse_th = max(evaluated.keys(), key=lambda t: (evaluated[t]["f0_5"], evaluated[t]["precision"], t))

    # 3. Fine evaluation around peak
    fine_start = max(0.01, best_coarse_th - 0.08)
    fine_end = min(0.99, best_coarse_th + 0.08)
    fine_grid = [round(x, 4) for x in np.arange(fine_start, fine_end + fine_step / 2, fine_step)]

    for th in fine_grid:
        if round(th, 4) not in evaluated:
            evaluated[round(th, 4)] = evaluate_threshold(df, th)

    rows = [evaluated[k] for k in sorted(evaluated.keys())]
    results_df = pd.DataFrame(rows)
    return results_df


def select_optimal_threshold(results_df: pd.DataFrame) -> dict[str, Any]:
    """Select the optimal probability threshold maximizing validation F0.5.
    
    Tie-breaking strategy:
      If multiple thresholds achieve within 1e-4 of the maximum F0.5, select the
      threshold with higher precision (or higher threshold value).
      Rationale: For entity resolution, false positives create invalid entity linkages
      in downstream databases, and the F0.5 challenge metric weights precision twice as
      heavily as recall.
    """
    max_f0_5 = results_df["f0_5"].max()
    tolerance = 1e-4

    # Filter candidates achieving near-maximal F0.5
    top_candidates = results_df[results_df["f0_5"] >= (max_f0_5 - tolerance)].copy()

    # Sort primarily by F0.5 (descending), secondarily by Precision (descending),
    # and tertiarily by Threshold (descending for conservative thresholding)
    sorted_candidates = top_candidates.sort_values(
        by=["f0_5", "precision", "threshold"],
        ascending=[False, False, False],
    )
    best_row = sorted_candidates.iloc[0].to_dict()
    for int_key in ["true_positives", "false_positives", "true_negatives", "false_negatives", "predicted_positives"]:
        if int_key in best_row:
            best_row[int_key] = int(best_row[int_key])
    return best_row


def evaluate_multi_match_distribution(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Analyze predicted match cardinality per Source 1 entity at the selected threshold."""
    all_s1_entities = set(df["source1_entity_id"].unique())
    total_s1 = len(all_s1_entities)

    # Predicted matches
    pred_mask = df["prob"] >= threshold
    matched_df = df[pred_mask]
    pred_counts_series = matched_df.groupby("source1_entity_id").size()

    # Ground-truth true matches (for comparison)
    gt_mask = df["label"] == 1
    gt_df = df[gt_mask]
    gt_counts_series = gt_df.groupby("source1_entity_id").size()

    pred_counts = [int(pred_counts_series.get(s1, 0)) for s1 in all_s1_entities]
    gt_counts = [int(gt_counts_series.get(s1, 0)) for s1 in all_s1_entities]

    zero_preds = sum(1 for c in pred_counts if c == 0)
    one_pred = sum(1 for c in pred_counts if c == 1)
    multi_preds = sum(1 for c in pred_counts if c >= 2)

    zero_gt = sum(1 for c in gt_counts if c == 0)
    one_gt = sum(1 for c in gt_counts if c == 1)
    multi_gt = sum(1 for c in gt_counts if c >= 2)

    return {
        "total_source1_entities": total_s1,
        "predicted_distribution": {
            "zero_matches_count": zero_preds,
            "zero_matches_pct": (zero_preds / total_s1) if total_s1 > 0 else 0.0,
            "one_match_count": one_pred,
            "one_match_pct": (one_pred / total_s1) if total_s1 > 0 else 0.0,
            "multi_matches_count": multi_preds,
            "multi_matches_pct": (multi_preds / total_s1) if total_s1 > 0 else 0.0,
            "average_matches_per_s1": float(np.mean(pred_counts)) if pred_counts else 0.0,
            "max_matches_per_s1": int(max(pred_counts)) if pred_counts else 0,
        },
        "ground_truth_distribution": {
            "zero_matches_count": zero_gt,
            "zero_matches_pct": (zero_gt / total_s1) if total_s1 > 0 else 0.0,
            "one_match_count": one_gt,
            "one_match_pct": (one_gt / total_s1) if total_s1 > 0 else 0.0,
            "multi_matches_count": multi_gt,
            "multi_matches_pct": (multi_gt / total_s1) if total_s1 > 0 else 0.0,
            "average_matches_per_s1": float(np.mean(gt_counts)) if gt_counts else 0.0,
            "max_matches_per_s1": int(max(gt_counts)) if gt_counts else 0,
        },
    }


def run_threshold_optimization_pipeline(
    predictions_path: Path | str = MODELS_DIR / "val_predictions.tsv",
    output_dir: Path | str = MODELS_DIR,
) -> dict[str, Any]:
    """Execute complete Step 9 threshold optimization workflow."""
    print("=" * 80)
    print("EXECUTING STEP 9: PROBABILITY THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # 1. Load and validate predictions
    print(f"Loading validation predictions from {predictions_path}...")
    val_df = load_and_validate_predictions(predictions_path)
    print(f"Loaded {len(val_df):,} candidate predictions across {len(val_df['source1_entity_id'].unique()):,} Source 1 entities.")
    print(f"Ground truth positives in validation: {int(np.sum(val_df['label'] == 1)):,} ({np.mean(val_df['label']):.2%})")

    # 2. Perform coarse and fine threshold sweeps
    print("\nRunning coarse and fine threshold sweeps...")
    results_df = sweep_thresholds(val_df)
    print(f"Evaluated {len(results_df)} distinct threshold candidates.")

    # 3. Select optimal threshold
    best_config = select_optimal_threshold(results_df)
    best_th = float(best_config["threshold"])
    print(f"\nOptimal threshold selected: {best_th:.4f} (F0.5 = {best_config['f0_5']:.4f}, Precision = {best_config['precision']:.4f}, Recall = {best_config['recall']:.4f})")

    # 4. Multi-match profiling at selected threshold
    mm_profile = evaluate_multi_match_distribution(val_df, best_th)

    # 5. Extract nearby thresholds table for stability inspection
    th_idx = results_df[results_df["threshold"] == best_th].index[0]
    start_idx = max(0, th_idx - 3)
    end_idx = min(len(results_df), th_idx + 4)
    nearby_df = results_df.iloc[start_idx:end_idx][[
        "threshold", "precision", "recall", "f0_5",
        "predicted_positives", "s2_f0_5", "s3_f0_5",
    ]]

    # 6. Save artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_tsv_path = out_path / "threshold_results.tsv"
    results_df.to_csv(results_tsv_path, sep="\t", index=False)
    print(f"Saved complete threshold evaluation table to {results_tsv_path}")

    config_json_path = out_path / "threshold_config.json"
    full_config = {
        "selected_threshold": best_th,
        "validation_precision": best_config["precision"],
        "validation_recall": best_config["recall"],
        "validation_f0_5": best_config["f0_5"],
        "predicted_positives": int(best_config["predicted_positives"]),
        "predicted_positive_rate": float(best_config["predicted_positive_rate"]),
        "confusion_matrix": {
            "tp": int(best_config["true_positives"]),
            "fp": int(best_config["false_positives"]),
            "tn": int(best_config["true_negatives"]),
            "fn": int(best_config["false_negatives"]),
        },
        "source_breakdown": {
            "S2": {
                "precision": best_config["s2_precision"],
                "recall": best_config["s2_recall"],
                "f0_5": best_config["s2_f0_5"],
            },
            "S3": {
                "precision": best_config["s3_precision"],
                "recall": best_config["s3_recall"],
                "f0_5": best_config["s3_f0_5"],
            },
        },
        "multi_match_profile": mm_profile,
        "selection_criterion": "Maximum validation F0.5 with precision-conservative tie-breaking",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)
    print(f"Saved optimal threshold configuration to {config_json_path}")

    # 7. Print summary report
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Selected Threshold:         {best_th:.4f}")
    print(f"Validation Precision:       {best_config['precision']:.4f}")
    print(f"Validation Recall:          {best_config['recall']:.4f}")
    print(f"Validation F0.5 Score:      {best_config['f0_5']:.4f}")
    print(f"Predicted Positives:        {best_config['predicted_positives']:,} ({best_config['predicted_positive_rate']:.2%})")
    print(f"Confusion Matrix:           TP={best_config['true_positives']} | FP={best_config['false_positives']} | TN={best_config['true_negatives']} | FN={best_config['false_negatives']}")
    print("-" * 80)
    print("Nearby Threshold Stability Comparison:")
    print(nearby_df.to_string(index=False))
    print("-" * 80)
    print("Source-Specific Performance at Selected Threshold:")
    print(f"  Source S2: Precision={best_config['s2_precision']:.4f} | Recall={best_config['s2_recall']:.4f} | F0.5={best_config['s2_f0_5']:.4f}")
    print(f"  Source S3: Precision={best_config['s3_precision']:.4f} | Recall={best_config['s3_recall']:.4f} | F0.5={best_config['s3_f0_5']:.4f}")
    print("-" * 80)
    p_dist = mm_profile["predicted_distribution"]
    gt_dist = mm_profile["ground_truth_distribution"]
    print("Multi-Match Cardinality Comparison (Validation Set):")
    print(f"  0 Matches:   Predicted = {p_dist['zero_matches_count']} ({p_dist['zero_matches_pct']:.1%})  |  True = {gt_dist['zero_matches_count']} ({gt_dist['zero_matches_pct']:.1%})")
    print(f"  1 Match:     Predicted = {p_dist['one_match_count']} ({p_dist['one_match_pct']:.1%})  |  True = {gt_dist['one_match_count']} ({gt_dist['one_match_pct']:.1%})")
    print(f"  2+ Matches:  Predicted = {p_dist['multi_matches_count']} ({p_dist['multi_matches_pct']:.1%})  |  True = {gt_dist['multi_matches_count']} ({gt_dist['multi_matches_pct']:.1%})")
    print(f"  Avg Matches: Predicted = {p_dist['average_matches_per_s1']:.2f}  |  True = {gt_dist['average_matches_per_s1']:.2f}")
    print(f"  Max Matches: Predicted = {p_dist['max_matches_per_s1']}  |  True = {gt_dist['max_matches_per_s1']}")
    print("=" * 80)

    return {
        "best_threshold": best_th,
        "best_config": best_config,
        "results_df": results_df,
        "nearby_df": nearby_df,
        "multi_match_profile": mm_profile,
    }


if __name__ == "__main__":
    run_threshold_optimization_pipeline()
