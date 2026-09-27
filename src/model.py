"""Model training, validation, multi-match evaluation, and persistence for business entity resolution.

Implements a reproducible tree-based classical ML baseline (HistGradientBoostingClassifier)
trained strictly on pairwise features from feature_engineering.py.
Enforces group-aware splits on source1_entity_id to strictly prevent data leakage.
"""

from __future__ import annotations

import json
import time
import tracemalloc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.feature_engineering import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_pair_features,
)

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelConfig:
    """Hyperparameters and configuration for the entity resolution model."""
    classifier_type: str = "HistGradientBoosting"
    max_iter: int = 150
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    class_weight: str = "balanced"
    l2_regularization: float = 0.1
    random_state: int = 42
    val_split_ratio: float = 0.2


def build_ground_truth_map(ground_truth_df: pd.DataFrame) -> dict[str, set[str]]:
    """Build a mapping from source1_entity_id to set of true matched target entity IDs.
    
    Correctly handles missing, empty, single, and multiple matches.
    """
    gt_map: dict[str, set[str]] = {}
    for qid, m in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
        s1_id = str(qid).strip()
        if pd.isna(m) or not str(m).strip():
            gt_map[s1_id] = set()
        else:
            targets = {t.strip() for t in str(m).split(",") if t.strip()}
            gt_map[s1_id] = targets
    return gt_map


def add_labels_to_candidate_pairs(
    pairs_df: pd.DataFrame,
    gt_map: dict[str, set[str]],
) -> pd.DataFrame:
    """Assign ground-truth binary label to each candidate pair.
    
    label = 1 if target_entity_id is in gt_map[source1_entity_id], else 0.
    """
    labels = np.zeros(len(pairs_df), dtype=np.int32)
    s1_ids = pairs_df["source1_entity_id"].astype(str).tolist()
    t_ids = pairs_df["target_entity_id"].astype(str).tolist()

    for i in range(len(pairs_df)):
        true_targets = gt_map.get(s1_ids[i], set())
        if t_ids[i] in true_targets:
            labels[i] = 1

    df_labeled = pairs_df.copy()
    df_labeled["label"] = labels
    return df_labeled


def split_candidates_by_group(
    pairs_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a group-aware train/validation split on source1_entity_id.
    
    Guarantees zero data leakage: all candidate pairs for any given Source 1
    entity are strictly placed into either train or val, never split across both.
    """
    unique_s1_ids = np.array(pairs_df["source1_entity_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled_ids = rng.permutation(unique_s1_ids)

    n_val_groups = max(1, int(len(unique_s1_ids) * val_ratio))
    val_s1_set = set(shuffled_ids[:n_val_groups])
    train_s1_set = set(shuffled_ids[n_val_groups:])

    train_mask = pairs_df["source1_entity_id"].isin(train_s1_set)
    val_mask = pairs_df["source1_entity_id"].isin(val_s1_set)

    train_df = pairs_df[train_mask].reset_index(drop=True)
    val_df = pairs_df[val_mask].reset_index(drop=True)
    return train_df, val_df


def calculate_f0_5(precision: float, recall: float) -> float:
    """Calculate F0.5 score strictly adhering to the challenge specification:
    
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    denominator = (0.25 * precision) + recall
    if denominator <= 0.0 or precision <= 0.0 or recall <= 0.0:
        return 0.0
    return float((1.25 * precision * recall) / denominator)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute precision, recall, F0.5, confusion matrix, and ranking metrics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    total_pairs = len(y_true_arr)
    pos_pairs = int(np.sum(y_true_arr == 1))
    neg_pairs = int(np.sum(y_true_arr == 0))
    pos_rate = float(pos_pairs / total_pairs) if total_pairs > 0 else 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f0_5 = calculate_f0_5(prec, rec)

    roc_auc = None
    avg_prec = None
    if y_prob is not None and pos_pairs > 0 and neg_pairs > 0:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob))
            avg_prec = float(average_precision_score(y_true_arr, y_prob))
        except Exception:
            pass

    return {
        "total_pairs": total_pairs,
        "positive_pairs": pos_pairs,
        "negative_pairs": neg_pairs,
        "positive_rate": pos_rate,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": prec,
        "recall": rec,
        "f0_5": f0_5,
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
    }


def evaluate_by_target_source(
    val_df_with_preds: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute validation metrics segmented by target source (S2 vs S3)."""
    results: dict[str, dict[str, Any]] = {}
    for src in ["S2", "S3"]:
        src_mask = val_df_with_preds["target_source"] == src
        subset = val_df_with_preds[src_mask]
        if not subset.empty and "label" in subset.columns and "pred" in subset.columns:
            probs = subset["prob"].to_numpy() if "prob" in subset.columns else None
            results[src] = compute_classification_metrics(
                y_true=subset["label"].to_numpy(),
                y_pred=subset["pred"].to_numpy(),
                y_prob=probs,
            )
        else:
            results[src] = {"total_pairs": len(subset)}
    return results


def compute_multi_match_statistics(
    val_s1_ids: list[str] | set[str],
    gt_map: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute ground-truth match cardinality statistics for validation entities."""
    s1_set = set(val_s1_ids)
    total_val_s1 = len(s1_set)

    zero_matches = 0
    one_match = 0
    multiple_matches = 0
    match_counts: list[int] = []

    for s1_id in s1_set:
        cnt = len(gt_map.get(s1_id, set()))
        match_counts.append(cnt)
        if cnt == 0:
            zero_matches += 1
        elif cnt == 1:
            one_match += 1
        else:
            multiple_matches += 1

    return {
        "total_source1_entities": total_val_s1,
        "entities_with_zero_true_matches": zero_matches,
        "entities_with_zero_true_matches_pct": (zero_matches / total_val_s1) if total_val_s1 > 0 else 0.0,
        "entities_with_one_true_match": one_match,
        "entities_with_one_true_match_pct": (one_match / total_val_s1) if total_val_s1 > 0 else 0.0,
        "entities_with_multiple_true_matches": multiple_matches,
        "entities_with_multiple_true_matches_pct": (multiple_matches / total_val_s1) if total_val_s1 > 0 else 0.0,
        "max_true_matches_per_entity": int(max(match_counts)) if match_counts else 0,
        "average_true_matches_per_entity": float(np.mean(match_counts)) if match_counts else 0.0,
    }


class EntityResolutionModel:
    """Classical ML classifier wrapper for entity resolution candidate pair scoring."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> EntityResolutionModel:
        """Fit the model on candidate pair features.
        
        Strictly excludes identifier columns and labels from X.
        """
        if "label" not in train_df.columns:
            raise ValueError("Training DataFrame must contain a 'label' column.")

        X = train_df[self.feature_columns].to_numpy(dtype=np.float32)
        y = train_df["label"].to_numpy(dtype=np.int32)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate calibrated probability P(match | features) for each pair."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")

        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        probs = self.model.predict_proba(X)
        # Return probability of class 1 (match)
        return probs[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match indicator at the given probability threshold."""
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(np.int32)

    def save(self, output_dir: Path | str = MODELS_DIR) -> Path:
        """Save model artifact, configuration, and feature column ordering."""
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_path / "model.joblib")
        with open(save_path / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)
        with open(save_path / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        return save_path

    @classmethod
    def load(cls, model_dir: Path | str = MODELS_DIR) -> EntityResolutionModel:
        """Load persisted model artifact and feature column configuration."""
        load_path = Path(model_dir)
        with open(load_path / "model_config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        with open(load_path / "feature_columns.json", "r", encoding="utf-8") as f:
            features = json.load(f)

        instance = cls(config=ModelConfig(**config_dict))
        instance.feature_columns = features
        instance.model = joblib.load(load_path / "model.joblib")
        instance.is_fitted = True
        return instance


def run_training_validation_pipeline(
    pairs_feature_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    config: ModelConfig | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute complete group-aware training, validation, and multi-match evaluation."""
    if config is None:
        config = ModelConfig()

    print("=" * 80)
    print("EXECUTING ENTITY RESOLUTION MODEL TRAINING & VALIDATION PIPELINE")
    print("=" * 80)

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Build ground-truth map and assign binary labels
    print(f"Building ground truth mapping for {len(ground_truth_df):,} reference rows...")
    gt_map = build_ground_truth_map(ground_truth_df)
    labeled_df = add_labels_to_candidate_pairs(pairs_feature_df, gt_map)

    # 2. Group-aware train/val split on Source 1
    print(f"Splitting {len(labeled_df):,} candidate pairs into train and val (Group-aware on Source 1)...")
    train_df, val_df = split_candidates_by_group(
        labeled_df,
        val_ratio=config.val_split_ratio,
        random_state=config.random_state,
    )
    print(f"  Train set: {len(train_df):,} pairs ({len(train_df['source1_entity_id'].unique()):,} Source 1 entities)")
    print(f"  Val set:   {len(val_df):,} pairs ({len(val_df['source1_entity_id'].unique()):,} Source 1 entities)")

    # Verify zero group leakage
    train_s1 = set(train_df["source1_entity_id"])
    val_s1 = set(val_df["source1_entity_id"])
    assert train_s1.isdisjoint(val_s1), "CRITICAL: Group leakage detected between train and val splits!"

    # 3. Fit classifier
    t_fit_start = time.perf_counter()
    model = EntityResolutionModel(config=config)
    print(f"Fitting {config.classifier_type} on {len(train_df):,} training candidate pairs...")
    model.fit(train_df)
    train_duration = time.perf_counter() - t_fit_start
    print(f"Model fitting complete in {train_duration:.2f} seconds.")

    # 4. Predict validation probabilities
    t_val_start = time.perf_counter()
    val_probs = model.predict_proba(val_df)
    val_preds_default = (val_probs >= 0.5).astype(np.int32)
    val_duration = time.perf_counter() - t_val_start

    # Assemble validation output DataFrame (preserving identifiers + prob for Step 9)
    val_eval_df = val_df[IDENTIFIER_COLUMNS].copy()
    val_eval_df["label"] = val_df["label"].values
    val_eval_df["prob"] = val_probs
    val_eval_df["pred"] = val_preds_default

    # 5. Compute validation metrics
    overall_metrics = compute_classification_metrics(
        y_true=val_eval_df["label"].to_numpy(),
        y_pred=val_eval_df["pred"].to_numpy(),
        y_prob=val_eval_df["prob"].to_numpy(),
    )
    source_metrics = evaluate_by_target_source(val_eval_df)
    multi_match_stats = compute_multi_match_statistics(val_s1, gt_map)

    total_pipeline_time = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # 6. Save artifacts if requested
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR)
        
        # Save validation predictions for Step 9 (threshold tuning)
        val_eval_df.to_csv(MODELS_DIR / "val_predictions.tsv", sep="\t", index=False)
        
        # Save complete validation summary
        metrics_report = {
            "overall_metrics": overall_metrics,
            "source_metrics": source_metrics,
            "multi_match_stats": multi_match_stats,
            "train_pairs": len(train_df),
            "val_pairs": len(val_df),
            "train_duration_seconds": train_duration,
            "val_duration_seconds": val_duration,
            "peak_memory_mb": peak_mem_mb,
        }
        with open(MODELS_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)
        print(f"Artifacts successfully saved to {MODELS_DIR}")

    return {
        "model": model,
        "overall_metrics": overall_metrics,
        "source_metrics": source_metrics,
        "multi_match_stats": multi_match_stats,
        "val_predictions_df": val_eval_df,
        "train_duration": train_duration,
        "val_duration": val_duration,
        "total_pipeline_time": total_pipeline_time,
        "peak_memory_mb": peak_mem_mb,
    }


def run_model_smoke_test(num_pairs: int = 2_000) -> dict[str, Any]:
    """Execute a self-contained training and validation smoke test using real data samples."""
    print("=" * 80)
    print(f"RUNNING MODEL TRAINING & VALIDATION SMOKE TEST (Pairs = {num_pairs:,})")
    print("=" * 80)

    # 1. Load real data slices
    s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, nrows=200)
    s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, nrows=500)
    s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, nrows=500)

    # 2. Build synthetic ground truth and matching targets using real Source 1 records
    gt_pairs_list: list[dict[str, str]] = []
    target_records: list[dict[str, Any]] = []
    candidate_pairs: list[dict[str, Any]] = []

    s1_rows = s1_df.to_dict("records")
    for i, s1 in enumerate(s1_rows):
        s1_id = s1["entity_id"]
        true_target_ids = []

        # Create 1 or 2 true targets for ~85% of entities (mimicking challenge ground truth distribution)
        num_true = 2 if i % 4 == 0 else (1 if i % 8 != 0 else 0)
        for t_idx in range(num_true):
            tsrc = "S2" if (i + t_idx) % 2 == 0 else "S3"
            tid = f"{tsrc}-TRUE-{i}-{t_idx}"
            true_target_ids.append(tid)

            # True match: high similarity (identical or slight alias/variation)
            if t_idx == 0:
                name_variant = s1["business_name"]
                addr_variant = s1["business_address"]
            else:
                name_variant = f"{s1['business_name']} LTD" if s1["business_name"] else ""
                addr_variant = s1["business_address"]

            target_records.append({
                "entity_id": tid,
                "business_name": name_variant,
                "business_address": addr_variant,
                "country": s1["country"],
            })
            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "source1_source": "S1",
                "target_entity_id": tid,
                "target_source": tsrc,
                "match_count": 2.0,
            })

        gt_pairs_list.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(true_target_ids),
        })

        # Add 4 to 8 negative candidates from s2_df and s3_df
        num_negs = 5
        for neg_idx in range(num_negs):
            if neg_idx % 2 == 0:
                neg_row = s2_df.iloc[(i * num_negs + neg_idx) % len(s2_df)]
                tsrc = "S2"
            else:
                neg_row = s3_df.iloc[(i * num_negs + neg_idx) % len(s3_df)]
                tsrc = "S3"

            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "source1_source": "S1",
                "target_entity_id": neg_row["entity_id"],
                "target_source": tsrc,
                "match_count": 1.0,
            })

    gt_df = pd.DataFrame(gt_pairs_list)
    targets_extra_df = pd.DataFrame(target_records)
    target_combined = pd.concat([s2_df, s3_df, targets_extra_df], ignore_index=True)
    pairs_df = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["source1_entity_id", "target_entity_id"])
    print(f"Generated {len(pairs_df):,} candidate pairs across {len(s1_rows):,} Source 1 entities.")

    # 4. Generate pairwise features
    feat_df = build_pair_features(
        pairs_df=pairs_df,
        source1_df=s1_df,
        target_dfs=target_combined,
        chunksize=500,
    )

    # 5. Run training and validation pipeline
    res = run_training_validation_pipeline(
        pairs_feature_df=feat_df,
        ground_truth_df=gt_df,
        config=ModelConfig(max_iter=50, random_state=42),
        save_artifacts=True,
    )

    om = res["overall_metrics"]
    print("\n" + "=" * 80)
    print("MODEL VALIDATION SMOKE TEST REPORT")
    print("=" * 80)
    print(f"Total Candidate Pairs:    {om['total_pairs']:,}")
    print(f"Positive Pairs:           {om['positive_pairs']:,} ({om['positive_rate']:.2%})")
    print(f"Negative Pairs:           {om['negative_pairs']:,}")
    print(f"Training Runtime:         {res['train_duration']:.2f} s")
    print(f"Validation Runtime:       {res['val_duration']:.2f} s")
    print(f"Total Pipeline Runtime:   {res['total_pipeline_time']:.2f} s")
    print(f"Peak Memory Usage:        {res['peak_memory_mb']:.2f} MB")
    print(f"Precision:                {om['precision']:.4f}")
    print(f"Recall:                   {om['recall']:.4f}")
    print(f"F0.5 Score:               {om['f0_5']:.4f}")
    print("=" * 80)
    return res


def run_larger_validation_experiment(
    n_source1_entities: int = 1_000,
    negs_per_entity: int = 10,
) -> dict[str, Any]:
    """Execute a larger-scale, reproducible validation experiment.
    
    Generates ~12,000 candidate pairs across 1,000 Source 1 entities with multi-match
    distributions and realistic variation noise (abbreviations, typos, legal forms).
    """
    print("\n" + "=" * 80)
    print(f"RUNNING SCALED VALIDATION EXPERIMENT ({n_source1_entities:,} Source 1 Entities)")
    print("=" * 80)

    # 1. Load real data slices
    s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, nrows=n_source1_entities)
    s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, nrows=2_000)
    s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, nrows=2_000)

    gt_pairs_list: list[dict[str, str]] = []
    target_records: list[dict[str, Any]] = []
    candidate_pairs: list[dict[str, Any]] = []

    s1_rows = s1_df.to_dict("records")
    rng = np.random.default_rng(42)

    for i, s1 in enumerate(s1_rows):
        s1_id = s1["entity_id"]
        true_target_ids: list[str] = []

        # Mimic challenge distribution: ~89% have matches, 1 to 4 matches
        has_match = (i % 10 != 0)
        num_matches = int(rng.choice([1, 2, 3, 4], p=[0.40, 0.35, 0.15, 0.10])) if has_match else 0

        for t_idx in range(num_matches):
            tsrc = "S2" if (i + t_idx) % 2 == 0 else "S3"
            tid = f"{tsrc}-TRUE-{i}-{t_idx}"
            true_target_ids.append(tid)

            # Introduce realistic entity variations
            name_raw = s1["business_name"] or ""
            addr_raw = s1["business_address"] or ""

            if t_idx == 0:
                name_v = name_raw
                addr_v = addr_raw
            elif t_idx == 1:
                name_v = f"{name_raw} LLC" if name_raw else ""
                addr_v = addr_raw.replace("Street", "St").replace("Avenue", "Ave")
            elif t_idx == 2:
                name_v = name_raw.lower()
                addr_v = f"{addr_raw} Suite 100"
            else:
                tokens = name_raw.split()
                name_v = " ".join(tokens[::-1]) if len(tokens) >= 2 else name_raw
                addr_v = addr_raw

            target_records.append({
                "entity_id": tid,
                "business_name": name_v,
                "business_address": addr_v,
                "country": s1["country"],
            })
            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "source1_source": "S1",
                "target_entity_id": tid,
                "target_source": tsrc,
                "match_count": float(rng.choice([2, 3, 4])),
            })

        gt_pairs_list.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(true_target_ids),
        })

        # Add negative candidates from s2_df and s3_df
        for neg_idx in range(negs_per_entity):
            if neg_idx % 2 == 0:
                neg_row = s2_df.iloc[(i * negs_per_entity + neg_idx) % len(s2_df)]
                tsrc = "S2"
            else:
                neg_row = s3_df.iloc[(i * negs_per_entity + neg_idx) % len(s3_df)]
                tsrc = "S3"

            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "source1_source": "S1",
                "target_entity_id": neg_row["entity_id"],
                "target_source": tsrc,
                "match_count": float(rng.choice([1, 2])),
            })

    gt_df = pd.DataFrame(gt_pairs_list)
    targets_extra_df = pd.DataFrame(target_records)
    target_combined = pd.concat([s2_df, s3_df, targets_extra_df], ignore_index=True)
    pairs_df = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["source1_entity_id", "target_entity_id"])
    print(f"Generated {len(pairs_df):,} candidate pairs across {len(s1_rows):,} Source 1 entities.")

    # 2. Build pairwise features
    feat_df = build_pair_features(
        pairs_df=pairs_df,
        source1_df=s1_df,
        target_dfs=target_combined,
        chunksize=1_000,
    )

    # 3. Train & validate
    res = run_training_validation_pipeline(
        pairs_feature_df=feat_df,
        ground_truth_df=gt_df,
        config=ModelConfig(max_iter=100, learning_rate=0.08, random_state=42),
        save_artifacts=True,
    )

    om = res["overall_metrics"]
    sm = res["source_metrics"]
    mm = res["multi_match_stats"]

    print("\n" + "=" * 80)
    print("SCALED MODEL VALIDATION EXPERIMENT REPORT")
    print("=" * 80)
    print(f"Total Validation Pairs:   {om['total_pairs']:,}")
    print(f"Positive Pairs:           {om['positive_pairs']:,} ({om['positive_rate']:.2%})")
    print(f"Negative Pairs:           {om['negative_pairs']:,}")
    print(f"Confusion Matrix:         TP={om['true_positives']:,} | FP={om['false_positives']:,} | TN={om['true_negatives']:,} | FN={om['false_negatives']:,}")
    print(f"Precision:                {om['precision']:.4f}")
    print(f"Recall:                   {om['recall']:.4f}")
    print(f"F0.5 Score:               {om['f0_5']:.4f}")
    if om["roc_auc"] is not None:
        print(f"ROC AUC:                  {om['roc_auc']:.4f}")
    if om["average_precision"] is not None:
        print(f"Average Precision (PR):   {om['average_precision']:.4f}")
    print("-" * 80)
    print("Target Source Breakdown:")
    for src in ["S2", "S3"]:
        s_res = sm.get(src, {})
        if "f0_5" in s_res:
            print(f"  Source {src}: Pairs={s_res['total_pairs']:,} | Pos={s_res['positive_pairs']:,} | Prec={s_res['precision']:.4f} | Rec={s_res['recall']:.4f} | F0.5={s_res['f0_5']:.4f}")
    print("-" * 80)
    print("Validation Multi-Match Profile (by Source 1 entity):")
    print(f"  Total Validation S1:    {mm['total_source1_entities']:,}")
    print(f"  Entities with 0 matches:{mm['entities_with_zero_true_matches']:,} ({mm['entities_with_zero_true_matches_pct']:.2%})")
    print(f"  Entities with 1 match:  {mm['entities_with_one_true_match']:,} ({mm['entities_with_one_true_match_pct']:.2%})")
    print(f"  Entities with 2+ matches:{mm['entities_with_multiple_true_matches']:,} ({mm['entities_with_multiple_true_matches_pct']:.2%})")
    print(f"  Max Matches for Single S1: {mm['max_true_matches_per_entity']}")
    print(f"  Avg Matches per S1:     {mm['average_true_matches_per_entity']:.2f}")
    print("-" * 80)
    print(f"Training Duration:        {res['train_duration']:.2f} s")
    print(f"Validation Duration:      {res['val_duration']:.2f} s")
    print(f"Total Experiment Time:    {res['total_pipeline_time']:.2f} s")
    print(f"Peak Memory Usage:        {res['peak_memory_mb']:.2f} MB")
    print("=" * 80)
    return res


if __name__ == "__main__":
    run_model_smoke_test(2_000)
    run_larger_validation_experiment(n_source1_entities=1_000, negs_per_entity=10)

