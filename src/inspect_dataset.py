"""Dataset inspection script for Business Entity Resolution challenge.

Loads and inspects all 7 TSV dataset files without making any modifications.
"""

from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
import sys
from typing import Any
import pandas as pd

# Ensure standard output can print Unicode characters on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

EXPECTED_FILES = [
    TRAIN_DIR / "train_source1.tsv",
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv",
    TRAIN_DIR / "train_ground_truth.tsv",
    TEST_DIR / "test_source1.tsv",
    TEST_DIR / "test_source2.tsv",
    TEST_DIR / "test_source3.tsv",
]


def inspect_ground_truth_details(df: pd.DataFrame, file_path: Path) -> dict[str, Any]:
    """Inspect the exact schema and characteristics of ground-truth data.
    
    Does not assume schema beforehand; dynamically checks columns and distributions.
    """
    print(f"\n{'=' * 80}")
    print(f"DETAILED GROUND TRUTH SCHEMA INSPECTION: {file_path.name}")
    print(f"{'=' * 80}")
    
    num_rows = len(df)
    columns = list(df.columns)
    print(f"Columns: {columns}")
    print(f"Number of rows: {num_rows}")
    
    print("\n--- First 10 Rows ---")
    print(df.head(10).to_string(index=False))
    
    print("\n--- Column Summaries & Unique Value Counts ---")
    summary: dict[str, Any] = {
        "num_rows": num_rows,
        "columns": columns,
        "column_stats": {},
    }
    
    for col in columns:
        null_count = int(df[col].isna().sum())
        blank_count = int((df[col].fillna("").astype(str).str.strip() == "").sum())
        nunique = int(df[col].nunique())
        print(f"\nColumn '{col}':")
        print(f"  - Non-null count: {num_rows - null_count}")
        print(f"  - Null count: {null_count}")
        print(f"  - Blank/empty count: {blank_count}")
        print(f"  - Unique values: {nunique}")
        
        sample_values = df[col].dropna().head(5).tolist()
        print(f"  - Sample values (first 5): {sample_values}")
        
        summary["column_stats"][col] = {
            "null_count": null_count,
            "blank_count": blank_count,
            "nunique": nunique,
            "sample_values": sample_values,
        }
    
    # Inspect matched target representation if detected (e.g. comma-delimited ID list)
    for col in columns:
        sample_series = df[col].fillna("").astype(str).str.strip()
        has_comma = sample_series.str.contains(",").any()
        is_id_column = any("id" in col.lower() for col in [col])
        
        if has_comma or (is_id_column and col != columns[0]):
            print(f"\n--- Distribution Analysis for '{col}' ---")
            
            # Count number of tokens per entry
            token_counts = sample_series.apply(
                lambda val: len([v for v in val.split(",") if v.strip()]) if val else 0
            )
            
            zero_matches = int((token_counts == 0).sum())
            one_match = int((token_counts == 1).sum())
            multi_matches = int((token_counts > 1).sum())
            max_matches = int(token_counts.max()) if len(token_counts) > 0 else 0
            
            print(f"  - Rows with 0 matches: {zero_matches} ({zero_matches / max(num_rows, 1):.2%})")
            print(f"  - Rows with 1 match:   {one_match} ({one_match / max(num_rows, 1):.2%})")
            print(f"  - Rows with >1 matches: {multi_matches} ({multi_matches / max(num_rows, 1):.2%})")
            print(f"  - Max matches in single row: {max_matches}")
            
            # Prefix analysis of target IDs
            all_tokens = [
                token.strip()
                for row_tokens in sample_series.str.split(",")
                for token in row_tokens
                if token.strip()
            ]
            prefix_counts = Counter(
                token.split("-")[0] if "-" in token else token[:2]
                for token in all_tokens
            )
            print(f"  - Total matched entity references: {len(all_tokens)}")
            print(f"  - Target ID prefix distribution: {dict(prefix_counts)}")
            
            summary["match_distribution"] = {
                "zero_matches": zero_matches,
                "one_match": one_match,
                "multi_matches": multi_matches,
                "max_matches": max_matches,
                "prefix_counts": dict(prefix_counts),
            }
            
    return summary


def inspect_file(file_path: Path) -> dict[str, Any] | None:
    """Inspect a single TSV dataset file.
    
    Returns a dictionary of inspection metrics, or None if the file cannot be read.
    """
    rel_path = file_path.relative_to(PROJECT_ROOT) if file_path.is_relative_to(PROJECT_ROOT) else file_path
    print(f"\n{'-' * 80}")
    print(f"FILE: {rel_path}")
    print(f"{'-' * 80}")
    
    if not file_path.exists():
        print(f"ERROR: File does not exist at {file_path}")
        return None
    
    if not file_path.is_file():
        print(f"ERROR: Path exists but is not a regular file: {file_path}")
        return None

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    try:
        df = pd.read_csv(file_path, sep="\t", dtype=str)
    except Exception as exc:
        print(f"ERROR: Failed to read {file_path} as TSV: {exc}")
        return None
    
    num_rows = len(df)
    num_cols = len(df.columns)
    column_names = list(df.columns)
    
    print(f"Number of rows:    {num_rows:,}")
    print(f"Number of columns: {num_cols}")
    print(f"Column names:      {column_names}")
    
    print("\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
        
    print("\nMissing Values & Blank Counts:")
    missing_summary: dict[str, dict[str, int]] = {}
    for col in column_names:
        null_count = int(df[col].isna().sum())
        blank_count = int((df[col].fillna("").astype(str).str.strip() == "").sum())
        missing_summary[col] = {"nulls": null_count, "blanks": blank_count}
        print(f"  {col}: nulls={null_count:,} ({null_count / max(num_rows, 1):.2%}), blanks={blank_count:,} ({blank_count / max(num_rows, 1):.2%})")
        
    duplicate_info: dict[str, Any] = {}
    if "entity_id" in df.columns:
        duplicate_count = int(df["entity_id"].duplicated().sum())
        unique_count = int(df["entity_id"].nunique())
        duplicate_info = {
            "duplicate_entity_ids": duplicate_count,
            "unique_entity_ids": unique_count,
        }
        print("\nEntity ID Statistics:")
        print(f"  Unique entity_id count:    {unique_count:,}")
        print(f"  Duplicate entity_id count: {duplicate_count:,}")
    
    print("\nFirst 5 Rows:")
    print(df.head(5).to_string(index=False))
    
    result: dict[str, Any] = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_mb": file_size_mb,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": column_names,
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_summary": missing_summary,
        "duplicate_info": duplicate_info,
        "head_5": df.head(5).to_dict(orient="records"),
    }
    
    if "ground_truth" in file_path.name.lower():
        result["ground_truth_details"] = inspect_ground_truth_details(df, file_path)

    del df
    gc.collect()

    return result


def inspect_all_datasets() -> dict[str, Any]:
    """Inspect all expected TSV files in the train and test directories."""
    print("=" * 80)
    print("STARTING DATASET INSPECTION")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_DIR}")
    print("=" * 80)
    
    results: dict[str, Any] = {}
    for file_path in EXPECTED_FILES:
        inspection_result = inspect_file(file_path)
        if inspection_result is not None:
            results[file_path.name] = inspection_result
        else:
            results[file_path.name] = {"error": f"Could not inspect {file_path.name}"}
            
    print(f"\n{'=' * 80}")
    print("DATASET INSPECTION COMPLETE")
    print(f"Successfully inspected {len([r for r in results.values() if 'error' not in r])}/{len(EXPECTED_FILES)} files.")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    inspect_all_datasets()
