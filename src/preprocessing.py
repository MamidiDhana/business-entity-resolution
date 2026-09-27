"""Scalable preprocessing pipeline for Business Entity Resolution.

Handles chunked and in-memory preprocessing of entity source datasets.
Preserves raw input columns and appends vectorized normalized representations:
  - name_normalized
  - address_normalized
  - country_normalized
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.normalization import (
        normalize_business_address_series,
        normalize_business_name_series,
        normalize_country_series,
    )
except ImportError:
    from normalization import (
        normalize_business_address_series,
        normalize_business_name_series,
        normalize_country_series,
    )

# Ensure standard output can print Unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
DERIVED_COLUMNS = ["name_normalized", "address_normalized", "country_normalized"]
EXPECTED_COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS

SMOKE_TEST_FILES = [
    TRAIN_DIR / "train_source1.tsv",
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv",
    TEST_DIR / "test_source1.tsv",
    TEST_DIR / "test_source2.tsv",
    TEST_DIR / "test_source3.tsv",
]


def preprocess_dataframe(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Preprocess an entity dataset DataFrame using vectorized normalization.
    
    Preserves all original columns without overwriting, adding:
      - name_normalized
      - address_normalized
      - country_normalized
      
    Args:
        df: DataFrame containing entity_id, business_name, business_address, country.
        copy: Whether to create a shallow/defensive copy of the DataFrame.
        
    Returns:
        DataFrame with original columns and derived normalized columns.
    """
    if copy:
        result_df = df.copy()
    else:
        result_df = df
        
    # Verify required source columns exist
    missing_cols = [col for col in RAW_COLUMNS if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required source columns: {missing_cols}")
        
    # Generate vectorized normalized columns
    result_df["name_normalized"] = normalize_business_name_series(result_df["business_name"])
    result_df["address_normalized"] = normalize_business_address_series(result_df["business_address"])
    result_df["country_normalized"] = normalize_country_series(result_df["country"])
    
    return result_df


def preprocess_tsv_chunked(
    file_path: Path | str,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Stream and preprocess a large TSV file in chunks to maintain low memory usage.
    
    Args:
        file_path: Path to the TSV file.
        chunksize: Number of rows per chunk.
        
    Yields:
        Preprocessed DataFrame chunks with original and normalized columns.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {path}")
        
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
        yield preprocess_dataframe(chunk, copy=False)


def load_and_preprocess_sample(
    file_path: Path | str,
    nrows: int = 10_000,
) -> pd.DataFrame:
    """Read a limited sample of rows from a TSV file and preprocess it."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {path}")
        
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    return preprocess_dataframe(df, copy=False)


def run_smoke_test(nrows: int = 10_000) -> dict[str, Any]:
    """Run a command-line smoke test processing the first nrows from each large source file.
    
    Measures processing time, memory usage, and inspects sample normalized records.
    """
    print("=" * 80)
    print(f"RUNNING PREPROCESSING SMOKE TEST ({nrows:,} rows per file)")
    print("=" * 80)
    
    tracemalloc.start()
    total_start_time = time.perf_counter()
    
    file_reports: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    
    for file_path in SMOKE_TEST_FILES:
        rel_path = file_path.relative_to(PROJECT_ROOT) if file_path.is_relative_to(PROJECT_ROOT) else file_path
        print(f"\nProcessing {rel_path}...")
        
        t0 = time.perf_counter()
        df_preprocessed = load_and_preprocess_sample(file_path, nrows=nrows)
        elapsed = time.perf_counter() - t0
        
        mem_mb = df_preprocessed.memory_usage(deep=True).sum() / (1024 * 1024)
        num_rows = len(df_preprocessed)
        rows_per_sec = num_rows / max(elapsed, 1e-6)
        
        print(f"  Rows processed:  {num_rows:,}")
        print(f"  Execution time:  {elapsed:.4f} seconds ({rows_per_sec:,.0f} rows/sec)")
        print(f"  DataFrame RAM:   {mem_mb:.2f} MB")
        print(f"  Columns:         {list(df_preprocessed.columns)}")
        
        file_reports.append({
            "file": str(rel_path),
            "rows": num_rows,
            "elapsed_seconds": elapsed,
            "dataframe_memory_mb": mem_mb,
            "rows_per_sec": rows_per_sec,
        })
        
        # Collect sample normalized record
        if not df_preprocessed.empty:
            rec = df_preprocessed.iloc[0].to_dict()
            sample_records.append({
                "source": file_path.name,
                "entity_id": rec["entity_id"],
                "raw_name": rec["business_name"],
                "norm_name": rec["name_normalized"],
                "raw_address": rec["business_address"],
                "norm_address": rec["address_normalized"],
                "raw_country": rec["country"],
                "norm_country": rec["country_normalized"],
            })
            
    total_elapsed = time.perf_counter() - total_start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mem_mb = peak_mem / (1024 * 1024)
    total_rows = sum(r["rows"] for r in file_reports)
    
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Total files processed:  {len(file_reports)}")
    print(f"Total rows processed:   {total_rows:,}")
    print(f"Total execution time:   {total_elapsed:.4f} seconds")
    print(f"Overall throughput:     {total_rows / max(total_elapsed, 1e-6):,.0f} rows/sec")
    print(f"Peak Python Heap RAM:   {peak_mem_mb:.2f} MB")
    
    print("\n--- Sample Normalized Records ---")
    for s in sample_records:
        print(f"\n[{s['source']}] (ID: {s['entity_id']})")
        print(f"  Name:    '{s['raw_name']}' -> '{s['norm_name']}'")
        print(f"  Address: '{s['raw_address']}' -> '{s['norm_address']}'")
        print(f"  Country: '{s['raw_country']}' -> '{s['norm_country']}'")
        
    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        "file_reports": file_reports,
        "sample_records": sample_records,
        "total_rows": total_rows,
        "total_elapsed_seconds": total_elapsed,
        "peak_memory_mb": peak_mem_mb,
    }


if __name__ == "__main__":
    run_smoke_test(nrows=10_000)
