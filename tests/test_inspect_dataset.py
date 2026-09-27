"""Unit tests for dataset inspection and dataset file integrity."""

from __future__ import annotations

from pathlib import Path
import unittest
import pandas as pd

from src.inspect_dataset import (
    EXPECTED_FILES,
    PROJECT_ROOT,
    TEST_DIR,
    TRAIN_DIR,
    inspect_file,
)


class TestDatasetInspection(unittest.TestCase):
    """Test suite verifying dataset presence, readability, schema, and non-modification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source_files = [
            TRAIN_DIR / "train_source1.tsv",
            TRAIN_DIR / "train_source2.tsv",
            TRAIN_DIR / "train_source3.tsv",
            TEST_DIR / "test_source1.tsv",
            TEST_DIR / "test_source2.tsv",
            TEST_DIR / "test_source3.tsv",
        ]
        cls.ground_truth_file = TRAIN_DIR / "train_ground_truth.tsv"
        cls.all_files = cls.source_files + [cls.ground_truth_file]

    def test_expected_dataset_files_can_be_located(self) -> None:
        """Verify that all 7 expected TSV files exist in the project directory."""
        self.assertEqual(len(EXPECTED_FILES), 7)
        for file_path in EXPECTED_FILES:
            self.assertTrue(
                file_path.exists(),
                f"Missing expected dataset file: {file_path}",
            )
            self.assertTrue(
                file_path.is_file(),
                f"Path is not a regular file: {file_path}",
            )
            self.assertGreater(
                file_path.stat().st_size,
                0,
                f"File is unexpectedly empty (0 bytes): {file_path}",
            )

    def test_tsv_files_can_be_read_with_tab_separator(self) -> None:
        """Verify that all TSV files can be read using pandas with sep='\\t' and dtype=str."""
        for file_path in self.all_files:
            try:
                sample_df = pd.read_csv(file_path, sep="\t", dtype=str, nrows=10)
            except Exception as exc:
                self.fail(f"Failed to read {file_path.name} with tab separator: {exc}")

            self.assertFalse(sample_df.empty, f"Sample from {file_path.name} is empty")
            self.assertGreaterEqual(
                len(sample_df.columns),
                2,
                f"{file_path.name} has fewer columns than expected: {sample_df.columns}",
            )

    def test_expected_source_files_contain_required_columns(self) -> None:
        """Verify that all 6 source files contain entity_id, business_name, business_address, country."""
        required_columns = {"entity_id", "business_name", "business_address", "country"}
        for file_path in self.source_files:
            header_df = pd.read_csv(file_path, sep="\t", dtype=str, nrows=0)
            columns_set = set(header_df.columns)
            self.assertTrue(
                required_columns.issubset(columns_set),
                f"{file_path.name} is missing required columns. "
                f"Expected subset: {required_columns}, Found: {columns_set}",
            )

    def test_ground_truth_file_schema(self) -> None:
        """Verify train_ground_truth.tsv exists, can be read, and has non-empty column schema."""
        header_df = pd.read_csv(self.ground_truth_file, sep="\t", dtype=str, nrows=5)
        self.assertGreater(len(header_df.columns), 0, "Ground truth file has no columns")
        self.assertEqual(len(header_df), 5, "Ground truth sample has fewer than 5 rows")

    def test_inspection_does_not_modify_dataset(self) -> None:
        """Verify that running inspect_file does not alter dataset file contents, size, or timestamps."""
        for file_path in self.all_files:
            stat_before = file_path.stat()
            size_before = stat_before.st_size
            mtime_before = stat_before.st_mtime_ns

            # Run inspect_file function
            result = inspect_file(file_path)
            self.assertIsNotNone(result, f"Inspection returned None for {file_path.name}")

            stat_after = file_path.stat()
            size_after = stat_after.st_size
            mtime_after = stat_after.st_mtime_ns

            self.assertEqual(
                size_before,
                size_after,
                f"File size changed for {file_path.name}: {size_before} != {size_after}",
            )
            self.assertEqual(
                mtime_before,
                mtime_after,
                f"File mtime changed for {file_path.name}: {mtime_before} != {mtime_after}",
            )


if __name__ == "__main__":
    unittest.main()
