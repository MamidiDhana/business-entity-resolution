"""Unit tests for preprocessing pipeline using synthetic DataFrames."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.preprocessing import (
    DERIVED_COLUMNS,
    RAW_COLUMNS,
    load_and_preprocess_sample,
    preprocess_dataframe,
    preprocess_tsv_chunked,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for DataFrame preprocessing, chunking, and column integrity."""

    def setUp(self) -> None:
        self.synthetic_data = pd.DataFrame({
            "entity_id": ["E-101", "E-102", "E-103", "E-104", "E-105"],
            "business_name": [
                "Acme Corporation",
                "Beta Private Limited",
                "मॉडर्न फाइनेंस Ltd.",
                None,
                "Delta & Sons, LLC",
            ],
            "business_address": [
                "123 Main Street, Suite 100",
                "456 Industrial Boulevard",
                "No 10 Enkay Square, 448A",
                "P.O. Box 789",
                None,
            ],
            "country": ["US", "India", "India", "France", None],
        })

    def test_original_columns_preserved_and_not_overwritten(self) -> None:
        """Verify that all raw columns remain completely intact and unchanged."""
        orig_copy = self.synthetic_data.copy(deep=True)
        result = preprocess_dataframe(self.synthetic_data)

        # Check raw columns are present
        for col in RAW_COLUMNS:
            self.assertIn(col, result.columns)
            pd.testing.assert_series_equal(result[col], orig_copy[col])

        # Check input dataframe was not mutated
        pd.testing.assert_frame_equal(self.synthetic_data, orig_copy)

    def test_derived_columns_created(self) -> None:
        """Verify derived columns exist and contain normalized values."""
        result = preprocess_dataframe(self.synthetic_data)

        for col in DERIVED_COLUMNS:
            self.assertIn(col, result.columns)

        # Specific normalization checks
        self.assertEqual(result.loc[0, "name_normalized"], "acme corp")
        self.assertEqual(result.loc[0, "address_normalized"], "123 main st ste 100")
        self.assertEqual(result.loc[0, "country_normalized"], "us")

        self.assertEqual(result.loc[1, "name_normalized"], "beta pvt ltd")
        self.assertEqual(result.loc[1, "address_normalized"], "456 industrial blvd")
        self.assertEqual(result.loc[1, "country_normalized"], "india")

        self.assertEqual(result.loc[2, "name_normalized"], "मॉडर्न फाइनेंस ltd")

    def test_missing_values_handled_without_literal_nan(self) -> None:
        """Verify nulls in raw columns become empty strings in normalized columns."""
        result = preprocess_dataframe(self.synthetic_data)

        # Row 3 has None for business_name
        self.assertEqual(result.loc[3, "name_normalized"], "")

        # Row 4 has None for business_address and country
        self.assertEqual(result.loc[4, "address_normalized"], "")
        self.assertEqual(result.loc[4, "country_normalized"], "")

    def test_deterministic_preprocessing(self) -> None:
        """Verify preprocessing is strictly deterministic."""
        run1 = preprocess_dataframe(self.synthetic_data)
        run2 = preprocess_dataframe(self.synthetic_data)
        pd.testing.assert_frame_equal(run1, run2)

    def test_missing_required_column_raises_error(self) -> None:
        """Verify that missing essential columns raises a ValueError."""
        invalid_df = pd.DataFrame({"entity_id": ["E-1"], "business_name": ["Foo"]})
        with self.assertRaises(ValueError):
            preprocess_dataframe(invalid_df)

    def test_chunked_preprocessing_stream(self) -> None:
        """Verify preprocess_tsv_chunked correctly processes and yields chunks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tsv_path = Path(tmp_dir) / "synthetic.tsv"
            self.synthetic_data.to_csv(tsv_path, sep="\t", index=False)

            chunks = list(preprocess_tsv_chunked(tsv_path, chunksize=2))
            self.assertEqual(len(chunks), 3)  # 5 rows in chunks of 2 -> 2 + 2 + 1

            combined = pd.concat(chunks, ignore_index=True)
            expected = preprocess_dataframe(self.synthetic_data)
            pd.testing.assert_frame_equal(combined, expected)


if __name__ == "__main__":
    unittest.main()
