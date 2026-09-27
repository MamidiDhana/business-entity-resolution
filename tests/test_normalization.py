"""Unit tests for text normalization functions using small synthetic data."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.normalization import (
    is_missing_value,
    normalize_business_address,
    normalize_business_address_series,
    normalize_business_name,
    normalize_business_name_series,
    normalize_country,
    normalize_country_series,
)


class TestNormalization(unittest.TestCase):
    """Test suite for business name, address, and country normalization."""

    # ---------------------------------------------------------
    # Missing Value Handling Tests
    # ---------------------------------------------------------
    def test_is_missing_value(self) -> None:
        """Verify proper identification of missing, null, or corrupted missing values."""
        self.assertTrue(is_missing_value(None))
        self.assertTrue(is_missing_value(np.nan))
        self.assertTrue(is_missing_value(pd.NA))
        self.assertTrue(is_missing_value(""))
        self.assertTrue(is_missing_value("   "))
        self.assertTrue(is_missing_value("nan"))
        self.assertTrue(is_missing_value("NAN"))
        self.assertTrue(is_missing_value("null"))
        self.assertTrue(is_missing_value("None"))
        self.assertTrue(is_missing_value("<NA>"))

        self.assertFalse(is_missing_value("Acme"))
        self.assertFalse(is_missing_value("0"))
        self.assertFalse(is_missing_value("US"))

    def test_missing_values_do_not_produce_literal_nan_strings(self) -> None:
        """Verify that missing inputs return empty strings and never 'nan'."""
        missing_inputs = [None, np.nan, pd.NA, "", "   ", "nan", "NaN", "null", "None"]
        for val in missing_inputs:
            self.assertEqual(normalize_business_name(val), "")
            self.assertEqual(normalize_business_address(val), "")
            self.assertEqual(normalize_country(val), "")

        # Also test vectorized Series
        series = pd.Series(missing_inputs)
        self.assertTrue((normalize_business_name_series(series) == "").all())
        self.assertTrue((normalize_business_address_series(series) == "").all())
        self.assertTrue((normalize_country_series(series) == "").all())

    # ---------------------------------------------------------
    # Business Name Normalization Tests
    # ---------------------------------------------------------
    def test_normalize_business_name_basic_and_case(self) -> None:
        """Verify case folding, whitespace stripping, and ampersand conversion."""
        self.assertEqual(normalize_business_name("  Apple Computer  "), "apple computer")
        self.assertEqual(normalize_business_name("Barnes & Noble"), "barnes and noble")
        self.assertEqual(normalize_business_name("AT&T"), "at and t")

    def test_normalize_business_name_legal_suffixes(self) -> None:
        """Verify that legal suffix variations are properly normalized."""
        test_cases = [
            ("Alpha Incorporated", "alpha inc"),
            ("Alpha Inc.", "alpha inc"),
            ("Beta Corporation", "beta corp"),
            ("Beta Corp.", "beta corp"),
            ("Gamma Company", "gamma co"),
            ("Gamma Co.", "gamma co"),
            ("Delta Limited", "delta ltd"),
            ("Delta Ltd.", "delta ltd"),
            ("Epsilon Private Limited", "epsilon pvt ltd"),
            ("Epsilon Pvt. Ltd.", "epsilon pvt ltd"),
            ("Zeta L.L.C.", "zeta llc"),
            ("Zeta LLC", "zeta llc"),
            ("Theta Public Limited", "theta plc"),
            ("Iota S.A.R.L.", "iota sarl"),
            ("Kappa S.A.S.", "kappa sas"),
            ("Lambda S.A.", "lambda sa"),
            ("Mu GmbH", "mu gmbh"),
        ]
        for raw, expected in test_cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_business_name(raw), expected)

    def test_normalize_business_name_multilingual_and_non_ascii(self) -> None:
        """Verify support for international non-ASCII scripts without mangling."""
        # Devanagari Hindi
        self.assertEqual(
            normalize_business_name("मॉडर्न फाइनेंस Ltd."),
            "मॉडर्न फाइनेंस ltd",
        )
        # Accented French
        self.assertEqual(
            normalize_business_name("SCI Ptit Àmicale S.A.R.L."),
            "sci ptit àmicale sarl",  # Preserves valid non-ASCII letter
        )
        # Punctuation cleaning with symbols
        self.assertEqual(
            normalize_business_name("<< Team Ecole >> (France)"),
            "team ecole france",
        )

    # ---------------------------------------------------------
    # Business Address Normalization Tests
    # ---------------------------------------------------------
    def test_normalize_business_address_abbreviations(self) -> None:
        """Verify justified address abbreviation normalization."""
        test_cases = [
            ("123 Main Street, Suite 400", "123 main st ste 400"),
            ("456 Park Avenue, Floor 2", "456 park ave fl 2"),
            ("789 Industrial Boulevard", "789 industrial blvd"),
            ("101 Ocean Drive, Apartment 3B", "101 ocean dr apt 3b"),
            ("55 Oak Lane, Building C", "55 oak ln bldg c"),
            ("99 Elm Court", "99 elm ct"),
            ("P.O. Box 1234", "po box 1234"),
            ("Post Office Box 567", "po box 567"),
            ("POB 890", "po box 890"),
            ("Route 66 Highway", "route 66 hwy"),
        ]
        for raw, expected in test_cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_business_address(raw), expected)

    def test_normalize_business_address_numbers_and_non_ascii(self) -> None:
        """Verify numbers, unit tokens, and non-ASCII address components are preserved."""
        raw_addr = "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR"
        expected = "no 10 enkay square 448a udyog vihar phase v gurugram gurgaon hr"
        self.assertEqual(normalize_business_address(raw_addr), expected)

        # Indian address with Hindi state name
        raw_indic = "H.no 910 A 3503, Mumbai, महाराष्ट्र"
        expected_indic = "h no 910 a 3503 mumbai महाराष्ट्र"
        self.assertEqual(normalize_business_address(raw_indic), expected_indic)

    # ---------------------------------------------------------
    # Country Normalization Tests
    # ---------------------------------------------------------
    def test_normalize_country(self) -> None:
        """Verify country normalization preserves countries while cleaning case and whitespace."""
        test_cases = [
            ("US", "us"),
            ("U.S.", "us"),
            ("U.S.A.", "usa"),
            ("  United States  ", "united states"),
            ("India", "india"),
            ("France", "france"),
            ("Deutschland", "deutschland"),
            ("Brasil", "brasil"),
            ("IN", "in"),
            ("FR", "fr"),
        ]
        for raw, expected in test_cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), expected)

    # ---------------------------------------------------------
    # Vectorized vs Scalar Equivalence Tests
    # ---------------------------------------------------------
    def test_vectorized_matches_scalar_exact(self) -> None:
        """Verify that vectorized Series functions produce identical output to scalar functions."""
        sample_names = [
            "Apple, Inc.",
            "Beta Private Limited",
            "मॉडर्न फाइनेंस Ltd.",
            None,
            "nan",
            "AT&T Co.",
            "Omega LLC",
            "   ",
        ]
        sample_addresses = [
            "123 Main Street, Apt. 4B",
            "P.O. Box 100",
            "No 10 Enkay Square, 448A",
            None,
            "null",
            "63 R. DE DIEPPE, LILLE",
            "   ",
        ]
        sample_countries = ["US", "U.S.A.", "India", "France", None, "nan", "   "]

        # Names
        scalar_names = [normalize_business_name(x) for x in sample_names]
        vec_names = list(normalize_business_name_series(pd.Series(sample_names)))
        self.assertEqual(scalar_names, vec_names)

        # Addresses
        scalar_addrs = [normalize_business_address(x) for x in sample_addresses]
        vec_addrs = list(normalize_business_address_series(pd.Series(sample_addresses)))
        self.assertEqual(scalar_addrs, vec_addrs)

        # Countries
        scalar_c = [normalize_country(x) for x in sample_countries]
        vec_c = list(normalize_country_series(pd.Series(sample_countries)))
        self.assertEqual(scalar_c, vec_c)


if __name__ == "__main__":
    unittest.main()
