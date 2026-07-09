"""Unit tests for the rule-based extraction logic.

Run with:  python -m unittest discover -s tests
(uses the standard library unittest module -- no extra dependencies needed)
"""

import sys
import unittest
from pathlib import Path

# Make backend/ importable as a package root, same as it is inside the container.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from utils.arabic_normalizer import normalize_arabic_digits, normalize_text  # noqa: E402
from utils.ocr_quality import calculate_ocr_quality  # noqa: E402
from services.validation_service import is_valid_national_id  # noqa: E402
from services.person_extraction_service import find_national_ids, is_company_line  # noqa: E402
from services.document_extraction_service import extract_document_fields  # noqa: E402
from services.name_splitting_service import split_grouped_name  # noqa: E402
from services.entity_merge_service import _is_duplicate_name  # noqa: E402


class TestArabicDigitNormalization(unittest.TestCase):
    def test_converts_arabic_indic_digits(self):
        self.assertEqual(normalize_arabic_digits("١٢٣٤"), "1234")

    def test_leaves_ascii_digits_untouched(self):
        self.assertEqual(normalize_arabic_digits("1234"), "1234")

    def test_full_normalize_pipeline_on_case_number_line(self):
        result = normalize_text("رقم القضية : ١٢٣٤/٢٠٢٦")
        self.assertIn("1234/2026", result["cleaned_text"])


class TestNationalIdValidation(unittest.TestCase):
    def test_valid_11_digit_id(self):
        self.assertTrue(is_valid_national_id("12345678901"))

    def test_invalid_short_id(self):
        self.assertFalse(is_valid_national_id("123456"))

    def test_invalid_long_id(self):
        self.assertFalse(is_valid_national_id("123456789012"))

    def test_none_is_allowed(self):
        self.assertTrue(is_valid_national_id(None))

    def test_regex_does_not_match_within_longer_digit_run(self):
        ids = find_national_ids("رقم الهاتف 123456789012345")
        self.assertEqual(ids, [])

    def test_regex_finds_exact_11_digits(self):
        ids = find_national_ids("الرقم الوطني 98765432109 للمدين")
        self.assertEqual(ids, ["98765432109"])


class TestDateExtraction(unittest.TestCase):
    def test_extracts_date_after_keyword(self):
        document = extract_document_fields("بتاريخ 2026/07/09")
        self.assertEqual(document.document_date.value, "2026/07/09")

    def test_missing_date_returns_null(self):
        document = extract_document_fields("لا يوجد تاريخ هنا")
        self.assertIsNone(document.document_date.value)
        self.assertTrue(document.document_date.needs_review)


class TestGroupedNameSplitting(unittest.TestCase):
    def test_splits_three_brothers(self):
        names = split_grouped_name("أحمد ومحمد وخالد أبناء محمود سالم")
        self.assertEqual(len(names), 3)
        self.assertIn("أحمد محمود سالم", names)
        self.assertIn("محمد محمود سالم", names)
        self.assertIn("خالد محمود سالم", names)

    def test_single_name_not_split(self):
        names = split_grouped_name("أحمد محمود سالم")
        self.assertEqual(names, ["أحمد محمود سالم"])


class TestCompanyDetection(unittest.TestCase):
    def test_detects_sharika_keyword(self):
        self.assertTrue(is_company_line("شركة المثال للتجارة"))

    def test_detects_muassasa_keyword(self):
        self.assertTrue(is_company_line("مؤسسة الأمل"))

    def test_individual_line_is_not_company(self):
        self.assertFalse(is_company_line("المدين: أحمد محمود سالم"))


class TestOCRQualityScoring(unittest.TestCase):
    def test_empty_text_scores_zero(self):
        self.assertEqual(calculate_ocr_quality("", 0.9), 0.0)

    def test_legal_text_scores_higher_than_gibberish(self):
        legal_text = "محكمة بداية عمان\nرقم القضية 1234\nتنفيذ حجز على المدين"
        gibberish = "xzq  ##@ 111"
        legal_score = calculate_ocr_quality(legal_text, 0.8)
        gibberish_score = calculate_ocr_quality(gibberish, 0.8)
        self.assertGreater(legal_score, gibberish_score)

    def test_score_is_clamped_between_zero_and_one(self):
        score = calculate_ocr_quality("محكمة قضية تنفيذ حجز المدين رقم تاريخ" * 5, 1.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)


class TestDuplicatePersonMerge(unittest.TestCase):
    def test_exact_name_match_is_duplicate(self):
        self.assertTrue(_is_duplicate_name("أحمد محمود سالم", ["أحمد محمود سالم"]))

    def test_different_names_are_not_duplicate(self):
        self.assertFalse(_is_duplicate_name("خالد علي حسن", ["أحمد محمود سالم"]))

    def test_substring_match_is_duplicate(self):
        self.assertTrue(_is_duplicate_name("أحمد محمود", ["أحمد محمود سالم"]))


if __name__ == "__main__":
    unittest.main()
