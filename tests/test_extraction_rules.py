"""Unit tests for the rule-based extraction logic.

Run with:  python -m unittest discover -s tests
(uses the standard library unittest module -- no extra dependencies needed)
"""

import json
import sys
import unittest
from pathlib import Path

# Make backend/ importable as a package root, same as it is inside the container.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from utils.arabic_normalizer import normalize_arabic_digits, normalize_text  # noqa: E402
from utils.ocr_quality import calculate_ocr_quality  # noqa: E402
from services.validation_service import is_valid_national_id  # noqa: E402
from services.person_extraction_service import (  # noqa: E402
    find_national_ids,
    is_company_line,
    is_valid_person_name,
    extract_person_candidates,
    _extract_name_near_national_id,
)
from services.document_extraction_service import extract_document_fields  # noqa: E402
from services.name_splitting_service import split_grouped_name  # noqa: E402
from services.entity_merge_service import (  # noqa: E402
    _is_duplicate_name,
    _deduplicate_and_prefer_best,
    merge_rules_and_ner,
)
from services.evaluation_service import evaluate_accuracy  # noqa: E402
from schemas.person_schema import PersonRecord  # noqa: E402


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

    def test_extracts_date_with_spaced_colon(self):
        document = extract_document_fields("التاريخ : 2026/02/04")
        self.assertEqual(document.document_date.value, "2026/02/04")
        self.assertFalse(document.document_date.needs_review)

    def test_normalizes_compact_date_without_separators(self):
        document = extract_document_fields("التاريخ: 20260204")
        self.assertEqual(document.document_date.value, "2026/02/04")

    def test_normalizes_compact_date_with_partial_separator(self):
        document = extract_document_fields("التاريخ: 2026/0204")
        self.assertEqual(document.document_date.value, "2026/02/04")

    def test_normalizes_day_first_date_to_year_first(self):
        document = extract_document_fields("بتاريخ 04/02/2026")
        self.assertEqual(document.document_date.value, "2026/02/04")

    def test_searches_header_text_before_full_text(self):
        # The header crop found the date; the full page didn't have it at
        # all. document_date should still be found via header_text.
        document = extract_document_fields(
            cleaned_text="محكمة اربد الابتدائية\nبالدعوى رقم 2026/41",
            header_text="الرقم: 2026/41\nالتاريخ: 20260204",
        )
        self.assertEqual(document.document_date.value, "2026/02/04")
        self.assertFalse(document.document_date.needs_review)


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


class TestNameNearNationalId(unittest.TestCase):
    def test_extracts_full_name_before_id(self):
        text = "للمستدعى ضده سامي فواز يوسف الخوالدة و رقم وطني (98123456789)"
        name = _extract_name_near_national_id(text, "98123456789")
        self.assertEqual(name, "سامي فواز يوسف الخوالدة")

    def test_returns_none_when_id_not_in_text(self):
        self.assertIsNone(_extract_name_near_national_id("لا يوجد رقم هنا", "98123456789"))


class TestMergePrefersRulesOverNer(unittest.TestCase):
    def test_rules_record_wins_and_source_is_combined(self):
        rules_record = PersonRecord(
            full_name="سامي فواز يوسف الخوالدة",
            national_id="98123456789",
            person_type="Individual",
            confidence=0.9,
            needs_review=False,
            source="rules",
        )
        ner_record = PersonRecord(
            full_name="فواز بوسف",
            national_id="98123456789",
            person_type="Individual",
            confidence=0.75,
            needs_review=False,
            source="ner",
        )

        merged = _deduplicate_and_prefer_best([rules_record, ner_record])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].full_name, "سامي فواز يوسف الخوالدة")
        self.assertEqual(merged[0].national_id, "98123456789")
        self.assertEqual(merged[0].source, "rules+ner")


class TestDocumentNumberFallback(unittest.TestCase):
    def test_falls_back_to_case_number_when_document_number_missing(self):
        # No "رقم" substring anywhere else, so the document_number keyword
        # search finds nothing and must fall back to case_number.
        document = extract_document_fields("محكمة اربد الابتدائية\nبالدعوى 2026/41")
        self.assertEqual(document.case_number.value, "2026/41")
        self.assertEqual(document.document_number.value, "2026/41")
        self.assertTrue(document.document_number.needs_review)

    def test_case_number_not_corrupted_by_following_date_line(self):
        # Regression test: a colon on the *next* line (the date) must not
        # leak into the case_number value extracted from the previous line.
        document = extract_document_fields(
            "محكمة اربد الابتدائية\nبالدعوى رقم 2026/41\nالتاريخ : 2026/02/04"
        )
        self.assertEqual(document.case_number.value, "2026/41")
        self.assertEqual(document.document_date.value, "2026/02/04")


class TestDirectLegalPhraseExtraction(unittest.TestCase):
    def test_extracts_name_and_id_directly(self):
        text = "للمستدعى ضده سامي فواز يوسف الخوالدة و رقم وطني (98123456789)"
        records = extract_person_candidates(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].full_name, "سامي فواز يوسف الخوالدة")
        self.assertEqual(records[0].national_id, "98123456789")
        self.assertEqual(records[0].extraction_method, "direct_legal_phrase")

    def test_ignores_property_text_before_the_legal_phrase(self):
        # Regression test for the reported bug: a property/land clause
        # sitting right before the real person clause must not get pulled
        # in as (part of) the person's name.
        text = (
            "قرار الحجز التحفظي على قطعة الارض حوض بوسف الخوالدة\n"
            "للمستدعى ضده سامي فواز يوسف الخوالدة و رقم وطني (98123456789)\n"
        )
        records = extract_person_candidates(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].full_name, "سامي فواز يوسف الخوالدة")


class TestLineContextFallbackExtraction(unittest.TestCase):
    def test_fires_when_id_connector_phrase_is_missing(self):
        # No "رقم وطني" connector at all -- the direct-phrase regex (Method 1)
        # cannot match this, so the line-context fallback (Method 3) must
        # still find the name from the legal-hint phrase before the ID.
        text = "للمستدعى ضده سامي فواز يوسف الخوالدة (98123456789)"
        records = extract_person_candidates(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].full_name, "سامي فواز يوسف الخوالدة")
        self.assertEqual(records[0].national_id, "98123456789")
        self.assertEqual(records[0].source, "rules")
        self.assertFalse(records[0].needs_review)

    def test_still_rejects_property_clause_without_connector(self):
        text = (
            "قرار الحجز التحفظي على قطعة الارض حوض بوسف الخوالدة\n"
            "للمستدعى ضده سامي فواز يوسف الخوالدة (98123456789)\n"
        )
        records = extract_person_candidates(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].full_name, "سامي فواز يوسف الخوالدة")


class TestInvalidNameFilter(unittest.TestCase):
    def test_rejects_property_description(self):
        self.assertFalse(is_valid_person_name("قطعة الارض حوض بوسف الخوالدة"))

    def test_rejects_bare_legal_term(self):
        self.assertFalse(is_valid_person_name("الحجز"))

    def test_rejects_short_name_without_id_confirmation(self):
        self.assertFalse(is_valid_person_name("فواز سامي", has_national_id=False))

    def test_accepts_full_name(self):
        self.assertTrue(is_valid_person_name("سامي فواز يوسف الخوالدة"))


class TestNerOnlyPersonsExcluded(unittest.TestCase):
    def test_bare_ner_person_not_added_to_final_persons(self):
        cleaned_text = "نص لا علاقة له بالرقم الوطني هنا."
        ner_result = {
            "ner_status": "success",
            "selected_engine": "stanza",
            "entities": [
                {"text": "فواز سامي", "label": "PERSON", "start_char": None, "end_char": None, "confidence": 0.75},
            ],
            "error": None,
        }

        persons, suggested = merge_rules_and_ner(cleaned_text, [], ner_result)

        self.assertEqual(persons, [])
        self.assertEqual(len(suggested), 1)
        self.assertEqual(suggested[0]["text"], "فواز سامي")


class TestEvaluationAccuracy(unittest.TestCase):
    def setUp(self):
        fixture_path = (
            Path(__file__).resolve().parent / "expected_outputs" / "sample_1_expected.json"
        )
        with open(fixture_path, encoding="utf-8") as f:
            self.expected = json.load(f)

    def test_perfect_match_scores_full_accuracy(self):
        actual = {
            "document": {
                "court_name": {"value": "محكمة اربد الابتدائية"},
                "case_number": {"value": "2026/41"},
                "document_number": {"value": "2026/41"},
                "document_date": {"value": "2026/02/04"},
            },
            "persons": [
                {"full_name": "سامي فواز يوسف الخوالدة", "national_id": "98123456789", "person_type": "Individual"},
            ],
        }
        result = evaluate_accuracy(self.expected, actual)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["correct_fields"], result["total_fields"])

    def test_wrong_name_and_missing_date_reduce_accuracy(self):
        actual = {
            "document": {
                "court_name": {"value": "محكمة اربد الابتدائية"},
                "case_number": {"value": "2026/41"},
                "document_number": {"value": "2026/41"},
                "document_date": {"value": None},
            },
            "persons": [
                {"full_name": "قطعة الارض حوض بوسف الخوالدة", "national_id": "98123456789", "person_type": "Individual"},
            ],
        }
        result = evaluate_accuracy(self.expected, actual)
        self.assertEqual(result["total_fields"], 7)
        self.assertEqual(result["correct_fields"], 5)
        self.assertFalse(result["field_results"]["document_date"])
        self.assertFalse(result["field_results"]["person_full_name"])


if __name__ == "__main__":
    unittest.main()
