"""Targeted regression tests for the structured target-list extraction fix.

These cover the root problems reported against the real OCR sample for
case 1208/2026 / document UW-2026-0008 (data/ocr_outputs/
fd0b6f53-e826-4735-b11c-2791d3b0e748.json):

- records mixed across lines
- person type leaking from a company line onto an individual line
- lost/duplicated national IDs
- damaged Arabic company word order
- truncated court name
- hardcoded confidence

Run with:  python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from utils.arabic_normalizer import normalize_text  # noqa: E402
from services.person_extraction_service import extract_person_candidates  # noqa: E402
from services.document_extraction_service import extract_document_fields  # noqa: E402
from services.validation_service import validate_all  # noqa: E402
from services.evaluation_service import evaluate_accuracy_multi  # noqa: E402
from services.entity_merge_service import (  # noqa: E402
    merge_rules_and_ner,
    _deduplicate_and_prefer_best,
)
from schemas.person_schema import PersonRecord  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_OCR_PATH = (
    REPO_ROOT / "data" / "ocr_outputs" / "fd0b6f53-e826-4735-b11c-2791d3b0e748.json"
)
# Second real sample (case 1201/2026 / document UW-2026-0101): this one
# needs *two* leading name-slots recovered for one person and *one*
# interior (grandfather) slot recovered for another -- the ambiguity that
# a naive "always prepend" or "always insert before last word" rule can't
# resolve on its own.
SAMPLE_OCR_PATH_2 = (
    REPO_ROOT / "data" / "ocr_outputs" / "2035e211-b13a-4167-8484-a781fef95f36.json"
)


def _persons_by_type(persons, person_type):
    return [p for p in persons if p.person_type == person_type]


def _run_pipeline(cleaned_text, header_text=None):
    """Mirror backend/api/process.py's normalize -> extract order."""
    normalized = normalize_text(cleaned_text)["cleaned_text"]
    normalized_header = normalize_text(header_text)["cleaned_text"] if header_text else None

    persons = extract_person_candidates(normalized)
    document = extract_document_fields(normalized, header_text=normalized_header)

    return document, persons


class TestSuppliedStructuredDocument(unittest.TestCase):
    """Test 1: the exact supplied structured document (real OCR text,
    including its line breaks, OCR mistakes, and scattered tokens)."""

    def setUp(self):
        data = json.loads(SAMPLE_OCR_PATH.read_text(encoding="utf-8"))
        self.header_text = data["header_text"]
        self.full_page_text = data["full_page_text"]
        combined_text = f"{self.header_text}\n{self.full_page_text}"
        self.document, self.persons = _run_pipeline(combined_text, self.header_text)

    def test_court_name_captures_kind_and_city(self):
        self.assertEqual(self.document.court_name.value, "محكمة بداية عمان")

    def test_case_and_document_numbers(self):
        self.assertEqual(self.document.case_number.value, "1208/2026")
        self.assertEqual(self.document.document_number.value, "UW-2026-0008")

    def test_document_date(self):
        self.assertEqual(self.document.document_date.value, "2026/07/09")

    def test_three_persons_extracted(self):
        self.assertEqual(len(self.persons), 3)

    def test_first_individual_first_name_recovered(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98000000081", by_id)
        self.assertEqual(by_id["98000000081"].full_name, "يوسف وليد عادل القضاه")
        self.assertEqual(by_id["98000000081"].person_type, "Individual")

    def test_second_individual_not_mixed_with_first(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98000000082", by_id)
        self.assertEqual(by_id["98000000082"].full_name, "محمود عادل مصطفى بني يونس")
        self.assertEqual(by_id["98000000082"].person_type, "Individual")

    def test_company_record_correct_order_and_registration(self):
        companies = _persons_by_type(self.persons, "Company")
        self.assertEqual(len(companies), 1)
        company = companies[0]
        self.assertEqual(company.registration_number, "REG-202608")
        # Hamza folds to plain alef during normalization everywhere in this
        # pipeline (arabic_normalizer.normalize_arabic_letters), so "الأفق"
        # is expected to read "الافق" post-normalization.
        self.assertEqual(company.full_name, "شركة الافق للخدمات المالية")
        self.assertTrue(company.full_name.startswith("شركة"))

    def test_no_record_needs_review(self):
        for person in self.persons:
            self.assertFalse(person.needs_review, msg=person.full_name)

    def test_confidence_not_hardcoded_flat_value(self):
        # Records that needed OCR reconstruction score differently from a
        # record that didn't -- confidence isn't one hardcoded constant.
        confidences = {round(p.confidence, 2) for p in self.persons}
        self.assertIn(0.95, confidences)
        self.assertIn(0.9, confidences)


class TestSectionTitleVariants(unittest.TestCase):
    def _single_individual_text(self, section_title):
        return (
            f"{section_title} :\n"
            "الرقم الوطني : 98111111111\n"
            "سامي فواز يوسف الخوالدة\n"
        )

    def test_section_title_without_slash(self):
        # Test 2
        text = self._single_individual_text("الأشخاص الجهات المطلوبة")
        persons = extract_person_candidates(text)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].national_id, "98111111111")
        self.assertEqual(persons[0].full_name, "سامي فواز يوسف الخوالدة")

    def test_section_title_without_hamza(self):
        # Test 3
        text = self._single_individual_text("الاشخاص / الجهات المطلوبة")
        persons = extract_person_candidates(text)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].national_id, "98111111111")

    def test_section_title_only_jihat(self):
        # Test 4
        text = self._single_individual_text("الجهات المطلوبة")
        persons = extract_person_candidates(text)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].national_id, "98111111111")


class TestRegistrationNumberFormats(unittest.TestCase):
    def _company_text(self, registration_line):
        return (
            "الأشخاص / الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            f"{registration_line}\n"
            "شركة النخبة للحلول التقنية\n"
        )

    def test_registration_with_spaces_normalizes(self):
        # Test 5
        text = self._company_text("REG - 202608 : رقم التسجيل")
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].registration_number, "REG-202608")

    def test_numeric_registration_number(self):
        # Test 6
        text = self._company_text("123456 : رقم التسجيل")
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].registration_number, "123456")


class TestOcrRowSplitAcrossLines(unittest.TestCase):
    def test_person_row_split_into_two_lines(self):
        # Test 7
        text = (
            "الجهات المطلوبة :\n"
            "الرقم الوطني : 98222222222 | نوع الشخص :\n"
            "خالد ناصر مازن العبدالله\n"
        )
        persons = extract_person_candidates(text)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].national_id, "98222222222")
        self.assertEqual(persons[0].full_name, "خالد ناصر مازن العبدالله")
        self.assertEqual(persons[0].person_type, "Individual")

    def test_company_name_and_registration_split_across_lines(self):
        # Test 8
        text = (
            "الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            "REG-303090 : رقم التسجيل\n"
            "للحلول التقنية\n"
            "شركة\n"
        )
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].registration_number, "REG-303090")
        self.assertTrue(companies[0].full_name.startswith("شركة"))
        self.assertNotIn("للحلول التقنية شركة", companies[0].full_name)


class TestFooterNeverBecomesPerson(unittest.TestCase):
    def test_footer_after_watafaddalu_is_not_a_person(self):
        # Test 9
        text = (
            "الجهات المطلوبة :\n"
            "الرقم الوطني : 98333333333 | نوع الشخص :\n"
            "سامي فواز يوسف الخوالدة\n"
            "وتفضلوا بقبول الاحترام\n"
            "كاتب المحكمة\n"
            "صفحة 1\n"
        )
        persons = extract_person_candidates(text)
        names = [p.full_name for p in persons]
        self.assertNotIn("كاتب المحكمة", names)
        self.assertTrue(any("سامي فواز يوسف الخوالدة" == n for n in names))


class TestOrphanRecoveryRestrictions(unittest.TestCase):
    def test_orphan_recovery_never_modifies_an_already_complete_company(self):
        # Test 10: orphan recovery must never run on a company record that
        # structured_target_list already built complete (own keyword,
        # brand, and service phrase all present) -- even when there's an
        # unrelated eligible stray token sitting in the footer pool.
        text = (
            "الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            "REG-404040 : رقم التسجيل\n"
            "شركة النخبة للحلول التقنية\n"
            "وتفضلوا بقبول الاحترام\n"
            "محمود\n"
        )
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].full_name, "شركة النخبة للحلول التقنية")
        self.assertEqual(companies[0].extraction_method, "structured_target_list")


class TestThreeRecordsStayIndependent(unittest.TestCase):
    def setUp(self):
        self.text = (
            "الأشخاص / الجهات المطلوبة :\n"
            "الرقم الوطني : 98444444444 | نوع الشخص :\n"
            "احمد سالم محمود الخالدي\n"
            "الرقم الوطني : 98555555555 | نوع الشخص :\n"
            "سارة رنا هدى الطراونة\n"
            "نوع الشخص : شركة\n"
            "REG-505050 : رقم التسجيل\n"
            "شركة النخبة للحلول التقنية\n"
        )
        self.persons = extract_person_candidates(self.text)

    def test_two_individuals_and_one_company_stay_separate(self):
        # Test 11
        self.assertEqual(len(self.persons), 3)
        individuals = _persons_by_type(self.persons, "Individual")
        companies = _persons_by_type(self.persons, "Company")
        self.assertEqual(len(individuals), 2)
        self.assertEqual(len(companies), 1)

    def test_sharika_keyword_does_not_leak_into_individuals(self):
        # Test 12
        individuals = _persons_by_type(self.persons, "Individual")
        for person in individuals:
            self.assertNotIn("شركة", person.full_name.split())
            self.assertEqual(person.person_type, "Individual")


class TestCompanyNameOrder(unittest.TestCase):
    def test_company_keyword_leads_the_name(self):
        # Test 13 -- same split pattern as the real sample: "شركة" trails
        # the service-type phrase inside the record, and the brand word
        # ("الأفق") landed on its own stray line past the footer boundary.
        text = (
            "الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            "REG-202608 : رقم التسجيل\n"
            "للخدمات المالية\n"
            "شركة\n"
            "وتفضلوا بقبول الاحترام\n"
            "الأفق\n"
        )
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].full_name, "شركة الأفق للخدمات المالية")
        self.assertNotIn("للخدمات المالية شركة", companies[0].full_name)


class TestDeduplication(unittest.TestCase):
    def test_duplicate_individuals_same_national_id_merge(self):
        # Test 14
        text = (
            "الجهات المطلوبة :\n"
            "الرقم الوطني : 98666666666 | نوع الشخص :\n"
            "عمر خالد ناصر العبدالله\n"
            "المدين: عمر خالد ناصر العبدالله و رقم وطني (98666666666)\n"
        )
        persons = extract_person_candidates(text)
        matching = [p for p in persons if p.national_id == "98666666666"]
        self.assertEqual(len(matching), 1)

    def test_duplicate_companies_same_registration_number_merge(self):
        # Test 15
        text = (
            "الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            "REG-606060 : رقم التسجيل\n"
            "شركة النخبة للحلول التقنية\n"
            "نوع الشخص : شركة\n"
            "REG-606060 : رقم التسجيل\n"
            "شركة النخبة للحلول التقنية\n"
        )
        persons = extract_person_candidates(text)
        matching = [p for p in persons if p.registration_number == "REG-606060"]
        self.assertEqual(len(matching), 1)


class TestGenericPhraseRejection(unittest.TestCase):
    def test_generic_instruction_sentence_is_not_a_person(self):
        # Test 16
        text = (
            "استنادا الى قرار المحكمة يرجى اتخاذ الإجراءات اللازمة\n"
            "الأشخاص / الجهات المطلوبة :\n"
            "الرقم الوطني : 98777777777 | نوع الشخص :\n"
            "خالد ناصر مازن العبدالله\n"
        )
        persons = extract_person_candidates(text)
        names = [p.full_name for p in persons]
        self.assertNotIn("يرجى اتخاذ الإجراءات اللازمة", names)
        for name in names:
            self.assertNotIn("يرجى", name.split())
            self.assertNotIn("اللازمة", name.split())


class TestFieldLevelAccuracyOnSuppliedSample(unittest.TestCase):
    """Repeatable field-level accuracy check (correct fields / total
    expected fields) against the hand-labeled fixture for this document,
    covering court_name, case_number, document_number, document_date, and
    every person's full_name/national_id/registration_number/person_type."""

    def test_accuracy_is_at_least_90_percent(self):
        data = json.loads(SAMPLE_OCR_PATH.read_text(encoding="utf-8"))
        combined_text = f"{data['header_text']}\n{data['full_page_text']}"
        document, persons = _run_pipeline(combined_text, data["header_text"])
        validated_document, validated_persons = validate_all(document, list(persons))

        actual = {
            "document": {
                "court_name": {"value": validated_document.court_name.value},
                "case_number": {"value": validated_document.case_number.value},
                "document_number": {"value": validated_document.document_number.value},
                "document_date": {"value": validated_document.document_date.value},
            },
            "persons": [
                {
                    "full_name": p.full_name,
                    "national_id": p.national_id,
                    "registration_number": p.registration_number,
                    "person_type": p.person_type,
                }
                for p in validated_persons
            ],
        }

        expected_path = (
            Path(__file__).resolve().parent / "expected_outputs" / "sample_2_expected.json"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        result = evaluate_accuracy_multi(expected, actual)

        self.assertGreaterEqual(result["accuracy"], 0.90, msg=result["field_results"])


class TestCourtNameNotTruncated(unittest.TestCase):
    def test_court_name_keeps_kind_and_city(self):
        # Test 17
        document = extract_document_fields(
            cleaned_text="محكمة بداية عمان\nرقم القضية : 1208/2026",
            header_text="محكمة\nعمان",
        )
        self.assertEqual(document.court_name.value, "محكمة بداية عمان")
        self.assertNotEqual(document.court_name.value, "محكمة")


class TestSecondSuppliedDocumentMultiGapNames(unittest.TestCase):
    """Real OCR sample for case 1201/2026 / document UW-2026-0101
    (data/ocr_outputs/2035e211-b13a-4167-8484-a781fef95f36.json).

    This document's own OCR text needs two different name-gap repairs at
    once: one person is missing *two* leading slots (given + father name,
    scattered onto their own stray lines in reversed order), the other is
    missing exactly *one interior* slot (grandfather) while its captured
    fragments ("الخطيب" then "ليث محمود") are themselves in the wrong
    line order. It also exercises the "UW" document-number OCR artifact
    (EasyOCR reads the W as a vertical bar: "U|-2026-0101").
    """

    def setUp(self):
        data = json.loads(SAMPLE_OCR_PATH_2.read_text(encoding="utf-8"))
        combined_text = f"{data['header_text']}\n{data['full_page_text']}"
        self.document, self.persons = _run_pipeline(combined_text, data["header_text"])

    def test_document_number_w_is_restored(self):
        self.assertEqual(self.document.document_number.value, "UW-2026-0101")

    def test_case_number_and_date(self):
        self.assertEqual(self.document.case_number.value, "1201/2026")
        self.assertEqual(self.document.document_date.value, "2026/07/10")

    def test_three_persons_extracted_none_bogus(self):
        self.assertEqual(len(self.persons), 3)
        for person in self.persons:
            self.assertNotIn("حجز", person.full_name.split())

    def test_person_missing_two_leading_slots_reconstructed_in_order(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000001", by_id)
        self.assertEqual(by_id["98100000001"].full_name, "يوسف سامر فؤاد العزام")

    def test_person_missing_interior_slot_reconstructed_in_order(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000002", by_id)
        self.assertEqual(by_id["98100000002"].full_name, "ليث محمود سليم الخطيب")

    def test_company_complete_without_recovery(self):
        companies = _persons_by_type(self.persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].full_name, "شركة الريادة للخدمات المالية")
        self.assertEqual(companies[0].registration_number, "REG-202701")
        self.assertEqual(companies[0].extraction_method, "structured_target_list")

    def test_no_record_needs_review(self):
        for person in self.persons:
            self.assertFalse(person.needs_review, msg=person.full_name)

    def test_field_level_accuracy_at_least_90_percent(self):
        validated_document, validated_persons = validate_all(self.document, list(self.persons))

        actual = {
            "document": {
                "court_name": {"value": validated_document.court_name.value},
                "case_number": {"value": validated_document.case_number.value},
                "document_number": {"value": validated_document.document_number.value},
                "document_date": {"value": validated_document.document_date.value},
            },
            "persons": [
                {
                    "full_name": p.full_name,
                    "national_id": p.national_id,
                    "registration_number": p.registration_number,
                    "person_type": p.person_type,
                }
                for p in validated_persons
            ],
        }

        expected_path = (
            Path(__file__).resolve().parent / "expected_outputs" / "sample_3_expected.json"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        result = evaluate_accuracy_multi(expected, actual)

        self.assertGreaterEqual(result["accuracy"], 0.90, msg=result["field_results"])


class TestNerNeverFlipsIndividualToCompany(unittest.TestCase):
    """Regression test for a real-environment bug (Stanza NER enabled):
    a rules-based Individual record with a verified National ID was
    getting merged with a lower-scored NER "PERSON" entity that happened
    to be tagged person_type="Company" (its proximity window brushed up
    against an unrelated company keyword) -- and the merge unconditionally
    adopted "Company" from *either* side, wiping the National ID during
    validation and flagging a perfectly good record for review."""

    def test_merge_keeps_rules_individual_type_despite_mistyped_duplicate(self):
        rules_record = PersonRecord(
            full_name="يوسف سامر فؤاد العزام",
            national_id="98100000001",
            person_type="Individual",
            confidence=0.9,
            needs_review=False,
            source="rules",
            extraction_method="orphan_first_name_recovered",
        )
        mistyped_ner_record = PersonRecord(
            full_name="فؤاد العزام",
            national_id="98100000001",
            person_type="Company",
            confidence=0.75,
            needs_review=False,
            source="ner",
            extraction_method="nearby_ner_person",
        )

        merged = _deduplicate_and_prefer_best([rules_record, mistyped_ner_record])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].person_type, "Individual")
        self.assertEqual(merged[0].national_id, "98100000001")
        self.assertFalse(merged[0].needs_review)

    def test_person_entity_near_company_keyword_goes_to_suggested_not_merged(self):
        cleaned_text = (
            "الرقم الوطني : 98100000001 نوع الشخص : فرد فؤاد العزام "
            "شركة الريادة للخدمات المالية رقم التسجيل : REG-202701"
        )
        start = cleaned_text.find("فؤاد العزام")
        end = start + len("فؤاد العزام")

        ner_result = {
            "ner_status": "success",
            "selected_engine": "stanza",
            "entities": [
                {
                    "text": "فؤاد العزام",
                    "label": "PERSON",
                    "start_char": start,
                    "end_char": end,
                    "confidence": 0.8,
                },
            ],
            "error": None,
        }

        persons, suggested = merge_rules_and_ner(cleaned_text, [], ner_result)

        self.assertEqual(persons, [])
        self.assertEqual(len(suggested), 1)
        self.assertEqual(suggested[0]["reason"], "person_entity_near_company_context")


class TestThirdSuppliedDocumentSevereOcrDamage(unittest.TestCase):
    """Real OCR sample for case 1202/2026 / document UW-2026-0102
    (data/ocr_outputs/5a0e008c-5ceb-456e-b618-04c607464f88.json).

    This document's OCR is damaged badly enough that one person's given,
    father, and family names are *all* missing from its own record and
    scattered as ambiguous/truncated fragments ("قر", "ل", "فر", "راس")
    across the footer. Reconstructing it needs two extra capabilities
    beyond the missing<=2 model: (1) a record whose only surviving word's
    position is otherwise unknown gets it slotted into the interior
    (grandfather) position, between the reversed-and-recovered
    [given, father, family] triplet; (2) fragment stitching -- "قر"
    (too short to trust alone) combines with the orphan token "راس" into
    "قراس" because the fragment's last letter overlaps the token's first
    (the shape produced when EasyOCR splits one cursive-joined word
    across two bounding boxes). This is a best-effort tier (confidence
    0.80, below the 0.90/0.95 confident tiers) -- it still must never
    contaminate the result with unrelated junk ("فر" is a "فرد"
    type-label leftover, not a name fragment, and must never appear).
    """

    def setUp(self):
        data = json.loads(
            (REPO_ROOT / "data" / "ocr_outputs" / "5a0e008c-5ceb-456e-b618-04c607464f88.json")
            .read_text(encoding="utf-8")
        )
        combined_text = f"{data['header_text']}\n{data['full_page_text']}"
        self.document, self.persons = _run_pipeline(combined_text, data["header_text"])

    def test_document_fields(self):
        self.assertEqual(self.document.court_name.value, "محكمة بداية عمان")
        self.assertEqual(self.document.case_number.value, "1202/2026")
        self.assertEqual(self.document.document_number.value, "UW-2026-0102")
        self.assertEqual(self.document.document_date.value, "2026/07/11")

    def test_three_records_extracted(self):
        self.assertEqual(len(self.persons), 3)

    def test_company_extracted_despite_label_far_from_value(self):
        companies = _persons_by_type(self.persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].full_name, "شركة المسار للتقنيات المالية")
        self.assertEqual(companies[0].registration_number, "REG-202702")
        self.assertFalse(companies[0].needs_review)

    def test_cleanly_anchored_person_fully_recovered(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000004", by_id)
        self.assertEqual(by_id["98100000004"].full_name, "ناصر امجد سالم الرواشدة")
        self.assertFalse(by_id["98100000004"].needs_review)

    def test_severely_damaged_person_reconstructed_via_stitching(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000003", by_id)
        damaged = by_id["98100000003"]

        self.assertEqual(damaged.full_name, "عدي قراس محمود الزعبي")
        self.assertEqual(damaged.person_type, "Individual")
        self.assertFalse(damaged.needs_review)

        # Best-effort tier -- below the confident 0.90/0.95 bands.
        self.assertLess(damaged.confidence, 0.90)
        self.assertGreaterEqual(damaged.confidence, 0.70)

        # Never contaminated with unrelated junk ("فر" is a "فرد"
        # type-label leftover, "قم" is unrelated footer noise, "ل" isn't
        # even a real word) or words that belong to a different record.
        for junk_word in ("فر", "قم", "ل"):
            self.assertNotIn(junk_word, damaged.full_name.split())
        self.assertNotIn("شركة", damaged.full_name.split())
        self.assertNotIn("المسار", damaged.full_name.split())

    def test_field_level_accuracy_at_least_90_percent(self):
        validated_document, validated_persons = validate_all(self.document, list(self.persons))

        actual = {
            "document": {
                "court_name": {"value": validated_document.court_name.value},
                "case_number": {"value": validated_document.case_number.value},
                "document_number": {"value": validated_document.document_number.value},
                "document_date": {"value": validated_document.document_date.value},
            },
            "persons": [
                {
                    "full_name": p.full_name,
                    "national_id": p.national_id,
                    "registration_number": p.registration_number,
                    "person_type": p.person_type,
                }
                for p in validated_persons
            ],
        }

        expected_path = (
            Path(__file__).resolve().parent / "expected_outputs" / "sample_4_expected.json"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        result = evaluate_accuracy_multi(expected, actual)

        self.assertGreaterEqual(result["accuracy"], 0.90, msg=result["field_results"])


class TestRegistrationLabelFarFromValue(unittest.TestCase):
    def test_registration_number_extracted_with_company_name_between(self):
        # OCR reversed reading order so the company name sits between the
        # registration value and its own label on one line.
        line = "REG-202702 : شركة المسار للتقنيات المالية رقم التسجيل"
        text = (
            "الجهات المطلوبة :\n"
            "نوع الشخص : شركة\n"
            f"{line}\n"
        )
        persons = extract_person_candidates(text)
        companies = _persons_by_type(persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].registration_number, "REG-202702")
        self.assertEqual(companies[0].full_name, "شركة المسار للتقنيات المالية")


class TestFourthSuppliedDocumentUnconfirmedPair(unittest.TestCase):
    """Real OCR sample for case 1203/2026 / document UW-2026-0103
    (data/ocr_outputs/6afa6e2b-2c17-4001-95a5-a6aa1003327a.json).

    One person's record captures exactly 2 words ("حمل" then "وائل"),
    but each sits on its *own* single-word OCR line -- unlike the
    reliably-recoverable case where a 2-word capture comes from one
    multi-word line (see the second supplied document), there's no
    structural confirmation these two are the trailing (grandfather+
    family) pair rather than an interior pair with gaps on both sides.
    Two things must both hold:
    1. the two single-word lines get reordered ("وائل" before "حمل",
       restoring true father->grandfather order -- they were emitted
       reversed, the same left-to-right/right-to-left sorting artifact
       fixed elsewhere for multi-word-vs-single-word lines);
    2. front-filling both remaining slots is *not* attempted -- the
       orphan pool here starts with two look-alike/unrelated tokens
       ("المحمة", "سمي") before the real given/family names ("سليم",
       "الجبور"), so confidently prepending would silently produce a
       wrong name instead of a merely incomplete one.
    """

    def setUp(self):
        data = json.loads(
            (REPO_ROOT / "data" / "ocr_outputs" / "6afa6e2b-2c17-4001-95a5-a6aa1003327a.json")
            .read_text(encoding="utf-8")
        )
        combined_text = f"{data['header_text']}\n{data['full_page_text']}"
        self.document, self.persons = _run_pipeline(combined_text, data["header_text"])

    def test_document_fields(self):
        self.assertEqual(self.document.court_name.value, "محكمة بداية عمان")
        self.assertEqual(self.document.case_number.value, "1203/2026")
        self.assertEqual(self.document.document_number.value, "UW-2026-0103")
        self.assertEqual(self.document.document_date.value, "2026/07/12")

    def test_three_records_extracted(self):
        self.assertEqual(len(self.persons), 3)

    def test_cleanly_anchored_records_unaffected(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000006", by_id)
        self.assertEqual(by_id["98100000006"].full_name, "رائد نضال فؤاد البطاينة")
        self.assertFalse(by_id["98100000006"].needs_review)

        companies = _persons_by_type(self.persons, "Company")
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].full_name, "شركة المدار للخدمات الرقمية")
        self.assertEqual(companies[0].registration_number, "REG-202703")

    def test_unconfirmed_pair_reordered_but_not_guessed(self):
        by_id = {p.national_id: p for p in self.persons if p.national_id}
        self.assertIn("98100000005", by_id)
        unconfirmed = by_id["98100000005"]

        # Reordered to true reading order (father before grandfather),
        # but not front-filled with an unreliable guess.
        self.assertEqual(unconfirmed.full_name, "وائل حمل")
        self.assertTrue(unconfirmed.needs_review)
        self.assertEqual(unconfirmed.extraction_method, "structured_target_list_unconfirmed_pair")

        # Never contaminated with the look-alike/unrelated pool tokens
        # that sit ahead of the real given/family names.
        self.assertNotIn("سمي", unconfirmed.full_name.split())
        self.assertNotIn("المحمة", unconfirmed.full_name.split())


if __name__ == "__main__":
    unittest.main()
