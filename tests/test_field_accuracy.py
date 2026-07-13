"""Unit tests for calculate_field_accuracy -- the measured-accuracy metric
compared against a human review, as distinct from the Streamlit
"Extraction Quality Score" (completeness/confidence/review-flag based).

Run with:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.evaluation_service import calculate_field_accuracy  # noqa: E402


def _document(court_name, case_number, document_number, document_date):
    return {
        "court_name": {"value": court_name},
        "case_number": {"value": case_number},
        "document_number": {"value": document_number},
        "document_date": {"value": document_date},
    }


def _field_map(field_results):
    """field_results is a list; tests mostly want to look up by label."""
    return {item["field"]: item for item in field_results}


BASE_DOCUMENT = _document(
    "محكمة بداية عمان", "1201/2026", "UW-2026-0101", "2026/07/10"
)

TWO_INDIVIDUALS_ONE_COMPANY = [
    {
        "full_name": "يوسف سامر فؤاد العزام",
        "national_id": "98100000001",
        "registration_number": None,
        "person_type": "Individual",
        "record_index": 0,
    },
    {
        "full_name": "ليث محمود سليم الخطيب",
        "national_id": "98100000002",
        "registration_number": None,
        "person_type": "Individual",
        "record_index": 1,
    },
    {
        "full_name": "شركة الريادة للخدمات المالية",
        "national_id": None,
        "registration_number": "REG-202701",
        "person_type": "Company",
        "record_index": 2,
    },
]


class TestAllFieldsCorrect(unittest.TestCase):
    def test_all_13_fields_correct(self):
        # Test 1
        auto = {"document": BASE_DOCUMENT, "persons": TWO_INDIVIDUALS_ONE_COMPANY}
        reviewed = {"document": BASE_DOCUMENT, "persons": TWO_INDIVIDUALS_ONE_COMPANY}

        result = calculate_field_accuracy(auto, reviewed)

        self.assertEqual(result["total_fields"], 13)
        self.assertEqual(result["correct_fields"], 13)
        self.assertEqual(result["incorrect_fields"], 0)
        self.assertEqual(result["accuracy"], 100.0)


class TestWrongDocumentNumber(unittest.TestCase):
    def test_one_wrong_document_number(self):
        # Test 2
        auto = {
            "document": _document("محكمة بداية عمان", "1201/2026", "U-2026-0101", "2026/07/10"),
            "persons": TWO_INDIVIDUALS_ONE_COMPANY,
        }
        reviewed = {"document": BASE_DOCUMENT, "persons": TWO_INDIVIDUALS_ONE_COMPANY}

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertFalse(fields["Document Number"]["correct"])
        self.assertEqual(result["total_fields"], 13)
        self.assertEqual(result["correct_fields"], 12)
        self.assertEqual(result["accuracy"], 92.3)


class TestWrongIndividualName(unittest.TestCase):
    def test_one_wrong_individual_name_rest_correct(self):
        # Test 3
        auto_persons = [
            {**TWO_INDIVIDUALS_ONE_COMPANY[0], "full_name": "اسم خاطئ تماما"},
            TWO_INDIVIDUALS_ONE_COMPANY[1],
            TWO_INDIVIDUALS_ONE_COMPANY[2],
        ]
        auto = {"document": BASE_DOCUMENT, "persons": auto_persons}
        reviewed = {"document": BASE_DOCUMENT, "persons": TWO_INDIVIDUALS_ONE_COMPANY}

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertFalse(fields["Person 1 - Full Name"]["correct"])
        self.assertTrue(fields["Person 1 - National ID"]["correct"])
        self.assertTrue(fields["Person 1 - Person Type"]["correct"])
        self.assertEqual(result["correct_fields"], 12)
        self.assertEqual(result["total_fields"], 13)


class TestArabicNormalization(unittest.TestCase):
    def test_diacritics_and_hamza_variants_are_equal(self):
        # Test 4
        auto = {"document": _document("محكمة بداية عمان", "1/2026", "A-2026-0001", "2026/01/01"), "persons": []}
        reviewed = {
            "document": _document("مَحْكَمَةُ بداية عمّان", "1/2026", "A-2026-0001", "2026/01/01"),
            "persons": [],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertTrue(fields["Court Name"]["correct"])


class TestDateNormalization(unittest.TestCase):
    def test_slash_and_hyphen_dates_are_equal(self):
        # Test 5
        auto = {"document": _document("محكمة", "1/2026", "A-2026-0001", "10/07/2026"), "persons": []}
        reviewed = {"document": _document("محكمة", "1/2026", "A-2026-0001", "2026-07-10"), "persons": []}

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertTrue(fields["Document Date"]["correct"])


class TestRegistrationNormalization(unittest.TestCase):
    def test_spaced_and_hyphenated_registration_are_equal(self):
        # Test 6
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "شركة الريادة للخدمات المالية",
                    "national_id": None,
                    "registration_number": "REG - 202701",
                    "person_type": "Company",
                    "record_index": 0,
                }
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "شركة الريادة للخدمات المالية",
                    "national_id": None,
                    "registration_number": "REG-202701",
                    "person_type": "Company",
                    "record_index": 0,
                }
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertTrue(fields["Person 1 - Registration Number"]["correct"])


class TestArabicIndicNationalId(unittest.TestCase):
    def test_arabic_indic_digits_equal_ascii_digits(self):
        # Test 7
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "اسم",
                    "national_id": "٩٩٠٠٠٠٠٠١٠١",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                }
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "اسم",
                    "national_id": "99000000101",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                }
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertTrue(fields["Person 1 - National ID"]["correct"])


class TestCorrectedNationalId(unittest.TestCase):
    def test_corrected_national_id_is_incorrect(self):
        # Test 8
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "اسم",
                    "national_id": "99000000102",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                }
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "اسم",
                    "national_id": "99000000101",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                }
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertFalse(fields["Person 1 - National ID"]["correct"])


class TestRowAssociationPreservedAcrossCorrections(unittest.TestCase):
    def test_corrected_name_and_id_still_compares_to_correct_original_record(self):
        # Test 9: the reviewer corrects *both* the name and the national ID
        # of the first row -- national-ID matching alone could no longer
        # find its original record, so this only works if record_index is
        # actually being used to keep row association.
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "OLD NAME A",
                    "national_id": "111",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                },
                {
                    "full_name": "OLD NAME B",
                    "national_id": "222",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 1,
                },
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "NEW NAME A",
                    "national_id": "999",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                },
                {
                    "full_name": "OLD NAME B",
                    "national_id": "222",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 1,
                },
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        # Row 0 was compared against auto row 0 (auto_value == "OLD NAME A"),
        # not auto row 1 or a blank/missing record.
        self.assertEqual(fields["Person 1 - Full Name"]["auto_value"], "OLD NAME A")
        self.assertFalse(fields["Person 1 - Full Name"]["correct"])
        self.assertEqual(fields["Person 1 - National ID"]["auto_value"], "111")
        self.assertFalse(fields["Person 1 - National ID"]["correct"])

        # Row 1 was untouched and still compares correctly.
        self.assertTrue(fields["Person 2 - Full Name"]["correct"])
        self.assertTrue(fields["Person 2 - National ID"]["correct"])


class TestAddedReviewedPerson(unittest.TestCase):
    def test_added_person_fields_are_incorrect(self):
        # Test 10
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [TWO_INDIVIDUALS_ONE_COMPANY[0]],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                TWO_INDIVIDUALS_ONE_COMPANY[0],
                {
                    "full_name": "شخص جديد أضافه المراجع",
                    "national_id": "98199999999",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": None,
                },
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertFalse(fields["Person 2 - Full Name"]["correct"])
        self.assertFalse(fields["Person 2 - National ID"]["correct"])
        self.assertIsNone(fields["Person 2 - Full Name"]["auto_value"])


class TestRemovedFalsePositivePerson(unittest.TestCase):
    def test_removed_person_counts_as_incorrect(self):
        # Test 11
        auto = {
            "document": BASE_DOCUMENT,
            "persons": [
                TWO_INDIVIDUALS_ONE_COMPANY[0],
                {
                    "full_name": "حجز اموال محافظ الكترونية",
                    "national_id": None,
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 1,
                },
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [TWO_INDIVIDUALS_ONE_COMPANY[0]],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        removed_field = fields["Removed record (was Person 2) - Full Name"]
        self.assertFalse(removed_field["correct"])
        self.assertEqual(removed_field["auto_value"], "حجز اموال محافظ الكترونية")
        self.assertIsNone(removed_field["reviewed_value"])

        # The false-positive record's fields count toward the total.
        self.assertEqual(result["total_fields"], 4 + 3 + 3)


class TestIndividualFieldSet(unittest.TestCase):
    def test_registration_number_not_evaluated_for_individual(self):
        # Test 12
        auto = {"document": BASE_DOCUMENT, "persons": [TWO_INDIVIDUALS_ONE_COMPANY[0]]}
        reviewed = {"document": BASE_DOCUMENT, "persons": [TWO_INDIVIDUALS_ONE_COMPANY[0]]}

        result = calculate_field_accuracy(auto, reviewed)
        labels = [item["field"] for item in result["field_results"]]

        self.assertNotIn("Person 1 - Registration Number", labels)
        self.assertIn("Person 1 - National ID", labels)


class TestCompanyFieldSet(unittest.TestCase):
    def test_national_id_not_evaluated_for_company(self):
        # Test 13
        auto = {"document": BASE_DOCUMENT, "persons": [TWO_INDIVIDUALS_ONE_COMPANY[2]]}
        reviewed = {"document": BASE_DOCUMENT, "persons": [TWO_INDIVIDUALS_ONE_COMPANY[2]]}

        result = calculate_field_accuracy(auto, reviewed)
        labels = [item["field"] for item in result["field_results"]]

        self.assertNotIn("Person 1 - National ID", labels)
        self.assertIn("Person 1 - Registration Number", labels)


class TestAccuracyIndependentOfConfidence(unittest.TestCase):
    def test_quality_score_fields_do_not_affect_measured_accuracy(self):
        # Test 14
        low_confidence_person = {
            **TWO_INDIVIDUALS_ONE_COMPANY[0],
            "confidence": 0.05,
            "needs_review": True,
        }
        high_confidence_person = {
            **TWO_INDIVIDUALS_ONE_COMPANY[0],
            "confidence": 0.99,
            "needs_review": False,
        }

        auto = {"document": BASE_DOCUMENT, "persons": [low_confidence_person]}
        reviewed = {"document": BASE_DOCUMENT, "persons": [high_confidence_person]}

        result = calculate_field_accuracy(auto, reviewed)

        # Same underlying field values -> 100%, regardless of how far apart
        # the confidence/needs_review flags are.
        self.assertEqual(result["accuracy"], 100.0)

        labels = [item["field"] for item in result["field_results"]]
        self.assertNotIn("confidence", labels)
        self.assertNotIn("needs_review", labels)


class TestIntegrationScenario(unittest.TestCase):
    """The task's full worked example: a document number missing its "W",
    one individual with a wrong name *and* a missing National ID, one
    individual entirely correct, and a company name with a trailing junk
    word cleaned up during review."""

    def test_measured_accuracy_reflects_real_corrections_not_100_percent(self):
        auto = {
            "document": _document(
                "محكمة بداية عمان", "1201/2026", "U-2026-0101", "2026/07/10"
            ),
            "persons": [
                {
                    "full_name": "يوسف الخطيب ليث محمود",
                    "national_id": None,
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                },
                {
                    "full_name": "ليث محمود سليم الخطيب",
                    "national_id": "98100000002",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 1,
                },
                {
                    "full_name": "شركة الريادة للخدمات المالية ام",
                    "national_id": None,
                    "registration_number": "REG-202701",
                    "person_type": "Company",
                    "record_index": 2,
                },
            ],
        }
        reviewed = {
            "document": BASE_DOCUMENT,
            "persons": [
                {
                    "full_name": "يوسف سامر فؤاد العزام",
                    "national_id": "98100000001",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 0,
                },
                {
                    "full_name": "ليث محمود سليم الخطيب",
                    "national_id": "98100000002",
                    "registration_number": None,
                    "person_type": "Individual",
                    "record_index": 1,
                },
                {
                    "full_name": "شركة الريادة للخدمات المالية",
                    "national_id": None,
                    "registration_number": "REG-202701",
                    "person_type": "Company",
                    "record_index": 2,
                },
            ],
        }

        result = calculate_field_accuracy(auto, reviewed)
        fields = _field_map(result["field_results"])

        self.assertFalse(fields["Document Number"]["correct"])
        self.assertFalse(fields["Person 1 - Full Name"]["correct"])
        self.assertFalse(fields["Person 1 - National ID"]["correct"])
        self.assertTrue(fields["Person 1 - Person Type"]["correct"])
        self.assertTrue(fields["Person 2 - Full Name"]["correct"])
        self.assertTrue(fields["Person 2 - National ID"]["correct"])
        self.assertFalse(fields["Person 3 - Full Name"]["correct"])
        self.assertTrue(fields["Person 3 - Registration Number"]["correct"])

        self.assertEqual(result["total_fields"], 13)
        self.assertEqual(result["correct_fields"], 9)
        self.assertEqual(result["incorrect_fields"], 4)
        self.assertNotEqual(result["accuracy"], 100.0)
        self.assertEqual(result["accuracy"], 69.2)


if __name__ == "__main__":
    unittest.main()
