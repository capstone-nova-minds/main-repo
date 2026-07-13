"""Unit tests for utils/json_utils.make_json_serializable.

Run with:  python -m unittest discover -s tests
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from utils.json_utils import make_json_serializable  # noqa: E402


class TestMakeJsonSerializable(unittest.TestCase):
    def test_numpy_bool_becomes_python_bool(self):
        result = make_json_serializable(np.bool_(True))
        self.assertIs(result, True)

    def test_numpy_integer_becomes_python_int(self):
        result = make_json_serializable(np.int64(3))
        self.assertEqual(result, 3)
        self.assertIsInstance(result, int)

    def test_numpy_floating_becomes_python_float(self):
        result = make_json_serializable(np.float32(0.85))
        self.assertIsInstance(result, float)

    def test_numpy_array_becomes_list(self):
        result = make_json_serializable(np.array([1, 2, 3]))
        self.assertEqual(result, [1, 2, 3])

    def test_path_becomes_string(self):
        result = make_json_serializable(Path("data") / "uploads" / "file.pdf")
        self.assertIsInstance(result, str)

    def test_none_remains_none(self):
        self.assertIsNone(make_json_serializable(None))

    def test_plain_types_unchanged(self):
        self.assertEqual(make_json_serializable("text"), "text")
        self.assertEqual(make_json_serializable(5), 5)
        self.assertEqual(make_json_serializable(1.5), 1.5)
        self.assertEqual(make_json_serializable(True), True)

    def test_ocr_like_payload_is_fully_json_serializable(self):
        """Mirrors the shape that caused the original TypeError: a dict with
        nested numpy scalars/arrays inside a list, as in ocr_result["pages"].
        """
        payload = {
            "pages": [
                {
                    "needs_review": np.bool_(True),
                    "confidence": np.float32(0.85),
                    "count": np.int64(3),
                    "items": np.array([1, 2, 3]),
                }
            ]
        }

        safe_payload = make_json_serializable(payload)

        # Should not raise TypeError: Object of type bool is not JSON serializable
        serialized = json.dumps(safe_payload)
        self.assertIn('"needs_review": true', serialized)

        page = safe_payload["pages"][0]
        self.assertIs(page["needs_review"], True)
        self.assertIsInstance(page["confidence"], float)
        self.assertIsInstance(page["count"], int)
        self.assertEqual(page["items"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
