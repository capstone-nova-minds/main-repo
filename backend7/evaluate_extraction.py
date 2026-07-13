"""Evaluate extraction pipeline accuracy against human-reviewed ground truth.

Compares data/extracted/{document_id}.json (raw pipeline output) against
data/reviewed/{document_id}.json (the same document AFTER a human corrected
it via the Streamlit review screen). Only documents that exist in BOTH
folders are usable as ground truth -- a document that was never reviewed
tells us nothing about accuracy.

Usage:
    python evaluate_extraction.py
    python evaluate_extraction.py --extracted-dir data/extracted --reviewed-dir data/reviewed
    python evaluate_extraction.py --out data/eval_report.json

What it measures:
1. Document-level fields (court_name, case_number, document_number,
   document_date): exact match rate after normalization.
2. Person-level detection: precision / recall / F1 for finding the right
   people at all (matched primarily by national_id, falling back to
   fuzzy name matching for people with no ID).
3. Person-level field accuracy: for people the pipeline DID find, how
   often was full_name exactly right and national_id exactly right.
4. A single blended "overall accuracy" number so you can track progress
   toward a target (e.g. 95%) over time.

No network calls, no external dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCUMENT_FIELDS = ["court_name", "case_number", "document_number", "document_date"]

# Below this similarity ratio, two names are considered "different people"
# rather than an OCR/spacing variant of the same name.
NAME_MATCH_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(value: Optional[str]) -> str:
    """Collapse whitespace and strip so trivial formatting diffs don't
    count as mismatches. Does NOT fix Arabic OCR letter confusion -- those
    are real errors and should count against accuracy.
    """
    if value is None:
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [WARN] could not read {path}: {exc}")
        return None


def find_paired_documents(extracted_dir: Path, reviewed_dir: Path) -> List[str]:
    """Return document_ids present in both folders (i.e. actually reviewed)."""
    extracted_ids = {p.stem for p in extracted_dir.glob("*.json")}
    reviewed_ids = {p.stem for p in reviewed_dir.glob("*.json")}
    paired = sorted(extracted_ids & reviewed_ids)

    only_extracted = extracted_ids - reviewed_ids
    if only_extracted:
        print(
            f"  [INFO] {len(only_extracted)} document(s) have no human review yet "
            f"-- skipped (not usable as ground truth)."
        )

    return paired


# ---------------------------------------------------------------------------
# Document-field comparison
# ---------------------------------------------------------------------------

@dataclass
class FieldStats:
    correct: int = 0
    total: int = 0
    mismatches: List[Dict[str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


def compare_document_fields(
    extracted_doc: Dict[str, Any],
    reviewed_doc: Dict[str, Any],
    document_id: str,
    stats_by_field: Dict[str, FieldStats],
) -> None:
    for field_name in DOCUMENT_FIELDS:
        extracted_value = _normalize_text(
            (extracted_doc.get(field_name) or {}).get("value")
        )
        reviewed_value = _normalize_text(
            (reviewed_doc.get(field_name) or {}).get("value")
        )

        stats = stats_by_field.setdefault(field_name, FieldStats())
        stats.total += 1

        if extracted_value == reviewed_value:
            stats.correct += 1
        else:
            stats.mismatches.append(
                {
                    "document_id": document_id,
                    "extracted": extracted_value or "(empty)",
                    "reviewed": reviewed_value or "(empty)",
                }
            )


# ---------------------------------------------------------------------------
# Person-level comparison
# ---------------------------------------------------------------------------

@dataclass
class PersonMatchResult:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    name_correct: int = 0
    id_correct: int = 0
    type_correct: int = 0
    matched_pairs: int = 0
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    missed_people: List[Dict[str, Any]] = field(default_factory=list)
    hallucinated_people: List[Dict[str, Any]] = field(default_factory=list)


def _match_persons(
    extracted_persons: List[Dict[str, Any]],
    reviewed_persons: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy matching: national_id exact match first, then best-available
    fuzzy name match for anyone left over. Returns (matched_pairs,
    unmatched_extracted, unmatched_reviewed).
    """
    remaining_extracted = list(extracted_persons)
    remaining_reviewed = list(reviewed_persons)
    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    # Pass 1: exact national_id match.
    reviewed_by_id: Dict[str, Dict[str, Any]] = {
        p["national_id"]: p for p in remaining_reviewed if p.get("national_id")
    }
    still_unmatched_extracted = []

    for ext_person in remaining_extracted:
        ext_id = ext_person.get("national_id")
        if ext_id and ext_id in reviewed_by_id:
            rev_person = reviewed_by_id.pop(ext_id)
            matched_pairs.append((ext_person, rev_person))
            remaining_reviewed.remove(rev_person)
        else:
            still_unmatched_extracted.append(ext_person)

    remaining_extracted = still_unmatched_extracted

    # Pass 2: fuzzy name match among whoever's left (people with no ID,
    # or an ID that was wrong/dropped).
    still_unmatched_extracted = []

    for ext_person in remaining_extracted:
        best_match = None
        best_score = 0.0

        for rev_person in remaining_reviewed:
            score = _name_similarity(
                ext_person.get("full_name"), rev_person.get("full_name")
            )
            if score > best_score:
                best_score = score
                best_match = rev_person

        if best_match is not None and best_score >= NAME_MATCH_THRESHOLD:
            matched_pairs.append((ext_person, best_match))
            remaining_reviewed.remove(best_match)
        else:
            still_unmatched_extracted.append(ext_person)

    return matched_pairs, still_unmatched_extracted, remaining_reviewed


def compare_persons(
    extracted_persons: List[Dict[str, Any]],
    reviewed_persons: List[Dict[str, Any]],
    document_id: str,
    result: PersonMatchResult,
) -> None:
    matched_pairs, unmatched_extracted, unmatched_reviewed = _match_persons(
        extracted_persons, reviewed_persons
    )

    result.true_positives += len(matched_pairs)
    result.false_positives += len(unmatched_extracted)
    result.false_negatives += len(unmatched_reviewed)

    for ext_person in unmatched_extracted:
        result.hallucinated_people.append(
            {"document_id": document_id, "full_name": ext_person.get("full_name")}
        )

    for rev_person in unmatched_reviewed:
        result.missed_people.append(
            {"document_id": document_id, "full_name": rev_person.get("full_name")}
        )

    for ext_person, rev_person in matched_pairs:
        result.matched_pairs += 1

        name_ok = _normalize_text(ext_person.get("full_name")) == _normalize_text(
            rev_person.get("full_name")
        )
        id_ok = _normalize_text(ext_person.get("national_id")) == _normalize_text(
            rev_person.get("national_id")
        )
        type_ok = ext_person.get("person_type") == rev_person.get("person_type")

        result.name_correct += int(name_ok)
        result.id_correct += int(id_ok)
        result.type_correct += int(type_ok)

        if not (name_ok and id_ok and type_ok):
            result.mismatches.append(
                {
                    "document_id": document_id,
                    "extracted": {
                        "full_name": ext_person.get("full_name"),
                        "national_id": ext_person.get("national_id"),
                        "person_type": ext_person.get("person_type"),
                    },
                    "reviewed": {
                        "full_name": rev_person.get("full_name"),
                        "national_id": rev_person.get("national_id"),
                        "person_type": rev_person.get("person_type"),
                    },
                    "name_correct": name_ok,
                    "id_correct": id_ok,
                    "type_correct": type_ok,
                }
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def build_report(
    field_stats: Dict[str, FieldStats],
    person_result: PersonMatchResult,
    num_documents: int,
) -> Dict[str, Any]:
    precision = _safe_div(
        person_result.true_positives,
        person_result.true_positives + person_result.false_positives,
    )
    recall = _safe_div(
        person_result.true_positives,
        person_result.true_positives + person_result.false_negatives,
    )
    f1 = _safe_div(2 * precision * recall, precision + recall)

    name_accuracy = _safe_div(person_result.name_correct, person_result.matched_pairs)
    id_accuracy = _safe_div(person_result.id_correct, person_result.matched_pairs)
    type_accuracy = _safe_div(person_result.type_correct, person_result.matched_pairs)

    document_field_summary = {
        name: {
            "accuracy": round(stats.accuracy, 4),
            "correct": stats.correct,
            "total": stats.total,
        }
        for name, stats in field_stats.items()
    }

    avg_document_field_accuracy = _safe_div(
        sum(s.correct for s in field_stats.values()),
        sum(s.total for s in field_stats.values()),
    )

    # Blended "overall accuracy": average of document-field accuracy,
    # person detection F1, and person name accuracy (name + id both being
    # correct is what actually matters for downstream use of the data).
    overall_accuracy = round(
        (avg_document_field_accuracy + f1 + name_accuracy) / 3, 4
    )

    return {
        "documents_evaluated": num_documents,
        "overall_accuracy": overall_accuracy,
        "document_fields": document_field_summary,
        "document_fields_avg_accuracy": round(avg_document_field_accuracy, 4),
        "person_detection": {
            "true_positives": person_result.true_positives,
            "false_positives_hallucinated": person_result.false_positives,
            "false_negatives_missed": person_result.false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "person_field_accuracy_on_matched": {
            "full_name": round(name_accuracy, 4),
            "national_id": round(id_accuracy, 4),
            "person_type": round(type_accuracy, 4),
        },
        "examples": {
            "document_field_mismatches": [
                m for stats in field_stats.values() for m in stats.mismatches
            ][:20],
            "person_mismatches": person_result.mismatches[:20],
            "missed_people": person_result.missed_people[:20],
            "hallucinated_people": person_result.hallucinated_people[:20],
        },
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("EXTRACTION ACCURACY REPORT")
    print("=" * 70)
    print(f"Documents evaluated (had a human review): {report['documents_evaluated']}")
    print(f"\nOVERALL ACCURACY: {report['overall_accuracy'] * 100:.1f}%")
    print("(blend of document-field accuracy, person-detection F1, person-name accuracy)")

    print("\n--- Document-level fields ---")
    for name, stats in report["document_fields"].items():
        print(f"  {name:20s} {stats['accuracy']*100:5.1f}%  ({stats['correct']}/{stats['total']})")

    pd = report["person_detection"]
    print("\n--- Person detection ---")
    print(f"  True positives (found correctly):     {pd['true_positives']}")
    print(f"  False positives (hallucinated):        {pd['false_positives_hallucinated']}")
    print(f"  False negatives (missed entirely):     {pd['false_negatives_missed']}")
    print(f"  Precision: {pd['precision']*100:.1f}%   Recall: {pd['recall']*100:.1f}%   F1: {pd['f1']*100:.1f}%")

    pf = report["person_field_accuracy_on_matched"]
    print("\n--- Field accuracy (only on correctly-detected people) ---")
    print(f"  full_name matches reviewed exactly:  {pf['full_name']*100:.1f}%")
    print(f"  national_id matches reviewed exactly: {pf['national_id']*100:.1f}%")
    print(f"  person_type matches reviewed exactly: {pf['person_type']*100:.1f}%")

    if report["examples"]["document_field_mismatches"]:
        print("\n--- Sample document field mismatches (up to 20) ---")
        for m in report["examples"]["document_field_mismatches"]:
            print(f"  [{m['document_id']}] extracted='{m['extracted']}' reviewed='{m['reviewed']}'")

    if report["examples"]["person_mismatches"]:
        print("\n--- Sample person field mismatches (up to 20) ---")
        for m in report["examples"]["person_mismatches"]:
            print(f"  [{m['document_id']}]")
            print(f"    extracted: {m['extracted']}")
            print(f"    reviewed:  {m['reviewed']}")

    if report["examples"]["missed_people"]:
        print("\n--- People the pipeline missed entirely (up to 20) ---")
        for m in report["examples"]["missed_people"]:
            print(f"  [{m['document_id']}] {m['full_name']}")

    if report["examples"]["hallucinated_people"]:
        print("\n--- People the pipeline invented (false positives, up to 20) ---")
        for m in report["examples"]["hallucinated_people"]:
            print(f"  [{m['document_id']}] {m['full_name']}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(extracted_dir: Path, reviewed_dir: Path) -> Dict[str, Any]:
    print(f"Extracted dir: {extracted_dir}")
    print(f"Reviewed dir:  {reviewed_dir}")

    if not extracted_dir.exists():
        raise SystemExit(f"extracted-dir does not exist: {extracted_dir}")
    if not reviewed_dir.exists():
        raise SystemExit(f"reviewed-dir does not exist: {reviewed_dir}")

    document_ids = find_paired_documents(extracted_dir, reviewed_dir)

    if not document_ids:
        raise SystemExit(
            "No documents found in both folders. You need at least a few "
            "documents that went through the review screen (POST /review) "
            "before this script has anything to measure."
        )

    print(f"  [INFO] {len(document_ids)} reviewed document(s) will be used for evaluation.\n")

    field_stats: Dict[str, FieldStats] = {}
    person_result = PersonMatchResult()

    for document_id in document_ids:
        extracted = _load_json(extracted_dir / f"{document_id}.json")
        reviewed = _load_json(reviewed_dir / f"{document_id}.json")

        if extracted is None or reviewed is None:
            continue

        compare_document_fields(
            extracted.get("document", {}),
            reviewed.get("document", {}),
            document_id,
            field_stats,
        )

        compare_persons(
            extracted.get("persons", []) or [],
            reviewed.get("persons", []) or [],
            document_id,
            person_result,
        )

    return build_report(field_stats, person_result, len(document_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("data/extracted"),
        help="Folder with raw pipeline output JSON files (default: data/extracted)",
    )
    parser.add_argument(
        "--reviewed-dir",
        type=Path,
        default=Path("data/reviewed"),
        help="Folder with human-reviewed JSON files (default: data/reviewed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the full JSON report (e.g. data/eval_report.json)",
    )
    args = parser.parse_args()

    report = run_evaluation(args.extracted_dir, args.reviewed_dir)
    print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
