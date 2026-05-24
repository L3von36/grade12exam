#!/usr/bin/env python3
"""
Validation utilities: ground truth loading, OCR accuracy metrics,
topic-detection precision/recall.

Ground truth format (validation/ground_truth.json):
{
  "exams": [
    {
      "exam_id": "mathematics-2008-questions",
      "subject": "mathematics",
      "year": "2008",
      "type": "questions",
      "expected_topics": ["calculus", "algebra"],
      "questions": [
        {"page": 1, "id": "q1", "text": "Solve x^2 - 4 = 0"},
        ...
      ]
    }
  ]
}
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'validation',
    'ground_truth.json'
)


@dataclass
class ExamGroundTruth:
    exam_id: str
    subject: str
    year: str
    type: str
    expected_topics: List[str]
    questions: List[dict]


def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> List[ExamGroundTruth]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [ExamGroundTruth(**e) for e in data.get('exams', [])]


def character_error_rate(predicted: str, reference: str) -> float:
    """
    Levenshtein distance / len(reference). Lower is better.
    Pure Python implementation to avoid an editdistance dependency.
    """
    if not reference:
        return 0.0 if not predicted else 1.0

    a, b = predicted, reference
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            sub = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(insert, delete, sub)
        previous = current

    return previous[-1] / len(reference)


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def word_overlap_f1(predicted: str, reference: str) -> float:
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    intersect = pred_tokens & ref_tokens
    if not intersect:
        return 0.0
    precision = len(intersect) / len(pred_tokens)
    recall = len(intersect) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def topic_prf(predicted_topics: List[str],
              expected_topics: List[str]) -> Dict[str, float]:
    """Precision, recall, F1 for a predicted-vs-expected topic list."""
    pred = set(predicted_topics)
    ref = set(expected_topics)
    if not pred and not ref:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not pred or not ref:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_ocr(extracted_by_exam: Dict[str, str],
                 ground_truth: Optional[List[ExamGroundTruth]] = None
                 ) -> Dict[str, dict]:
    """
    Evaluate OCR output against ground truth.
    extracted_by_exam: {exam_id: full_extracted_text}
    Returns {exam_id: {cer, f1, n_reference_questions}}
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    results = {}
    for gt in ground_truth:
        if gt.exam_id not in extracted_by_exam:
            continue
        ref = ' '.join(q.get('text', '') for q in gt.questions)
        pred = extracted_by_exam[gt.exam_id]
        results[gt.exam_id] = {
            'cer': character_error_rate(pred, ref),
            'word_f1': word_overlap_f1(pred, ref),
            'n_reference_questions': len(gt.questions),
        }
    return results
