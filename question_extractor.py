#!/usr/bin/env python3
"""
Parse OCR'd exam text into individual structured questions.

Strategy:
- Split on question-number markers ("1.", "2)", "12.", "Q3.", etc.).
- For each question chunk, peel off MCQ option markers (A. / B) / C: / D/)
  into a separate options dict.
- Drop chunks too short to be real questions.

Heuristic, not perfect: OCR noise loses some questions. Aim is 80%+ recall
on cleanly-scanned exams; quality is measured against ground_truth.json.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# "1." | "1)" | "12)" | "Q1." | "(15)"
_QUESTION_HEAD = re.compile(
    r'(?:(?<=\s)|^)\(?\s*(?:Q(?:uestion)?\s*)?(\d{1,3})\s*[\)\.\:]\s*',
    re.IGNORECASE,
)

# A. | B) | C: | D/  (single option marker, not part of a word)
_OPTION_HEAD = re.compile(
    r'(?:(?<=\s)|^)\(?([A-Da-d])\s*[\)\.\:/]\s*'
)

_NOISE_PATTERN = re.compile(r'[_~`|\\]+')
_WHITESPACE = re.compile(r'\s+')


@dataclass
class Question:
    number: int
    text: str
    options: Dict[str, str] = field(default_factory=dict)
    raw: str = ''

    def to_dict(self) -> dict:
        return {
            'number': self.number,
            'text': self.text,
            'options': self.options,
            'raw': self.raw,
        }


def _normalize(text: str) -> str:
    text = _NOISE_PATTERN.sub(' ', text)
    return _WHITESPACE.sub(' ', text).strip()


def _split_options(body: str) -> tuple[str, Dict[str, str]]:
    """Split a question body into stem text and {A,B,C,D} options."""
    matches = list(_OPTION_HEAD.finditer(body))
    if len(matches) < 2:
        return _normalize(body), {}

    stem = _normalize(body[:matches[0].start()])
    options: Dict[str, str] = {}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        options[letter] = _normalize(body[start:end])
    return stem, options


def extract_questions(text: str,
                      min_stem_chars: int = 12,
                      max_questions: Optional[int] = None) -> List[Question]:
    """Parse a full exam's OCR text into Question objects."""
    if not text:
        return []

    text = _NOISE_PATTERN.sub(' ', text)
    matches = list(_QUESTION_HEAD.finditer(text))
    if not matches:
        return []

    questions: List[Question] = []
    for i, m in enumerate(matches):
        try:
            number = int(m.group(1))
        except (ValueError, TypeError):
            continue
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        stem, options = _split_options(body)
        if len(stem) < min_stem_chars:
            continue

        questions.append(Question(
            number=number,
            text=stem,
            options=options,
            raw=text[m.start():body_end].strip(),
        ))
        if max_questions and len(questions) >= max_questions:
            break

    return questions


def extract_question_texts(text: str) -> List[str]:
    """Convenience: just the question stems (for downstream TF-IDF)."""
    return [q.text for q in extract_questions(text)]
