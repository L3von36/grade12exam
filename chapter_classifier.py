#!/usr/bin/env python3
"""
Index textbook PDFs (split into chapters when possible) and classify questions
to textbook chapters/sections using TF-IDF + cosine similarity.

Functions:
 - index_textbooks(textbook_dir)
 - classify_questions(questions, index)
 - report_percentages(assignments)

This is a lightweight approach: chapter splitting uses 'Chapter' headings when
present; otherwise falls back to coarse fixed-size splits.
"""
from typing import List, Dict, Tuple
import os
import re
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from ocr_engine import extract_text_from_pdf, clean_text

CHAPTER_RE = re.compile(
    r'(^|\n)\s*(?:Chapter|CHAPTER|chapter)\s+(\d+|[IVXLCDM]+)(?:\.|:|\s)([^\n]+)',
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    r'(^|\n)\s*(?:Section|SECTION|section)\s+(\d+[\.\d]*)(?:\.|:|\s)([^\n]+)',
    re.IGNORECASE,
)
TOC_RE = re.compile(r'Table of Contents|Contents', re.IGNORECASE)


def normalize_title(title: str) -> str:
    if not title:
        return 'Unknown'
    title = title.strip()
    title = re.sub(r'\s+', ' ', title)
    return title


def extract_chapter_headings(text: str) -> List[Tuple[int, str]]:
    headings = []
    for m in CHAPTER_RE.finditer(text):
        headings.append((m.start(), normalize_title(m.group(0).strip())))
    if not headings:
        for m in SECTION_RE.finditer(text):
            headings.append((m.start(), normalize_title(m.group(0).strip())))
    return headings


def extract_text_quick(pdf_path: str) -> str:
    """Try fast text extraction via PyPDF2; fall back to OCR if empty."""
    text = ''
    if PyPDF2 is not None:
        try:
            with open(pdf_path, 'rb') as fh:
                reader = PyPDF2.PdfReader(fh)
                parts = []
                for p in reader.pages:
                    try:
                        t = p.extract_text() or ''
                    except Exception:
                        t = ''
                    parts.append(t)
                text = '\n\n'.join(parts)
        except Exception:
            text = ''

    if not text or len(text) < 200:
        # fallback to OCR (slower but robust)
        res = extract_text_from_pdf(pdf_path, engine='auto')
        text = res.full_text

    return clean_text(text)


def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    """Return list of (chapter_title, chapter_text).

    If no explicit chapters found, split into 6 coarse sections.
    """
    if not text:
        return []

    headings = extract_chapter_headings(text)
    if not headings:
        # Try to detect a table of contents section and use its numbered headings.
        toc_start = TOC_RE.search(text)
        if toc_start:
            possible_headings = SECTION_RE.finditer(text[toc_start.end():])
            headings = [(toc_start.end() + m.start(), normalize_title(m.group(0).strip())) for m in possible_headings]

    if not headings:
        approx = 6
        words = text.split()
        chunk = max(500, len(words) // approx)
        out = []
        for i in range(0, len(words), chunk):
            title = f'Section {i//chunk + 1}'
            out.append((title, ' '.join(words[i:i+chunk])))
        return out

    boundaries = [pos for pos, _ in headings] + [len(text)]
    out = []
    for i, (start, title) in enumerate(headings):
        end = boundaries[i+1]
        chunk = text[start:end].strip()
        out.append((normalize_title(title), chunk))
    return out


def report_curriculum_split(assignments: List[Dict]) -> Dict[str, float]:
    counts = Counter(a.get('curriculum', 'unknown') for a in assignments)
    total = max(len(assignments), 1)
    return {k: round(v / total * 100, 2) for k, v in counts.items()}


def report_grade_level_split(assignments: List[Dict]) -> Dict[str, float]:
    counts = Counter(a.get('grade_subject', 'unknown').split('|')[0] for a in assignments)
    total = max(len(assignments), 1)
    return {k: round(v / total * 100, 2) for k, v in counts.items()}


def estimate_difficulty(question: str) -> str:
    text = question.strip().lower()
    length = len(text.split())
    hard_keywords = {'prove', 'derive', 'calculate', 'show', 'evaluate', 'justify', 'explain', 'compare'}
    medium_keywords = {'find', 'determine', 'define', 'describe', 'identify', 'match'}
    score = 0
    if length >= 35:
        score += 1
    if length >= 50:
        score += 1
    if any(tok in text for tok in hard_keywords):
        score += 2
    elif any(tok in text for tok in medium_keywords):
        score += 1
    if score >= 3:
        return 'hard'
    if score == 2:
        return 'medium'
    return 'easy'


def index_textbooks(textbook_dir: str = 'textbooks') -> Dict[str, List[Dict]]:
    """Index PDFs under textbook_dir.

    Returns mapping grade/subject -> list of {pdf_path, curriculum, chapters: [(title,text)]}
    """
    index = defaultdict(list)
    for grade in os.listdir(textbook_dir) if os.path.isdir(textbook_dir) else []:
        grade_dir = os.path.join(textbook_dir, grade)
        for subject in os.listdir(grade_dir):
            subj_dir = os.path.join(grade_dir, subject)
            for fname in os.listdir(subj_dir):
                if not fname.lower().endswith('.pdf'):
                    continue
                pdf_path = os.path.join(subj_dir, fname)
                curriculum = 'unknown'
                if fname.lower().startswith('new'):
                    curriculum = 'new'
                elif fname.lower().startswith('old'):
                    curriculum = 'old'

                text = extract_text_quick(pdf_path)
                chapters = split_into_chapters(text)
                index[f'{grade}|{subject}'].append({'pdf': pdf_path, 'curriculum': curriculum, 'chapters': chapters})

    return index


def classify_questions(questions: List[str], index: Dict) -> List[Dict]:
    """Classify each question to the best matching chapter across the indexed textbooks.

    Returns list of assignments: {question, grade_subject, pdf, curriculum, chapter_title, score}
    """
    # Build a flat list of all chapters with identifiers
    chapter_texts = []
    chapter_meta = []
    for key, pdfs in index.items():
        for p in pdfs:
            for title, text in p['chapters']:
                chapter_meta.append({'grade_subject': key, 'pdf': p['pdf'], 'curriculum': p['curriculum'], 'chapter': title})
                chapter_texts.append(text)

    if not chapter_texts:
        return []

    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
    doc_vectors = vectorizer.fit_transform(chapter_texts)

    qvecs = vectorizer.transform(questions)
    sims = cosine_similarity(qvecs, doc_vectors)

    assignments = []
    for qi, q in enumerate(questions):
        row = sims[qi]
        best = row.argmax()
        score = float(row[best])
        meta = chapter_meta[best]
        assignments.append({'question': q, 'grade_subject': meta['grade_subject'], 'pdf': meta['pdf'], 'curriculum': meta['curriculum'], 'chapter': meta['chapter'], 'score': score})

    return assignments


def report_percentages(assignments: List[Dict]) -> Dict[str, float]:
    """Return percentage of questions per chapter (grade_subject|chapter).
    """
    counts = Counter()
    total = len(assignments)
    for a in assignments:
        key = f"{a['grade_subject']}|{a['chapter']}"
        counts[key] += 1

    pct = {k: round(v / total * 100, 2) for k, v in counts.items()} if total else {}
    return pct


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--textbooks', default='textbooks')
    ap.add_argument('--questions', help='JSON file with list of question strings')
    ap.add_argument('--out', help='output JSON for assignments')
    args = ap.parse_args()

    idx = index_textbooks(args.textbooks)
    questions = []
    if args.questions:
        with open(args.questions, 'r', encoding='utf-8') as f:
            questions = json.load(f)

    assigns = classify_questions(questions, idx)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(assigns, f, ensure_ascii=False, indent=2)
    else:
        import pprint
        pprint.pprint(assigns[:20])
        print('\nPercentages per chapter:')
        pprint.pprint(report_percentages(assigns))
