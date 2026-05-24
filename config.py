#!/usr/bin/env python3
"""
Shared configuration for the Ethiopian Grade 12 exam analysis project.
Paths default to a local layout but can be overridden with env vars
(useful in Colab: set PROJECT_ROOT=/content/ethiopian_exam_project).
"""

import os


PROJECT_ROOT = os.environ.get(
    'PROJECT_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
)

DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'downloads')
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, 'extracted_text')
TEXTBOOK_DIR = os.path.join(PROJECT_ROOT, 'textbooks')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'ethiopian_exams_extracted.csv')
PATTERN_CSV = os.path.join(PROJECT_ROOT, 'pattern_summary.csv')
STUDY_GUIDE_DIR = os.path.join(PROJECT_ROOT, 'study_guides')

BASE_URL = 'https://www.ethiobookreview.com/assets/exams/'

NATURAL_SCIENCE_SUBJECTS = [
    'english', 'mathematics', 'physics', 'chemistry', 'biology', 'civics'
]

YEARS = ['2005', '2006', '2007', '2008', '2009', '2010']
TYPES = ['questions', 'answers']


def stream_for_subject(subject: str) -> str:
    if subject in NATURAL_SCIENCE_SUBJECTS:
        return 'natural_science'
    return 'unknown'


def ensure_dirs() -> None:
    for d in (DOWNLOAD_DIR, EXTRACTED_DIR, TEXTBOOK_DIR, STUDY_GUIDE_DIR):
        os.makedirs(d, exist_ok=True)
