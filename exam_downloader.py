#!/usr/bin/env python3
"""
Download Ethiopian Grade 12 exam PDFs from ethiobookreview.com.
Replaces the original download_ethiopian_exams.py with a reusable module.
"""

import os
import time
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

from config import (
    BASE_URL,
    DOWNLOAD_DIR,
    NATURAL_SCIENCE_SUBJECTS,
    TYPES,
    YEARS,
)


def download_file(url: str, filepath: str, max_retries: int = 3,
                  delay: float = 2.0) -> bool:
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, filepath)
            return True
        except Exception as e:
            print(f'  Attempt {attempt + 1} failed: {e}')
            time.sleep(delay)
    return False


def download_subject_year(subject: str, year: str,
                          target_dir: str) -> List[Tuple[str, str]]:
    os.makedirs(target_dir, exist_ok=True)
    results = []
    for file_type in TYPES:
        filename = f'{subject}-{year}-{file_type}.pdf'
        url = f'{BASE_URL}{filename}'
        filepath = os.path.join(target_dir, filename)

        if os.path.exists(filepath):
            results.append((filename, 'already_downloaded'))
            continue

        ok = download_file(url, filepath)
        results.append((filename, 'success' if ok else 'failed'))
        time.sleep(0.5)
    return results


def run_downloader(download_dir: str = DOWNLOAD_DIR) -> Dict[str, dict]:
    summary = defaultdict(lambda: {'downloaded': 0, 'failed': 0, 'already': 0})
    streams = {
        'natural_science': NATURAL_SCIENCE_SUBJECTS,
    }
    for stream, subjects in streams.items():
        for subject in subjects:
            subject_dir = os.path.join(download_dir, stream, subject)
            for year in YEARS:
                for _, status in download_subject_year(subject, year, subject_dir):
                    if status == 'success':
                        summary[subject]['downloaded'] += 1
                    elif status == 'failed':
                        summary[subject]['failed'] += 1
                    else:
                        summary[subject]['already'] += 1
    return dict(summary)


if __name__ == '__main__':
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    report = run_downloader()
    for subject, stats in sorted(report.items()):
        print(f'{subject}: {stats}')
