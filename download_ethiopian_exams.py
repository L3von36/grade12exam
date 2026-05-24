#!/usr/bin/env python3
"""
Download Ethiopian Grade 12 exams from ethiobookreview.com.
Thin entry point that delegates to exam_downloader.py.
"""

import os

from config import DOWNLOAD_DIR
from exam_downloader import run_downloader


def main():
    print('=' * 60)
    print('Ethiopian Grade 12 Exam Downloader')
    print('=' * 60)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    report = run_downloader(DOWNLOAD_DIR)

    print('\nDOWNLOAD SUMMARY')
    print('-' * 40)
    total_ok = 0
    for subject, stats in sorted(report.items()):
        total_ok += stats['downloaded']
        print(f'  {subject}: {stats}')
    print(f'\nTotal new downloads: {total_ok}')
    print(f'Saved to: {DOWNLOAD_DIR}')


if __name__ == '__main__':
    main()
