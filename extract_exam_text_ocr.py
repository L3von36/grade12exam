#!/usr/bin/env python3
"""
Extract text from Ethiopian Grade 12 exam PDFs using OCR.
Thin entry point that delegates to ocr_engine.py.

Default engine is 'auto': try EasyOCR first, fall back to Tesseract.
Override with the OCR_ENGINE env var ('easyocr' or 'tesseract').
"""

import csv
import os
import tempfile

from config import (
    DOWNLOAD_DIR,
    EXTRACTED_DIR,
    OUTPUT_CSV,
    stream_for_subject,
)
from ocr_engine import clean_text, extract_text_from_pdf


def parse_filename(filename: str):
    name = filename.replace('.pdf', '')
    parts = name.split('-')
    subject = parts[0] if len(parts) > 0 else 'unknown'
    year = parts[1] if len(parts) > 1 else 'unknown'
    ftype = parts[2] if len(parts) > 2 else 'unknown'
    return subject, year, ftype


def main():
    engine = os.environ.get('OCR_ENGINE', 'auto')
    print('=' * 60)
    print(f'Ethiopian Grade 12 Exam Text Extractor (engine={engine})')
    print('=' * 60 + '\n')

    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    pdf_paths = []
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in sorted(files):
            if f.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, f))

    print(f'Found {len(pdf_paths)} PDF files to process\n')

    all_records = []
    errors = 0
    temp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')

    for i, pdf_path in enumerate(pdf_paths, start=1):
        filename = os.path.basename(pdf_path)
        subject, year, ftype = parse_filename(filename)
        stream = stream_for_subject(subject)

        print(f'[{i}/{len(pdf_paths)}] {filename}...', end=' ', flush=True)

        try:
            ocr = extract_text_from_pdf(pdf_path, engine=engine, temp_dir=temp_dir)
            text = clean_text(ocr.full_text)

            subject_dir = os.path.join(EXTRACTED_DIR, stream, subject)
            os.makedirs(subject_dir, exist_ok=True)
            txt_path = os.path.join(subject_dir, filename.replace('.pdf', '.txt'))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            all_records.append({
                'subject': subject,
                'year': year,
                'type': ftype,
                'stream': stream,
                'filename': filename,
                'file_size_bytes': os.path.getsize(pdf_path),
                'text_length': len(text),
                'ocr_engine': ocr.engine,
                'ocr_confidence': round(ocr.mean_confidence, 3),
                'text_preview': text[:500],
                'text_full': text,
            })
            print(f'OK ({len(text):,} chars, conf={ocr.mean_confidence:.2f})')
        except Exception as e:
            print(f'ERROR: {e}')
            errors += 1

    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    print(f'\nProcessed: {len(all_records)} | Errors: {errors}')
    print(f'CSV: {OUTPUT_CSV}')
    print(f'Text: {EXTRACTED_DIR}')


if __name__ == '__main__':
    main()
