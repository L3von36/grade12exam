#!/usr/bin/env python3
"""
Download textbooks (old + new) using download_links.json.

Produces a report of downloaded files and missing URLs.
"""
import json
import os
import time
import urllib.request
import ssl


def load_links(path="download_links.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def download_file(url, out_path, max_retries=3):
    ssl._create_default_https_context = ssl._create_unverified_context
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, 'wb') as out:
                out.write(resp.read())
            return True, None
        except Exception as e:
            last = e
            time.sleep(1)
    return False, str(last)


def download_all(links_path='download_links.json', target_dir='textbooks'):
    links = load_links(links_path)
    os.makedirs(target_dir, exist_ok=True)
    report = []

    for grade, subjects in links.items():
        for subject, books in subjects.items():
            subj_dir = os.path.join(target_dir, grade, subject)
            os.makedirs(subj_dir, exist_ok=True)
            for book in books:
                url = book.get('download_url')
                curriculum = book.get('curriculum', 'unknown')
                btype = book.get('type', 'book')
                name = book.get('book_name', 'unknown').replace('/', '_')
                filename = f"{curriculum}_{btype}_{name}.pdf"
                out_path = os.path.join(subj_dir, filename)

                if not url:
                    report.append({'grade': grade, 'subject': subject, 'book': name, 'status': 'missing_url'})
                    continue

                if os.path.exists(out_path):
                    report.append({'grade': grade, 'subject': subject, 'book': name, 'status': 'already'})
                    continue

                ok, err = download_file(url, out_path)
                report.append({'grade': grade, 'subject': subject, 'book': name, 'status': 'downloaded' if ok else 'failed', 'error': err})

    return report


def summarize_report(report):
    from collections import Counter, defaultdict

    status_counts = Counter(r['status'] for r in report)
    by_subject = defaultdict(lambda: Counter())
    missing = defaultdict(list)
    failed = defaultdict(list)

    for r in report:
        by_subject[r['subject']][r['status']] += 1
        if r['status'] == 'missing_url':
            missing[r['subject']].append(r['book'])
        if r['status'] == 'failed':
            failed[r['subject']].append({'book': r['book'], 'error': r.get('error')})

    return {
        'status_counts': dict(status_counts),
        'subject_breakdown': {k: dict(v) for k, v in by_subject.items()},
        'missing_urls': dict(missing),
        'failed_downloads': dict(failed),
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--links', default='download_links.json')
    ap.add_argument('--out', default='textbooks')
    args = ap.parse_args()
    rep = download_all(args.links, args.out)
    import pprint
    pprint.pprint(rep)
