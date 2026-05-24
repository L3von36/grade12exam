#!/usr/bin/env python3
"""
Build the trend_df time series at finer-than-taxonomy granularity.

The original notebook builds trend_df from the 3-4 coarse topics in
topics.py. That makes predictions too vague to act on ("study algebra") and
makes the backtest degenerate (top_k >= number of topics => hit_rate is a
constant 1.0). This helper builds the same [subject, year_num, topic, score]
frame from *chapter-level* assignments (e.g. chapter_classifier output), where
`topic` is a textbook chapter / subtopic instead of a coarse bucket.

Usage in the notebook (sketch):

    from trend_builder import build_trend_df

    records = []
    for _, row in question_rows.iterrows():          # one row per question
        records.append({
            'subject':  row['subject'],
            'year_num': row['year_num'],
            'topic':    row['chapter'],               # from chapter_classifier
            'weight':   row.get('chapter_score', 1.0) # optional: sim / marks
        })
    trend_df = build_trend_df(records, normalize=True)

`weight` is optional (defaults to 1.0 = one question = one count). Pass the
classifier's cosine similarity or the question's marks to weight stronger
matches more heavily.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import pandas as pd


_COLUMNS = ['subject', 'year_num', 'topic', 'score']


def build_trend_df(question_records: Iterable[Mapping],
                   normalize: bool = True,
                   min_year_questions: int = 0) -> pd.DataFrame:
    """Aggregate per-question (subject, year, topic) records into trend_df.

    Args:
        question_records: iterable of mappings with keys 'subject',
            'year_num', 'topic', and optional 'weight' (default 1.0).
        normalize: if True, scale each (subject, year)'s scores to sum to 1 so
            years with different question counts are comparable (recommended;
            the prediction signals assume comparable magnitudes across years).
        min_year_questions: drop any (subject, year) whose total weight is
            below this threshold (filters near-empty OCR years).

    Returns:
        DataFrame with columns [subject, year_num, topic, score], one row per
        (subject, year_num, topic).
    """
    agg: dict = defaultdict(float)
    for r in question_records:
        subject = r.get('subject')
        year = r.get('year_num')
        topic = r.get('topic')
        if subject is None or year is None or topic is None:
            continue
        try:
            year = float(year)
        except (TypeError, ValueError):
            continue
        agg[(subject, year, topic)] += float(r.get('weight', 1.0))

    rows = [{'subject': s, 'year_num': y, 'topic': t, 'score': v}
            for (s, y, t), v in agg.items()]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    if df.empty:
        return df

    if min_year_questions > 0:
        year_totals = df.groupby(['subject', 'year_num'])['score'].transform('sum')
        df = df[year_totals >= min_year_questions].reset_index(drop=True)
        if df.empty:
            return df

    if normalize:
        totals = df.groupby(['subject', 'year_num'])['score'].transform('sum')
        df['score'] = df['score'] / totals.where(totals > 0, 1.0)

    return df.sort_values(['subject', 'year_num', 'topic']).reset_index(drop=True)
