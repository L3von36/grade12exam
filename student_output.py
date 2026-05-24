#!/usr/bin/env python3
"""
Generate student-facing study guides from predictions.

A study guide is a ranked list of topics for a subject's next exam:
- Topic name + predicted likelihood (with components breakdown)
- Historical frequency (which past exams had this topic)
- Textbook links (grades 11-12, relevant chapters)
- Confidence + explanation
- Estimated study hours
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TopicGuide:
    rank: int
    topic: str
    likely_score: float
    confidence: float
    frequency: str  # e.g., "5/6 exams"
    frequency_count: int
    trend_description: str
    cyclical_description: str
    textbook_links: List[Dict[str, str]]  # [{grade, subject, url}]
    exams_with_topic: List[str]  # ["mathematics-2009-questions", ...]
    study_hours: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StudyGuide:
    subject: str
    exam_year: int
    generated: str
    model_version: str = "3.0"
    backtest_hit_rate: Optional[float] = None
    backtest_rank_correlation: Optional[float] = None
    topics: List[Dict] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = []

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'exam_year': self.exam_year,
            'generated': self.generated,
            'model_version': self.model_version,
            'backtest_hit_rate': self.backtest_hit_rate,
            'backtest_rank_correlation': self.backtest_rank_correlation,
            'topics': self.topics,
        }


# Textbook links (curated; can be enhanced with OCR'd TOC later)
TEXTBOOK_LINKS = {
    'mathematics': [
        {'grade': '12', 'subject': 'mathematics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_MATH_Language_ENGLISH_Retrieved_20200601.pdf'},
        {'grade': '11', 'subject': 'mathematics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_MATH_Language_ENGLISH_Retrieved_20200601.pdf'},
    ],
    'physics': [
        {'grade': '12', 'subject': 'physics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_PHYSICS_Language_ENGLISH_Retrieved_20200601.pdf'},
        {'grade': '11', 'subject': 'physics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_PHYSICS_Language_ENGLISH_Retrieved_20200601.pdf'},
    ],
    'chemistry': [
        {'grade': '12', 'subject': 'chemistry', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_CHEMISTRY_Language_ENGLISH_Retrieved_20150101.pdf'},
        {'grade': '11', 'subject': 'chemistry', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_CHEMISTRY_Language_ENGLISH_Retrieved_20150101.pdf'},
    ],
    'biology': [
        {'grade': '12', 'subject': 'biology', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_BIOLOGY_CURRICULUM_Language_ENGLISH_Retrieved_20150101.pdf'},
        {'grade': '11', 'subject': 'biology', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_BIOLOGY_CURRICULUM_Language_ENGLISH_To_Grade_12_Retrieved_20150101.pdf'},
    ],
    'english': [
        {'grade': '12', 'subject': 'english', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_ENGLISH_Language_ENGLISH_Retrieved_20200601.pdf'},
        {'grade': '11', 'subject': 'english', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_ENGLISH_Language_ENGLISH_Retrieved_20150101.pdf'},
    ],
    'civics': [
        {'grade': '12', 'subject': 'civics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2012/Grade_12_Subject_CIVICS_Language_ENGLISH_Retrieved_20150101.pdf'},
        {'grade': '11', 'subject': 'civics', 'url': 'https://files.ethiopialearning.com/textbooks/Grade%2011/Grade_11_Subject_CIVICS_Language_ENGLISH_Retrieved_20150101.pdf'},
    ],
}


def _trend_str(slope: float) -> str:
    if abs(slope) < 0.1:
        return "stable (no clear trend)"
    direction = "increasing" if slope > 0 else "decreasing"
    return f"{direction} (slope={slope:.2f})"


def _cyclical_str(cyclical: float, target_year: int) -> str:
    if abs(cyclical) < 0.3:
        return "no cyclical pattern"
    parity = "odd" if target_year % 2 == 1 else "even"
    phase = "aligned" if cyclical > 0 else "misaligned"
    return f"{parity}-year topic, target is {parity} ({phase})"


def generate_study_guide(
    subject: str,
    exam_year: int,
    predictions_df,
    trend_df,
    backtest_result: Optional[Dict] = None,
    generated_at: str = "2026-05-01",
    exams_df=None,
) -> StudyGuide:
    """
    Generate a StudyGuide from predictions dataframe.

    predictions_df should have columns: subject, topic, likely_score,
      confidence, recent, trend, cyclical, stability, history (list of
      (year, score) tuples).
    trend_df: the full trend dataframe for frequency lookups (subject,
      year_num, topic, score).
    exams_df: optional dataframe with a 'filename' column for listing the
      specific exams that contained each topic.
    """
    pred_sub = predictions_df[predictions_df['subject'] == subject]
    if pred_sub.empty:
        return StudyGuide(subject=subject, exam_year=exam_year, generated=generated_at)

    topics: List[Dict] = []
    for rank, (_, row) in enumerate(pred_sub.iterrows(), start=1):
        topic = row['topic']

        # Frequency: count distinct exam years where this topic had a positive
        # score (trend_df rows may include zero-score entries).
        topic_rows = trend_df[
            (trend_df['subject'] == subject) & (trend_df['topic'] == topic)
        ]
        positive_years = topic_rows[topic_rows['score'] > 0]['year_num'].dropna().unique()
        n_exams = int(len(positive_years))
        frequency_str = f"{n_exams}/6 exams"

        # Exam references: pull filenames from exams_df where this subject
        # appeared in a year with positive topic score.
        exams: List[str] = []
        if exams_df is not None and 'filename' in exams_df.columns and len(positive_years) > 0:
            mask = (exams_df['subject'] == subject) & (exams_df['type'] == 'questions')
            years_str = {str(int(y)) for y in positive_years}
            mask &= exams_df['year'].astype(str).isin(years_str)
            exams = exams_df.loc[mask, 'filename'].tolist()

        # Trend & cyclical description
        trend_desc = _trend_str(float(row.get('trend', 0)))
        cyclical_desc = _cyclical_str(float(row.get('cyclical', 0)), exam_year)

        # Study hours: rough estimate from frequency + confidence
        hours = max(3, min(20, int(n_exams * 2 * float(row['confidence']))))

        topics.append({
            'rank': rank,
            'topic': topic,
            'likely_score': round(float(row['likely_score']), 4),
            'confidence': round(float(row['confidence']), 4),
            'frequency': frequency_str,
            'frequency_count': n_exams,
            'trend_description': trend_desc,
            'cyclical_description': cyclical_desc,
            'textbook_links': TEXTBOOK_LINKS.get(subject, []),
            'exams_with_topic': exams,
            'study_hours': hours,
        })

    guide = StudyGuide(
        subject=subject,
        exam_year=exam_year,
        generated=generated_at,
        topics=topics,
    )
    if backtest_result:
        guide.backtest_hit_rate = backtest_result.get('hit_rate')
        guide.backtest_rank_correlation = backtest_result.get('rank_correlation')
    return guide


def save_study_guide(guide: StudyGuide, output_dir: str) -> str:
    """Save guide as JSON and return path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{guide.subject}_{guide.exam_year}_study_guide.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(guide.to_dict(), f, indent=2, sort_keys=True)
    return path


def render_html(guide: StudyGuide) -> str:
    """Render a study guide as HTML (can be printed to PDF in any browser)."""
    rows = []
    for topic in guide.topics:
        rows.append(f"""
    <tr>
      <td>{topic['rank']}</td>
      <td><strong>{topic['topic']}</strong></td>
      <td>{topic['likely_score']:.3f}</td>
      <td>{topic['confidence']:.2f}</td>
      <td>{topic['frequency']}</td>
      <td>{topic['trend_description']}</td>
      <td>{topic['study_hours']} hrs</td>
    </tr>
""")

    textbook_section = ""
    if guide.topics and guide.topics[0].get('textbook_links'):
        links_html = " | ".join(
            f'<a href="{link["url"]}" target="_blank">{link["grade"]}</a>'
            for link in guide.topics[0]['textbook_links']
        )
        textbook_section = f"""
<h3>Textbooks (Grade 11 & 12)</h3>
<p>{links_html}</p>
"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{guide.subject.title()} Study Guide - Exam Year {guide.exam_year}</title>
  <style>
    @media print {{
      body {{ margin: 0.5in; }}
      table {{ page-break-inside: avoid; }}
    }}
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
    h1 {{ color: #333; }}
    h2 {{ color: #666; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
    th {{ background: #4CAF50; color: white; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    .meta {{ font-size: 0.9em; color: #666; margin: 10px 0; }}
    .print-note {{ background: #e3f2fd; padding: 10px; border-left: 4px solid #2196F3; margin: 20px 0; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>{guide.subject.upper()} Study Plan for Exam Year {guide.exam_year}</h1>
  <div class="meta">
    Generated: {guide.generated} | Model v{guide.model_version}
  </div>

  <div class="print-note">
    💡 <strong>Tip:</strong> Click <strong>Print</strong> (Ctrl+P or Cmd+P) and select "Save as PDF" to download a printable version.
  </div>

  {textbook_section}

  <h2>Top Topics by Likelihood</h2>
  <table>
    <tr>
      <th>Rank</th>
      <th>Topic</th>
      <th>Score</th>
      <th>Confidence</th>
      <th>History</th>
      <th>Trend</th>
      <th>Study</th>
    </tr>
    {''.join(rows)}
  </table>

  <h2>How to Use This Guide</h2>
  <ol>
    <li>Start with the highest-ranked topics (highest "Score" + "Confidence").</li>
    <li>Download the textbook for your grade (11 or 12).</li>
    <li>Focus study time on the chapters relevant to each topic.</li>
    <li>Practice past exam questions on these topics.</li>
    <li>If confidence is low (&lt;0.6), these predictions are less reliable—review broader content.</li>
  </ol>

  <h2>Confidence Explanation</h2>
  <p>
    Confidence (0–1) reflects the model's certainty about the prediction:
  </p>
  <ul>
    <li><strong>≥0.8</strong>: High confidence. Topic is likely to appear.</li>
    <li><strong>0.6–0.8</strong>: Medium confidence. Topic may appear, but not guaranteed.</li>
    <li><strong>&lt;0.6</strong>: Low confidence. Use as guidance only; the topic's appearance is uncertain.</li>
  </ul>
  <p>
    Confidence depends on data sufficiency (how many exams were analyzed) and signal agreement
    (whether recent trend, cyclical phase, and stability all point the same way).
  </p>
</body>
</html>
"""
    return html


def save_html_guide(guide: StudyGuide, html: str, output_dir: str) -> str:
    """Save HTML guide and return path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{guide.subject}_{guide.exam_year}_study_guide.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path
