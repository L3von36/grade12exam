#!/usr/bin/env python3
"""
Topic classification with TF-IDF + hybrid taxonomy.

Hybrid approach: hand-curated TOPIC_RULES (in topics.py) provide topic seed
keywords; TF-IDF over the actual question corpus surfaces additional terms
that are statistically distinctive per topic. The classifier scores each
question against each topic using the union of seed + learned terms,
weighted by TF-IDF importance.

Why hybrid: pure keyword matching is brittle; pure unsupervised topic models
need more data than 6 years of exams provide. Seed keywords give us reliable
labels; TF-IDF augmentation catches synonyms/related vocabulary.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from topics import TOPIC_RULES


_TOKEN = re.compile(r"[a-z][a-z0-9'-]{1,}")
_STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'which', 'are', 'was',
    'were', 'have', 'has', 'had', 'not', 'but', 'into', 'than', 'then', 'them',
    'they', 'their', 'there', 'these', 'those', 'will', 'shall', 'can', 'could',
    'should', 'would', 'may', 'might', 'must', 'about', 'above', 'below', 'each',
    'one', 'two', 'three', 'all', 'any', 'some', 'such', 'also', 'only', 'most',
    'more', 'less', 'very', 'much', 'many', 'how', 'why', 'what', 'when', 'where',
    'who', 'whom', 'whose', 'because', 'been', 'being', 'does', 'did', 'doing',
    'following', 'figure', 'shown', 'given', 'find', 'show', 'find',
    'question', 'answer', 'choose', 'correct', 'best', 'true', 'false',
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _doc_freq(docs_tokens: List[List[str]]) -> Dict[str, int]:
    df: Dict[str, int] = defaultdict(int)
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] += 1
    return df


def compute_tfidf(docs: List[str]) -> List[Dict[str, float]]:
    """Return per-doc {term: tf-idf weight}."""
    docs_tokens = [tokenize(d) for d in docs]
    n_docs = max(len(docs_tokens), 1)
    df = _doc_freq(docs_tokens)
    out: List[Dict[str, float]] = []
    for tokens in docs_tokens:
        if not tokens:
            out.append({})
            continue
        tf = Counter(tokens)
        scores: Dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            scores[term] = (count / len(tokens)) * idf
        out.append(scores)
    return out


@dataclass
class TopicTaxonomy:
    """Per-subject mapping of topic -> weighted keyword set."""
    subject: str
    topics: Dict[str, Dict[str, float]]  # topic -> {term: weight}

    def to_dict(self) -> dict:
        return {'subject': self.subject, 'topics': self.topics}

    @classmethod
    def from_dict(cls, d: dict) -> 'TopicTaxonomy':
        return cls(subject=d['subject'], topics=d['topics'])


def build_taxonomy(subject: str,
                   question_texts: Iterable[str],
                   top_n_per_topic: int = 25,
                   seed_weight: float = 1.5) -> TopicTaxonomy:
    """
    Build a hybrid taxonomy for a subject:
    1. Seed each topic with TOPIC_RULES keywords (boosted weight).
    2. Compute TF-IDF over the subject's questions.
    3. For each topic, find terms whose TF-IDF mass is concentrated in
       documents already containing seed keywords; add the top N to the
       topic's keyword set.
    """
    seeds = TOPIC_RULES.get(subject, {})
    if not seeds:
        return TopicTaxonomy(subject=subject, topics={})

    docs = [d for d in question_texts if d and d.strip()]
    topics_out: Dict[str, Dict[str, float]] = {
        topic: {kw: seed_weight for kw in kws} for topic, kws in seeds.items()
    }

    if not docs:
        return TopicTaxonomy(subject=subject, topics=topics_out)

    tfidf_per_doc = compute_tfidf(docs)
    docs_lower = [d.lower() for d in docs]

    for topic, seed_kws in seeds.items():
        topic_term_mass: Dict[str, float] = defaultdict(float)
        seed_set = set(k.lower() for k in seed_kws)
        for doc_text, doc_scores in zip(docs_lower, tfidf_per_doc):
            if not any(s in doc_text for s in seed_set):
                continue
            for term, weight in doc_scores.items():
                if term in seed_set or len(term) < 4:
                    continue
                topic_term_mass[term] += weight

        ranked = sorted(topic_term_mass.items(), key=lambda kv: kv[1], reverse=True)
        for term, weight in ranked[:top_n_per_topic]:
            topics_out[topic][term] = round(float(weight), 4)

    return TopicTaxonomy(subject=subject, topics=topics_out)


def save_taxonomy(taxonomy: TopicTaxonomy, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy.to_dict(), f, indent=2, sort_keys=True)


def load_taxonomy(path: str) -> TopicTaxonomy:
    with open(path, 'r', encoding='utf-8') as f:
        return TopicTaxonomy.from_dict(json.load(f))


def score_text(text: str, taxonomy: TopicTaxonomy) -> Dict[str, float]:
    """Score a single text against every topic in the taxonomy."""
    text_l = text.lower()
    scores: Dict[str, float] = {}
    for topic, terms in taxonomy.topics.items():
        score = 0.0
        for term, weight in terms.items():
            if term in text_l:
                score += weight * text_l.count(term)
        scores[topic] = round(score, 3)
    return scores


def classify_questions(questions: List[str],
                       taxonomy: TopicTaxonomy,
                       min_score: float = 0.5) -> List[Optional[str]]:
    """Assign each question its best-matching topic, or None below threshold."""
    out: List[Optional[str]] = []
    for q in questions:
        scores = score_text(q, taxonomy)
        if not scores:
            out.append(None)
            continue
        best_topic, best_score = max(scores.items(), key=lambda kv: kv[1])
        out.append(best_topic if best_score >= min_score else None)
    return out
