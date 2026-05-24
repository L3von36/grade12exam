#!/usr/bin/env python3
"""
Ensemble next-year topic prediction + leave-one-year-out backtesting.

Replaces the original weighted-average model. Combines four signals:
  - recent_avg:  mean of the last `recent_window` years
  - trend:       slope of a linear fit over recent years (normalized)
  - cyclical:    phase-aware lookback when an even/odd alternation is
                 statistically more consistent than a flat baseline
  - stability:   recent values down-weighted if the topic is erratic

All component scores are normalized into a 0..1 range per subject so the
final `likely_score` is comparable across subjects. A rough confidence is
derived from data sufficiency and signal agreement.

Backtesting holds out one year at a time, predicts that year from the
preceding years only, and reports per-subject hit rate (top-k overlap)
and rank correlation with the actual top-k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class TopicPrediction:
    subject: str
    topic: str
    likely_score: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    history: List[Tuple[float, float]] = field(default_factory=list)  # (year, score)

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'topic': self.topic,
            'likely_score': round(self.likely_score, 4),
            'confidence': round(self.confidence, 4),
            'components': {k: round(v, 4) for k, v in self.components.items()},
            'history': [(int(y), round(s, 4)) for y, s in self.history],
        }


# --- Pure-python signal helpers ----------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _detect_cyclical(years: Sequence[float], scores: Sequence[float]) -> float:
    """
    Returns a phase score in [-1, 1]:
      +1  → odd-year topic, target year is odd
      -1  → odd-year topic, target year is even (and vice versa)
       0  → no significant alternation
    Computed by comparing odd-year mean vs even-year mean against overall
    variance. Requires at least 4 data points.
    """
    if len(years) < 4:
        return 0.0
    odd_vals = [s for y, s in zip(years, scores) if int(y) % 2 == 1]
    even_vals = [s for y, s in zip(years, scores) if int(y) % 2 == 0]
    if len(odd_vals) < 2 or len(even_vals) < 2:
        return 0.0
    diff = _mean(odd_vals) - _mean(even_vals)
    pooled = _stdev(scores)
    if pooled <= 1e-9:
        return 0.0
    cohen_d = diff / pooled
    return max(-1.0, min(1.0, cohen_d / 0.8))


def _normalize(values: Iterable[float]) -> List[float]:
    vs = list(values)
    if not vs:
        return []
    lo, hi = min(vs), max(vs)
    span = hi - lo
    if span <= 1e-9:
        return [0.0 for _ in vs]
    return [(v - lo) / span for v in vs]


# --- Core model --------------------------------------------------------------

@dataclass
class EnsembleWeights:
    recent: float = 0.40
    trend: float = 0.20
    cyclical: float = 0.20
    stability: float = 0.20

    def items(self):
        return [('recent', self.recent), ('trend', self.trend),
                ('cyclical', self.cyclical), ('stability', self.stability)]


def _topic_components(years: List[float], scores: List[float],
                      target_year: float, recent_window: int = 3
                      ) -> Dict[str, float]:
    if not scores:
        return {'recent': 0.0, 'trend': 0.0, 'cyclical': 0.0, 'stability': 0.0}

    recent = scores[-recent_window:]
    recent_mean = _mean(recent)

    # Normalize slope by mean of all scores so trend isn't dominated by
    # absolute magnitudes that differ wildly between topics.
    overall_mean = _mean(scores) or 1.0
    raw_slope = _slope(years[-recent_window:], recent) if len(recent) >= 2 else 0.0
    trend = max(-1.0, min(1.0, raw_slope / overall_mean))

    cyclical_phase = _detect_cyclical(years, scores)
    target_parity = 1 if int(target_year) % 2 == 1 else -1
    cyclical = cyclical_phase * target_parity  # +1 if phase aligns

    sd = _stdev(scores)
    stability = 1.0 / (1.0 + sd / max(overall_mean, 1e-9))

    return {
        'recent': recent_mean,
        'trend': trend,
        'cyclical': cyclical,
        'stability': stability,
    }


def _confidence(history_len: int, components: Dict[str, float]) -> float:
    """Rough confidence in [0, 1]:
        - 0.6 base from data sufficiency (caps at 6 data points)
        - up to +0.3 for signal agreement (recent + trend + cyclical aligned)
    """
    data_factor = min(1.0, history_len / 6.0) * 0.6
    signs = [
        1 if components['recent'] > 0 else 0,
        1 if components['trend'] > 0 else (-1 if components['trend'] < 0 else 0),
        1 if components['cyclical'] > 0 else (-1 if components['cyclical'] < 0 else 0),
    ]
    nonzero = [s for s in signs if s != 0]
    agreement = (sum(nonzero) / len(nonzero)) if nonzero else 0.0
    agree_factor = max(0.0, agreement) * 0.3
    return round(data_factor + agree_factor, 3)


def _trend_subset(trend_df,
                  subject: Optional[str] = None,
                  max_year: Optional[float] = None):
    """Filter a trend dataframe by subject and/or upper-bound year."""
    df = trend_df
    if subject is not None:
        df = df[df['subject'] == subject]
    if max_year is not None:
        df = df[df['year_num'] < max_year]
    return df


def predict_subject(trend_df,
                    subject: str,
                    target_year: float,
                    weights: Optional[EnsembleWeights] = None,
                    recent_window: int = 3) -> List[TopicPrediction]:
    """Predict topic likely_scores for a subject's next exam year."""
    weights = weights or EnsembleWeights()
    df = _trend_subset(trend_df, subject=subject, max_year=target_year)

    raw_components: Dict[str, Dict[str, float]] = {}
    histories: Dict[str, List[Tuple[float, float]]] = {}
    for topic, g in df.groupby('topic'):
        g = g.sort_values('year_num').dropna(subset=['year_num'])
        if g.empty:
            continue
        years = g['year_num'].astype(float).tolist()
        scores = g['score'].astype(float).tolist()
        raw_components[topic] = _topic_components(
            years, scores, target_year, recent_window
        )
        histories[topic] = list(zip(years, scores))

    if not raw_components:
        return []

    # Normalize each component across topics so weights are comparable.
    topics_order = list(raw_components.keys())
    norm: Dict[str, List[float]] = {}
    for key in ('recent', 'trend', 'cyclical', 'stability'):
        norm[key] = _normalize(raw_components[t][key] for t in topics_order)

    out: List[TopicPrediction] = []
    for i, topic in enumerate(topics_order):
        comp = {k: norm[k][i] for k in norm}
        likely = sum(w * comp[k] for k, w in weights.items())
        out.append(TopicPrediction(
            subject=subject,
            topic=topic,
            likely_score=round(likely, 4),
            confidence=_confidence(len(histories[topic]), raw_components[topic]),
            components=comp,
            history=histories[topic],
        ))

    out.sort(key=lambda p: p.likely_score, reverse=True)
    return out


def predict_all(trend_df,
                target_year: float,
                weights: Optional[EnsembleWeights] = None,
                recent_window: int = 3) -> List[TopicPrediction]:
    out: List[TopicPrediction] = []
    for subject in sorted(trend_df['subject'].dropna().unique()):
        out.extend(predict_subject(trend_df, subject, target_year,
                                   weights, recent_window))
    return out


# --- Backtesting -------------------------------------------------------------

def _kendall_tau(a: List[str], b: List[str]) -> float:
    """Rank correlation between two equal-length topic orderings."""
    common = [t for t in a if t in b]
    if len(common) < 2:
        return 0.0
    rank_a = {t: i for i, t in enumerate(a)}
    rank_b = {t: i for i, t in enumerate(b)}
    n = len(common)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = rank_a[common[i]], rank_a[common[j]]
            bi, bj = rank_b[common[i]], rank_b[common[j]]
            if (ai - aj) * (bi - bj) > 0:
                concordant += 1
            elif (ai - aj) * (bi - bj) < 0:
                discordant += 1
    pairs = n * (n - 1) / 2
    return (concordant - discordant) / pairs if pairs else 0.0


@dataclass
class BacktestResult:
    subject: str
    held_out_year: int
    top_k: int
    predicted: List[str]
    actual: List[str]
    hit_rate: float
    rank_correlation: float

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'held_out_year': self.held_out_year,
            'top_k': self.top_k,
            'predicted': self.predicted,
            'actual': self.actual,
            'hit_rate': round(self.hit_rate, 3),
            'rank_correlation': round(self.rank_correlation, 3),
        }


def backtest_subject(trend_df, subject: str,
                     top_k: int = 5,
                     weights: Optional[EnsembleWeights] = None
                     ) -> List[BacktestResult]:
    """Leave-one-year-out: hold each year >= second-earliest, predict it."""
    df = trend_df[trend_df['subject'] == subject].dropna(subset=['year_num'])
    years = sorted(df['year_num'].astype(float).unique())
    if len(years) < 3:
        return []

    results: List[BacktestResult] = []
    for held in years[2:]:  # need >=2 years of training data
        preds = predict_subject(trend_df, subject, target_year=held,
                                weights=weights)
        if not preds:
            continue
        actual = (df[df['year_num'] == held]
                  .sort_values('score', ascending=False)['topic']
                  .tolist())
        if not actual:
            continue

        predicted_top = [p.topic for p in preds[:top_k]]
        actual_top = actual[:top_k]
        hit = (len(set(predicted_top) & set(actual_top)) / len(actual_top)
               if actual_top else 0.0)
        tau = _kendall_tau(predicted_top, actual_top)

        results.append(BacktestResult(
            subject=subject,
            held_out_year=int(held),
            top_k=top_k,
            predicted=predicted_top,
            actual=actual_top,
            hit_rate=hit,
            rank_correlation=tau,
        ))
    return results


def backtest_all(trend_df, top_k: int = 5,
                 weights: Optional[EnsembleWeights] = None
                 ) -> List[BacktestResult]:
    out: List[BacktestResult] = []
    for subject in sorted(trend_df['subject'].dropna().unique()):
        out.extend(backtest_subject(trend_df, subject, top_k, weights))
    return out


def summarize_backtest(results: List[BacktestResult]) -> Dict[str, dict]:
    """Mean hit rate & rank correlation per subject."""
    by_subject: Dict[str, List[BacktestResult]] = {}
    for r in results:
        by_subject.setdefault(r.subject, []).append(r)
    summary = {}
    for subject, rs in by_subject.items():
        summary[subject] = {
            'n_folds': len(rs),
            'mean_hit_rate': round(_mean([r.hit_rate for r in rs]), 3),
            'mean_rank_correlation': round(
                _mean([r.rank_correlation for r in rs]), 3),
        }
    return summary
