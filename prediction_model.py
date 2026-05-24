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


# Minimum data before the even/odd alternation signal is trusted. With only
# 6 years the odd/even split is 3-vs-3, where Cohen's d is dominated by noise;
# requiring >=8 years (>=4 per parity) keeps this signal from contributing
# spurious phase scores. Below the threshold _detect_cyclical returns 0.
_MIN_CYCLE_YEARS = 8
_MIN_PER_PARITY = 4

# Default top-k used by backtest helpers; clamped per subject by _effective_k.
_DEFAULT_TOP_K = 5


def _detect_cyclical(years: Sequence[float], scores: Sequence[float]) -> float:
    """
    Returns a phase score in [-1, 1]:
      +1  → odd-year topic, target year is odd
      -1  → odd-year topic, target year is even (and vice versa)
       0  → no significant alternation (or too little data to tell)
    Computed by comparing odd-year mean vs even-year mean against overall
    variance. Gated on _MIN_CYCLE_YEARS / _MIN_PER_PARITY so it stays silent
    until there is enough history for the effect size to mean anything.
    """
    if len(years) < _MIN_CYCLE_YEARS:
        return 0.0
    odd_vals = [s for y, s in zip(years, scores) if int(y) % 2 == 1]
    even_vals = [s for y, s in zip(years, scores) if int(y) % 2 == 0]
    if len(odd_vals) < _MIN_PER_PARITY or len(even_vals) < _MIN_PER_PARITY:
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


# --- Ranking functions: model + naive baselines ----------------------------
# Each takes (trend_df, subject, target_year) and returns topics ranked
# most->least likely. Baselines exist so the ensemble's hit-rate / rank
# correlation can be judged against trivial strategies: if it doesn't beat
# "repeat last year" or "use historical frequency", the extra signals aren't
# earning their weight.

def _model_ranking(trend_df, subject, target_year,
                   weights: Optional[EnsembleWeights] = None,
                   recent_window: int = 3) -> List[str]:
    preds = predict_subject(trend_df, subject, target_year, weights,
                            recent_window)
    return [p.topic for p in preds]


def frequency_ranking(trend_df, subject, target_year) -> List[str]:
    """Baseline: rank by mean historical score over all prior years."""
    df = _trend_subset(trend_df, subject=subject, max_year=target_year)
    if df.empty:
        return []
    agg = df.groupby('topic')['score'].mean().sort_values(ascending=False)
    return list(agg.index)


def recency_ranking(trend_df, subject, target_year) -> List[str]:
    """Baseline: rank by the single most recent prior year's scores."""
    df = _trend_subset(trend_df, subject=subject, max_year=target_year)
    if df.empty:
        return []
    last_year = df['year_num'].max()
    last = df[df['year_num'] == last_year].sort_values('score', ascending=False)
    return list(last['topic'])


def _evaluate_fold(predicted: List[str], actual: List[str], k: int
                   ) -> Tuple[List[str], List[str], float, float]:
    predicted_top = predicted[:k]
    actual_top = actual[:k]
    hit = (len(set(predicted_top) & set(actual_top)) / len(actual_top)
           if actual_top else 0.0)
    tau = _kendall_tau(predicted_top, actual_top)
    return predicted_top, actual_top, hit, tau


def _effective_k(top_k: int, n_topics: int) -> int:
    """Clamp k below the topic count.

    With k >= n_topics the predicted and actual top-k are both "all topics",
    so hit_rate is a meaningless constant 1.0. Capping at n_topics-1 keeps the
    metric discriminative. The real fix is more topics (chapter-level), but
    this prevents a silently degenerate score in the meantime.
    """
    return max(1, min(top_k, n_topics - 1))


def _backtest_ranking(trend_df, subject: str, rank_fn,
                      top_k: int = 5) -> List[BacktestResult]:
    """Leave-one-year-out backtest for an arbitrary ranking function."""
    df = trend_df[trend_df['subject'] == subject].dropna(subset=['year_num'])
    years = sorted(df['year_num'].astype(float).unique())
    if len(years) < 3:
        return []
    k = _effective_k(top_k, int(df['topic'].nunique()))

    results: List[BacktestResult] = []
    for held in years[2:]:  # need >=2 years of training data
        predicted = rank_fn(trend_df, subject, held)
        if not predicted:
            continue
        actual = (df[df['year_num'] == held]
                  .sort_values('score', ascending=False)['topic']
                  .tolist())
        if not actual:
            continue
        predicted_top, actual_top, hit, tau = _evaluate_fold(
            predicted, actual, k)
        results.append(BacktestResult(
            subject=subject,
            held_out_year=int(held),
            top_k=k,
            predicted=predicted_top,
            actual=actual_top,
            hit_rate=hit,
            rank_correlation=tau,
        ))
    return results


def backtest_subject(trend_df, subject: str,
                     top_k: int = 5,
                     weights: Optional[EnsembleWeights] = None
                     ) -> List[BacktestResult]:
    """Leave-one-year-out backtest of the ensemble model for one subject."""
    return _backtest_ranking(
        trend_df, subject,
        lambda td, s, y: _model_ranking(td, s, y, weights),
        top_k=top_k,
    )


def compare_baselines(trend_df, top_k: int = 5,
                      weights: Optional[EnsembleWeights] = None
                      ) -> Dict[str, dict]:
    """Run the ensemble and naive baselines through identical LOYO folds.

    Returns {method: {n_folds, mean_hit_rate, mean_rank_correlation}} for
    'ensemble', 'frequency', and 'recency'. The ensemble is only worth its
    complexity if it beats the baselines here.
    """
    rankers = {
        'ensemble': lambda td, s, y: _model_ranking(td, s, y, weights),
        'frequency': frequency_ranking,
        'recency': recency_ranking,
    }
    subjects = sorted(trend_df['subject'].dropna().unique())
    summary: Dict[str, dict] = {}
    for name, fn in rankers.items():
        res: List[BacktestResult] = []
        for subject in subjects:
            res.extend(_backtest_ranking(trend_df, subject, fn, top_k=top_k))
        summary[name] = {
            'n_folds': len(res),
            'mean_hit_rate': round(_mean([r.hit_rate for r in res]), 3),
            'mean_rank_correlation': round(
                _mean([r.rank_correlation for r in res]), 3),
        }
    return summary


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


# --- Data prep ---------------------------------------------------------------

def normalize_trend_scores(trend_df):
    """Convert raw per-(subject, year) topic scores into shares summing to 1.

    The notebook builds `score` by summing score_text over every question in an
    exam, so a year with more (or longer) questions inflates every topic's
    score and years aren't comparable. Dividing by the per-(subject, year)
    total turns each exam into a topic *distribution*, which is what the
    recent/trend signals should be reading. Call this on trend_df before
    predict_all / backtest_all.
    """
    df = trend_df.copy()
    totals = df.groupby(['subject', 'year_num'])['score'].transform('sum')
    df['score'] = df['score'] / totals.where(totals > 0, 1.0)
    return df


# --- Weight tuning -----------------------------------------------------------

def tune_weights(trend_df, top_k: int = 5, step: float = 0.1,
                 metric: str = 'rank_correlation'
                 ) -> Tuple[EnsembleWeights, float]:
    """Coarse grid search over ensemble weights, scored by backtest.

    Returns (best_weights, best_score). `metric` is 'rank_correlation' or
    'hit_rate'. Weights are searched on a `step` grid and constrained to sum
    to 1.0.

    CAUTION: with only ~6 years per subject this WILL overfit if taken too
    seriously. Keep `step` coarse (0.1+), pool across subjects (this does),
    and treat the result as a sanity check that the chosen weights beat the
    uniform 0.25 split — not as finely tuned truth.
    """
    n = int(round(1.0 / step))
    grid = [round(i * step, 4) for i in range(n + 1)]
    best: Optional[Tuple[EnsembleWeights, float]] = None
    for r in grid:
        for t in grid:
            for c in grid:
                s = round(1.0 - r - t - c, 4)
                if s < -1e-9 or s > 1.0 + 1e-9:
                    continue
                s = max(0.0, s)
                w = EnsembleWeights(recent=r, trend=t, cyclical=c, stability=s)
                res = backtest_all(trend_df, top_k=top_k, weights=w)
                if not res:
                    continue
                score = _mean([getattr(x, metric) for x in res])
                if best is None or score > best[1]:
                    best = (w, round(score, 4))
    if best is None:
        return EnsembleWeights(), 0.0
    return best


# --- Confidence calibration --------------------------------------------------

def confidence_calibration(trend_df,
                           weights: Optional[EnsembleWeights] = None,
                           recent_window: int = 3,
                           bins: Sequence[float] = (0.0, 0.6, 0.8, 1.01)
                           ) -> Dict[str, dict]:
    """Check whether stated confidence tracks real per-topic hit rate.

    Across all LOYO folds, for each predicted top-k topic record
    (confidence, hit) where hit==1 iff that topic actually appeared
    (score > 0) in the held-out year. Bucket by the confidence bins and
    report the empirical hit rate per bucket. A well-calibrated model shows
    higher empirical hit rate in higher-confidence buckets; if not, the
    confidence shown to students is misleading.
    """
    records: List[Tuple[float, int]] = []
    for subject in sorted(trend_df['subject'].dropna().unique()):
        df = trend_df[trend_df['subject'] == subject].dropna(subset=['year_num'])
        years = sorted(df['year_num'].astype(float).unique())
        if len(years) < 3:
            continue
        k = _effective_k(_DEFAULT_TOP_K, int(df['topic'].nunique()))
        for held in years[2:]:
            preds = predict_subject(trend_df, subject, held, weights,
                                    recent_window)
            if not preds:
                continue
            actual = set(df[(df['year_num'] == held) & (df['score'] > 0)]['topic'])
            for p in preds[:k]:
                records.append((p.confidence, 1 if p.topic in actual else 0))

    out: Dict[str, dict] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        grp = [hit for conf, hit in records if lo <= conf < hi]
        out[f'{lo:.2f}-{hi:.2f}'] = {
            'n': len(grp),
            'empirical_hit_rate': round(_mean(grp), 3) if grp else None,
        }
    return out
