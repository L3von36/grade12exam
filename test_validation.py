#!/usr/bin/env python3
"""
Tests for validation.py metric functions and topic-accuracy evaluation.

Runs under pytest (`pytest test_validation.py`) or standalone
(`python test_validation.py`). Replaces the old throwaway test_json.py, which
only checked that a hard-coded notebook path parsed as JSON.
"""

from topic_classifier import TopicTaxonomy
from validation import (
    ExamGroundTruth,
    character_error_rate,
    word_overlap_f1,
    topic_prf,
    evaluate_topics,
)


def test_character_error_rate():
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("x", "") == 1.0
    # one substitution out of three reference chars
    assert abs(character_error_rate("abx", "abc") - 1 / 3) < 1e-9


def test_word_overlap_f1():
    assert word_overlap_f1("the cat sat", "the cat sat") == 1.0
    assert word_overlap_f1("apple", "orange") == 0.0
    assert word_overlap_f1("", "anything") == 0.0
    # half overlap: pred {a,b}, ref {b,c} -> P=0.5, R=0.5, F1=0.5
    assert abs(word_overlap_f1("a b", "b c") - 0.5) < 1e-9


def test_topic_prf():
    assert topic_prf(["algebra"], ["algebra"]) == {
        'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert topic_prf([], []) == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert topic_prf(["algebra"], []) == {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    # predicted {algebra, geometry}, expected {algebra} -> P=0.5, R=1.0
    prf = topic_prf(["algebra", "geometry"], ["algebra"])
    assert prf['precision'] == 0.5 and prf['recall'] == 1.0
    assert abs(prf['f1'] - 2 / 3) < 1e-9


def _math_taxonomy() -> TopicTaxonomy:
    return TopicTaxonomy(subject="mathematics", topics={
        "algebra": {"equation": 1.0, "factor": 1.0},
        "calculus": {"derivative": 1.0, "integral": 1.0},
    })


def test_evaluate_topics_perfect():
    taxonomies = {"mathematics": _math_taxonomy()}
    gt = [ExamGroundTruth(
        exam_id="mathematics-2008-questions",
        subject="mathematics",
        year="2008",
        type="questions",
        expected_topics=["algebra", "calculus"],
        questions=[
            {"text": "Solve the equation x - 1 = 0."},
            {"text": "Find the derivative of f."},
        ],
    )]
    res = evaluate_topics(taxonomies, gt, min_score=0.5)
    assert res["mathematics-2008-questions"]["f1"] == 1.0
    assert res["mathematics-2008-questions"]["predicted_topics"] == ["algebra", "calculus"]
    assert res["OVERALL"]["n_exams"] == 1
    assert res["OVERALL"]["f1"] == 1.0


def test_evaluate_topics_skips_unknown_subject_and_handles_empty():
    # subject not in taxonomies -> skipped -> no OVERALL key
    taxonomies = {"mathematics": _math_taxonomy()}
    gt = [ExamGroundTruth(
        exam_id="biology-2008-questions",
        subject="biology",
        year="2008",
        type="questions",
        expected_topics=["genetics"],
        questions=[{"text": "Describe DNA replication."}],
    )]
    res = evaluate_topics(taxonomies, gt)
    assert res == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
