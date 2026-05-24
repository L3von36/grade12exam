"""
Ethiopian Grade 12 Exam Study Platform.

Routes:
  GET  /                        → subject selection home
  GET  /subject/{subject}       → focus guide (chapter priority)
  GET  /practice/{subject}      → practice session
  GET  /api/topics              → all topic data (JSON)
  GET  /api/questions/{subject} → questions for a subject (JSON)
  POST /api/check               → check a submitted answer (JSON)
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

from fastapi import Cookie, FastAPI, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .chapter_data import SUBJECT_ICONS, TOPIC_META, get_meta
from .i18n import TEXT, t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(title="Ethiopian Grade 12 Study")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ─── Data loading ────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict | list:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_topics_cache: dict | None = None
_questions_cache: list | None = None


def get_topics() -> dict:
    global _topics_cache
    if _topics_cache is None:
        _topics_cache = _load_json("topics.json")
    return _topics_cache


def get_all_questions() -> list:
    global _questions_cache
    if _questions_cache is None:
        # Prefer real questions export from notebook; fall back to samples.
        data = _load_json("questions.json") or _load_json("sample_questions.json")
        _questions_cache = data.get("questions", []) if isinstance(data, dict) else []
    return _questions_cache


# ─── Helpers ─────────────────────────────────────────────────────────────────

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "SKIP": 3}
SUBJECTS = ["biology", "chemistry", "physics", "mathematics", "english", "civics"]


def _lang(request: Request, lang_cookie: Optional[str]) -> str:
    param = request.query_params.get("lang")
    if param in ("en", "am"):
        return param
    if lang_cookie in ("en", "am"):
        return lang_cookie
    return "en"


def _subject_summary(subject: str, topics_data: dict, lang: str) -> dict:
    subj = topics_data.get("subjects", {}).get(subject, {})
    topic_list = subj.get("topics", [])
    high_count = sum(1 for tp in topic_list if tp.get("priority") == "HIGH")
    total = len(topic_list)
    pct = round(high_count / total * 100) if total else 0
    top = topic_list[0] if topic_list else {}
    top_meta = get_meta(subject, top.get("topic", ""), lang)
    return {
        "subject": subject,
        "icon": SUBJECT_ICONS.get(subject, "📚"),
        "label": t(subject, lang),
        "n_topics": total,
        "high_count": high_count,
        "coverage_pct": pct,
        "top_topic": top_meta["label"],
        "hit_rate": subj.get("backtest_hit_rate", 0),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def home(
    request: Request,
    lang: Optional[str] = Query(None),
    lang_cookie: Optional[str] = Cookie(None),
):
    current_lang = _lang(request, lang_cookie)
    topics_data = get_topics()
    summaries = [_subject_summary(s, topics_data, current_lang) for s in SUBJECTS]
    all_q = get_all_questions()

    response = templates.TemplateResponse(
        request,
        "home.html",
        {
            "lang": current_lang,
            "texts": TEXT.get(current_lang, TEXT["en"]),
            "subjects": summaries,
            "target_year": topics_data.get("target_year", 2011),
            "total_questions": len(all_q),
        },
    )
    response.set_cookie("lang_cookie", current_lang, max_age=60 * 60 * 24 * 365)
    return response


@app.get("/subject/{subject}")
async def focus_guide(
    request: Request,
    subject: str,
    lang: Optional[str] = Query(None),
    lang_cookie: Optional[str] = Cookie(None),
):
    if subject not in SUBJECTS:
        return RedirectResponse("/")
    current_lang = _lang(request, lang_cookie)
    topics_data = get_topics()
    subj_data = topics_data.get("subjects", {}).get(subject, {})
    topic_list = subj_data.get("topics", [])

    all_q = get_all_questions()
    q_by_topic: dict[str, int] = {}
    for q in all_q:
        if q.get("subject") == subject:
            tp = q.get("topic", "")
            q_by_topic[tp] = q_by_topic.get(tp, 0) + 1

    enriched = []
    for tp in topic_list:
        key = tp["topic"]
        meta = get_meta(subject, key, current_lang)
        enriched.append({
            **tp,
            "label": meta["label"],
            "chapters": meta["chapters"],
            "n_questions": q_by_topic.get(key, 0),
        })

    high_topics = [tp for tp in enriched if tp["priority"] == "HIGH"]
    high_pct = round(len(high_topics) / len(enriched) * 100) if enriched else 0

    response = templates.TemplateResponse(
        request,
        "focus.html",
        {
            "lang": current_lang,
            "texts": TEXT.get(current_lang, TEXT["en"]),
            "subject": subject,
            "subject_label": t(subject, current_lang),
            "subject_icon": SUBJECT_ICONS.get(subject, "📚"),
            "topics": enriched,
            "high_pct": high_pct,
            "target_year": topics_data.get("target_year", 2011),
            "hit_rate": subj_data.get("backtest_hit_rate", 0),
            "rank_corr": subj_data.get("backtest_rank_correlation", 0),
        },
    )
    response.set_cookie("lang_cookie", current_lang, max_age=60 * 60 * 24 * 365)
    return response


@app.get("/practice/{subject}")
async def practice(
    request: Request,
    subject: str,
    topic: Optional[str] = Query(None),
    lang: Optional[str] = Query(None),
    lang_cookie: Optional[str] = Cookie(None),
):
    if subject not in SUBJECTS:
        return RedirectResponse("/")
    current_lang = _lang(request, lang_cookie)

    all_q = get_all_questions()
    qs = [q for q in all_q if q.get("subject") == subject]
    if topic and topic != "all":
        qs = [q for q in qs if q.get("topic") == topic]

    # Prioritise HIGH questions; shuffle within priority tiers.
    high = [q for q in qs if q.get("priority") == "HIGH"]
    med  = [q for q in qs if q.get("priority") == "MEDIUM"]
    low  = [q for q in qs if q.get("priority") in ("LOW", "SKIP")]
    random.shuffle(high); random.shuffle(med); random.shuffle(low)
    ordered = (high + med + low)[:30]

    topics_data = get_topics()
    topic_list = topics_data.get("subjects", {}).get(subject, {}).get("topics", [])
    topic_options = [
        {"key": tp["topic"], "label": get_meta(subject, tp["topic"], current_lang)["label"]}
        for tp in topic_list
    ]

    response = templates.TemplateResponse(
        request,
        "practice.html",
        {
            "lang": current_lang,
            "texts": TEXT.get(current_lang, TEXT["en"]),
            "subject": subject,
            "subject_label": t(subject, current_lang),
            "subject_icon": SUBJECT_ICONS.get(subject, "📚"),
            "questions": ordered,
            "selected_topic": topic or "all",
            "topic_options": topic_options,
            "total": len(ordered),
        },
    )
    response.set_cookie("lang_cookie", current_lang, max_age=60 * 60 * 24 * 365)
    return response


# ─── API ─────────────────────────────────────────────────────────────────────

@app.get("/api/topics")
async def api_topics():
    return JSONResponse(get_topics())


@app.get("/api/questions/{subject}")
async def api_questions(subject: str, topic: Optional[str] = Query(None)):
    qs = [q for q in get_all_questions() if q.get("subject") == subject]
    if topic and topic != "all":
        qs = [q for q in qs if q.get("topic") == topic]
    return JSONResponse({"questions": qs, "total": len(qs)})


@app.post("/api/check")
async def api_check(request: Request):
    body = await request.json()
    question_id = int(body.get("question_id", -1))
    chosen = str(body.get("answer", "")).upper()
    qs = get_all_questions()
    match = next((q for q in qs if q.get("id") == question_id), None)
    if not match:
        return JSONResponse({"error": "Question not found"}, status_code=404)
    correct_answer = match["correct_answer"].upper()
    return JSONResponse({
        "correct": chosen == correct_answer,
        "correct_answer": correct_answer,
        "chapter_ref": match.get("chapter_ref", ""),
    })


@app.get("/lang/{lang_code}")
async def set_lang(lang_code: str, redirect: str = "/"):
    if lang_code not in ("en", "am"):
        lang_code = "en"
    resp = RedirectResponse(redirect)
    resp.set_cookie("lang_cookie", lang_code, max_age=60 * 60 * 24 * 365)
    return resp
