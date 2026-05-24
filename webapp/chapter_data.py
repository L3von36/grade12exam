"""Topic → textbook chapter mapping with bilingual labels."""
from __future__ import annotations
from typing import Dict, List

TOPIC_META: Dict[str, Dict[str, dict]] = {
    "biology": {
        "cell_biology": {
            "label": "Cell Biology",
            "label_am": "የሕዋስ ባዮሎጂ",
            "chapters": "Grade 11 Ch.1–2 · Grade 12 Ch.1",
            "chapters_am": "11ኛ ክፍል ምዕ.1–2 · 12ኛ ክፍል ምዕ.1",
        },
        "genetics": {
            "label": "Genetics",
            "label_am": "ዘረ-ሐሊዮ",
            "chapters": "Grade 11 Ch.5 · Grade 12 Ch.2–3",
            "chapters_am": "11ኛ ክፍል ምዕ.5 · 12ኛ ክፍል ምዕ.2–3",
        },
        "ecology": {
            "label": "Ecology",
            "label_am": "ሥነ-ምህዳር",
            "chapters": "Grade 11 Ch.6 · Grade 12 Ch.5",
            "chapters_am": "11ኛ ክፍል ምዕ.6 · 12ኛ ክፍል ምዕ.5",
        },
        "human_biology": {
            "label": "Human Biology",
            "label_am": "የሰው ባዮሎጂ",
            "chapters": "Grade 11 Ch.3–4 · Grade 12 Ch.4",
            "chapters_am": "11ኛ ክፍል ምዕ.3–4 · 12ኛ ክፍል ምዕ.4",
        },
    },
    "chemistry": {
        "acid_base": {
            "label": "Acids, Bases & pH",
            "label_am": "አሲድ፣ ቤዝ እና pH",
            "chapters": "Grade 11 Ch.6–7 · Grade 12 Ch.3",
            "chapters_am": "11ኛ ክፍል ምዕ.6–7 · 12ኛ ክፍል ምዕ.3",
        },
        "stoichiometry": {
            "label": "Stoichiometry",
            "label_am": "ስቶኪዮሜትሪ",
            "chapters": "Grade 11 Ch.1–2 · Grade 12 Ch.1",
            "chapters_am": "11ኛ ክፍል ምዕ.1–2 · 12ኛ ክፍል ምዕ.1",
        },
        "electrochemistry": {
            "label": "Electrochemistry",
            "label_am": "ኤሌክትሮኬሚስትሪ",
            "chapters": "Grade 12 Ch.5",
            "chapters_am": "12ኛ ክፍል ምዕ.5",
        },
        "organic": {
            "label": "Organic Chemistry",
            "label_am": "ኦርጋኒክ ኬሚስትሪ",
            "chapters": "Grade 11 Ch.8 · Grade 12 Ch.7",
            "chapters_am": "11ኛ ክፍል ምዕ.8 · 12ኛ ክፍል ምዕ.7",
        },
    },
    "physics": {
        "mechanics": {
            "label": "Mechanics",
            "label_am": "ሜካኒክስ",
            "chapters": "Grade 11 Ch.1–3 · Grade 12 Ch.1–2",
            "chapters_am": "11ኛ ክፍል ምዕ.1–3 · 12ኛ ክፍል ምዕ.1–2",
        },
        "electricity_magnetism": {
            "label": "Electricity & Magnetism",
            "label_am": "ኤሌክትሪክና መግነጢስ",
            "chapters": "Grade 11 Ch.4–5 · Grade 12 Ch.3–4",
            "chapters_am": "11ኛ ክፍል ምዕ.4–5 · 12ኛ ክፍል ምዕ.3–4",
        },
        "waves_optics": {
            "label": "Waves & Optics",
            "label_am": "ሞገድና ኦፕቲክስ",
            "chapters": "Grade 11 Ch.6 · Grade 12 Ch.5",
            "chapters_am": "11ኛ ክፍል ምዕ.6 · 12ኛ ክፍል ምዕ.5",
        },
        "thermodynamics": {
            "label": "Thermodynamics",
            "label_am": "ቴርሞዳይናሚክስ",
            "chapters": "Grade 11 Ch.7 · Grade 12 Ch.6",
            "chapters_am": "11ኛ ክፍል ምዕ.7 · 12ኛ ክፍል ምዕ.6",
        },
    },
    "mathematics": {
        "algebra": {
            "label": "Algebra & Functions",
            "label_am": "አልጀብራ እና ፈንክሽን",
            "chapters": "Grade 11 Ch.1–3 · Grade 12 Ch.1–2",
            "chapters_am": "11ኛ ክፍል ምዕ.1–3 · 12ኛ ክፍል ምዕ.1–2",
        },
        "geometry": {
            "label": "Geometry",
            "label_am": "ጂኦሜትሪ",
            "chapters": "Grade 11 Ch.4–5 · Grade 12 Ch.3",
            "chapters_am": "11ኛ ክፍል ምዕ.4–5 · 12ኛ ክፍል ምዕ.3",
        },
        "calculus": {
            "label": "Calculus",
            "label_am": "ካልኩለስ",
            "chapters": "Grade 12 Ch.4–5",
            "chapters_am": "12ኛ ክፍል ምዕ.4–5",
        },
        "statistics_probability": {
            "label": "Statistics & Probability",
            "label_am": "ስታቲስቲክስ እና ዕድል",
            "chapters": "Grade 11 Ch.8 · Grade 12 Ch.6",
            "chapters_am": "11ኛ ክፍል ምዕ.8 · 12ኛ ክፍል ምዕ.6",
        },
    },
    "english": {
        "vocabulary": {
            "label": "Vocabulary",
            "label_am": "የቃላት ፍቺ",
            "chapters": "Grade 11–12 Vocabulary Units",
            "chapters_am": "11ኛ–12ኛ ክፍል የቃላት ክፍሎች",
        },
        "reading": {
            "label": "Reading Comprehension",
            "label_am": "ንባብ ና ምስጢር",
            "chapters": "Grade 11–12 Reading Passages",
            "chapters_am": "11ኛ–12ኛ ክፍል የንባብ ምንባቦች",
        },
        "grammar": {
            "label": "Grammar",
            "label_am": "ሰዋሰው",
            "chapters": "Grade 11–12 Grammar Units",
            "chapters_am": "11ኛ–12ኛ ክፍል የሰዋሰው ክፍሎች",
        },
    },
    "civics": {
        "constitution_governance": {
            "label": "Constitution & Governance",
            "label_am": "ሕገ-መንግሥትና አስተዳደር",
            "chapters": "Grade 11 Ch.1–3 · Grade 12 Ch.1–3",
            "chapters_am": "11ኛ ክፍል ምዕ.1–3 · 12ኛ ክፍል ምዕ.1–3",
        },
        "rights_duties": {
            "label": "Rights & Civic Duties",
            "label_am": "መብትና ግዴታ",
            "chapters": "Grade 11 Ch.4–6 · Grade 12 Ch.4–6",
            "chapters_am": "11ኛ ክፍል ምዕ.4–6 · 12ኛ ክፍል ምዕ.4–6",
        },
        "economy_society": {
            "label": "Economy & Society",
            "label_am": "ኢኮኖሚና ማህበረሰብ",
            "chapters": "Grade 12 Ch.7–9",
            "chapters_am": "12ኛ ክፍል ምዕ.7–9",
        },
    },
}

SUBJECT_ICONS = {
    "biology": "🧬",
    "chemistry": "⚗️",
    "physics": "⚡",
    "mathematics": "📐",
    "english": "📖",
    "civics": "🏛️",
}


def get_meta(subject: str, topic: str, lang: str = "en") -> dict:
    m = TOPIC_META.get(subject, {}).get(topic, {})
    return {
        "label": m.get("label_am" if lang == "am" else "label", topic),
        "chapters": m.get("chapters_am" if lang == "am" else "chapters", "—"),
    }
