#!/usr/bin/env python3
"""
Topic taxonomy for Ethiopian Grade 12 exam analysis.

TOPIC_RULES is the hand-curated baseline: subject -> {topic: [seed keywords]}.
The TF-IDF classifier in topic_classifier.py augments these seed keywords
with terms learned from the actual exam corpus.
"""

from typing import Dict, List


TOPIC_RULES: Dict[str, Dict[str, List[str]]] = {
    'mathematics': {
        'algebra': ['equation', 'polynomial', 'factor', 'inequality', 'function'],
        'geometry': ['triangle', 'circle', 'angle', 'area', 'volume', 'geometry'],
        'calculus': ['derivative', 'integral', 'limit', 'differentiation'],
        'statistics_probability': ['probability', 'mean', 'median', 'variance', 'distribution'],
    },
    'physics': {
        'mechanics': ['force', 'motion', 'velocity', 'acceleration', 'momentum', 'energy'],
        'electricity_magnetism': ['current', 'voltage', 'resistance', 'electric', 'magnetic'],
        'waves_optics': ['wave', 'frequency', 'wavelength', 'light', 'lens', 'mirror'],
        'thermodynamics': ['heat', 'temperature', 'entropy', 'gas'],
    },
    'chemistry': {
        'stoichiometry': ['mole', 'molar', 'stoichiometry', 'equation', 'reactant'],
        'acid_base': ['acid', 'base', 'ph', 'buffer', 'neutralization'],
        'organic': ['hydrocarbon', 'alkane', 'alkene', 'organic', 'polymer'],
        'electrochemistry': ['electrode', 'oxidation', 'reduction', 'electrolysis'],
    },
    'biology': {
        'genetics': ['gene', 'dna', 'rna', 'inheritance', 'chromosome', 'allele'],
        'ecology': ['ecosystem', 'population', 'food chain', 'environment'],
        'cell_biology': ['cell', 'organelle', 'membrane', 'mitosis', 'meiosis'],
        'human_biology': ['respiratory', 'circulatory', 'digestive', 'hormone'],
    },
    'english': {
        'grammar': ['tense', 'verb', 'noun', 'adjective', 'preposition',
                    'sentence', 'clause', 'subject', 'object', 'passive',
                    'active', 'pronoun', 'adverb', 'conjunction', 'article',
                    'singular', 'plural', 'phrase', 'modal'],
        'reading': ['passage', 'main idea', 'inference', 'author',
                    'paragraph', 'comprehension', 'context', 'implies'],
        'vocabulary': ['synonym', 'antonym', 'meaning', 'word',
                       'similar', 'opposite', 'definition', 'closest'],
    },
    'civics': {
        'constitution_governance': ['constitution', 'federal', 'government', 'democracy'],
        'rights_duties': ['right', 'duty', 'citizen', 'law', 'justice'],
        'economy_society': ['development', 'poverty', 'society', 'ethics'],
    },
}


def topics_for(subject: str) -> List[str]:
    return list(TOPIC_RULES.get(subject, {}).keys())


def keyword_topic_counts(text: str, subject: str) -> Dict[str, int]:
    """Baseline keyword-frequency scoring (used by the original notebook)."""
    text_l = text.lower()
    rules = TOPIC_RULES.get(subject, {})
    return {topic: sum(text_l.count(k) for k in kws) for topic, kws in rules.items()}
