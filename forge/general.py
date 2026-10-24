"""Garde-fou de capacités générales (leçon 5.1) : le fine-tuning a-t-il fait oublier le reste ?

Vingt questions générales, posées SANS le prompt système de la tâche, notées par mots-clés attendus.
Ce n'est pas un benchmark public (voir lm-evaluation-harness pour cela) : c'est un test de fumée rapide
qui détecte une spécialisation excessive — par exemple un modèle qui répond en JSON à tout.
"""

from __future__ import annotations

import re

from . import infer

QUESTIONS = [
    ("Quelle est la capitale de l'Italie ?", r"rome"),
    ("Combien font 7 fois 8 ?", r"56"),
    ("Traduis en anglais : « le chat dort ».", r"cat.*sleep"),
    ("Quel est le symbole chimique de l'eau ?", r"h\s*2\s*o|h₂o"),
    ("Qui a écrit « Les Misérables » ?", r"hugo"),
    ("Combien y a-t-il de jours dans une semaine ?", r"\b7\b|sept"),
    ("Quelle planète est la plus proche du Soleil ?", r"mercure"),
    ("Donne un synonyme de « rapide ».", r"vite|véloce|prompt|express|preste|vif"),
    ("Quel est le contraire de « chaud » ?", r"froid"),
    ("En quelle année a eu lieu la prise de la Bastille ?", r"1789"),
    ("Combien de côtés a un hexagone ?", r"\b6\b|six"),
    ("Quelle est la langue officielle du Brésil ?", r"portugais"),
    ("Écris le nombre 12 en chiffres romains.", r"xii"),
    ("Quel animal est surnommé le roi de la jungle ?", r"lion"),
    ("Quelle est la racine carrée de 81 ?", r"\b9\b|neuf"),
    ("De quelle couleur est le ciel par temps clair ?", r"bleu"),
    ("Quel océan borde la côte ouest de la France ?", r"atlantique"),
    ("Combien de minutes y a-t-il dans une heure ?", r"60|soixante"),
    ("Quel est le plus grand mammifère du monde ?", r"baleine"),
    ("Complète : « Un tiens vaut mieux que deux tu… »", r"l'auras|l’auras|auras"),
]


def general_score(model, tokenizer) -> tuple[float, float, list[str]]:
    """(part de bonnes réponses, part de réponses au format JSON, réponses)."""
    messages = [[{"role": "user", "content": q}] for q, _ in QUESTIONS]
    outputs, _ = infer.generate(model, tokenizer, messages, max_new_tokens=60)
    correct = sum(bool(re.search(p, o, re.I)) for (_, p), o in zip(QUESTIONS, outputs))
    as_json = sum(o.strip().startswith("{") for o in outputs)
    return correct / len(QUESTIONS), as_json / len(QUESTIONS), outputs
