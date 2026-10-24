"""Critères de succès mesurables (leçon 1.4) : la tâche se note sans juge, champ par champ.

    json_valid   : la sortie est-elle un JSON avec les quatre clés et des valeurs autorisées ?
    field_acc    : exactitude de chaque champ (sur toutes les questions : une sortie invalide compte faux)
    exact_match  : les quatre champs sont-ils tous justes ?
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from .tickets import CATEGORIES, PRODUITS, URGENCES, Example

ALLOWED = {
    "categorie": set(CATEGORIES),
    "produit": set(PRODUITS) | {"aucun"},
    "urgence": set(URGENCES),
    "remboursement": {True, False},
}
JSON_OBJECT = re.compile(r"\{.*?\}", re.S)


def parse(output: str) -> dict | None:
    """Premier objet JSON de la sortie, s'il a exactement les bonnes clés et des valeurs autorisées."""
    match = JSON_OBJECT.search(output)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != set(ALLOWED):
        return None
    if not isinstance(data["remboursement"], bool):  # 1 == True en Python : on exige un vrai booléen
        return None
    if any(not isinstance(data[k], (str, bool)) or data[k] not in allowed
           for k, allowed in ALLOWED.items()):
        return None
    return data


@dataclass
class Report:
    name: str
    n: int = 0
    valid: int = 0
    exact: int = 0
    correct: dict[str, int] = field(default_factory=lambda: {k: 0 for k in ALLOWED})
    errors: list[tuple[str, str, dict]] = field(default_factory=list)  # (message, sortie, attendu)
    flags: list[bool] = field(default_factory=list)  # exact ou non, exemple par exemple (comparaisons appariées)

    @property
    def json_valid(self) -> float:
        return self.valid / self.n if self.n else 0.0

    @property
    def exact_match(self) -> float:
        return self.exact / self.n if self.n else 0.0

    def field_acc(self, key: str) -> float:
        return self.correct[key] / self.n if self.n else 0.0

    def row(self) -> str:
        fields = " ".join(f"{k[:6]}={self.field_acc(k):.2f}" for k in ALLOWED)
        return f"{self.name:<34} json={self.json_valid:.2f} exact={self.exact_match:.2f}  {fields}"


def score(name: str, examples: list[Example], outputs: list[str], keep_errors: int = 20) -> Report:
    report = Report(name)
    for example, output in zip(examples, outputs):
        report.n += 1
        data = parse(output)
        expected = example.target()
        report.flags.append(data == expected)
        if data is None:
            if len(report.errors) < keep_errors:
                report.errors.append((example.message, output, expected))
            continue
        report.valid += 1
        hits = {k: data[k] == expected[k] for k in ALLOWED}
        for k, ok in hits.items():
            report.correct[k] += ok
        if all(hits.values()):
            report.exact += 1
        elif len(report.errors) < keep_errors:
            report.errors.append((example.message, output, expected))
    return report


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion (Wilson) : la précision d'une mesure."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def mcnemar(a: Report, b: Report) -> tuple[int, int, float]:
    """Comparaison APPARIÉE de deux modèles sur les mêmes exemples : seuls comptent les désaccords.
    Renvoie (justes pour A seul, justes pour B seul, p-valeur exacte bilatérale)."""
    only_a = sum(x and not y for x, y in zip(a.flags, b.flags))
    only_b = sum(y and not x for x, y in zip(a.flags, b.flags))
    n, k = only_a + only_b, min(only_a, only_b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2 if n else 1.0
    return only_a, only_b, min(1.0, p)
