"""Curation, nettoyage, déduplication et décontamination (leçons 2.1 et 2.4).

Chaque filtre renvoie ce qu'il garde ET ce qu'il rejette, avec la raison : un rapport de rejets se relit,
il révèle souvent un problème dans la source des données plutôt que dans les exemples eux-mêmes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .metrics import parse
from .tickets import Example

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}")
IBAN = re.compile(r"\bFR\d{2}(?:\s?\d{4}){5}\s?\d{3}\b")
# Mots outils français fréquents. Une heuristique : en production, utilisez un modèle d'identification de
# langue (fastText lid.176, lingua…), bien plus fiable sur les textes courts.
FRENCH_WORDS = set("""le la les l de des du d un une et est je j mon ma mes pas pour que qui en au aux sur avec
vous votre vos ce cette ces il elle on a ai été être comment combien quel quelle quand où pourquoi puis peut
merci bonjour bonsoir après depuis plusieurs encore toujours mais ne plus très mon ma mes me m""".split())


def normalize(text: str) -> str:
    """Minuscules, sans accents ni ponctuation, espaces réduits : la forme de comparaison."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def mask_pii(text: str) -> str:
    """Les données personnelles n'ont rien à faire dans les poids : un modèle peut les régurgiter."""
    # L'ordre compte : un IBAN contient des suites de chiffres qu'on prendrait pour un téléphone
    return PHONE.sub("[TELEPHONE]", IBAN.sub("[IBAN]", EMAIL.sub("[EMAIL]", text)))


def french_ratio(text: str) -> float:
    words = normalize(text).split()
    return sum(w in FRENCH_WORDS for w in words) / len(words) if words else 0.0


@dataclass
class CleanReport:
    kept: list[Example] = field(default_factory=list)
    rejected: dict[str, list[Example]] = field(default_factory=lambda: defaultdict(list))

    def summary(self) -> str:
        reasons = ", ".join(f"{k}: {len(v)}" for k, v in self.rejected.items()) or "aucun"
        return f"{len(self.kept)} gardés ; rejets — {reasons}"


def clean(examples: list[Example], min_chars: int = 15, max_chars: int = 1500, min_french: float = 0.1
          ) -> CleanReport:
    report = CleanReport()
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for e in examples:
        labels_by_text[normalize(e.message)].add(e.target_json())
    seen: set[str] = set()
    for e in examples:
        key = normalize(e.message)
        if not (min_chars <= len(e.message) <= max_chars):
            report.rejected["longueur"].append(e)
        elif parse(e.target_json()) is None:
            report.rejected["étiquette_invalide"].append(e)
        elif french_ratio(e.message) < min_french:
            report.rejected["langue"].append(e)
        elif len(labels_by_text[key]) > 1:
            # même texte, deux réponses : laquelle croire ?
            report.rejected["étiquettes_contradictoires"].append(e)
        elif key in seen:
            report.rejected["doublon_exact"].append(e)
        else:
            seen.add(key)
            report.kept.append(Example(**{**vars(e), "message": mask_pii(e.message)}))
    return report


# --- Quasi-doublons : MinHash + LSH --------------------------------------------------------------------------

def shingles(text: str, n: int = 3) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def minhash(sets: list[set[str]], num_perm: int = 64, seed: int = 0) -> np.ndarray:
    """Signature MinHash : pour chaque permutation (simulée par un hachage salé), le plus petit haché
    de l'ensemble. La part de signatures égales entre deux textes estime leur similarité de Jaccard."""
    rng = np.random.default_rng(seed)
    salts = rng.integers(1, 2**31, size=num_perm)
    out = np.full((len(sets), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sets):
        for sh in s:
            h = int(hashlib.md5(sh.encode()).hexdigest()[:15], 16)
            out[i] = np.minimum(out[i], (h * salts + 12345) % (2**61 - 1))
    return out


def near_duplicates(examples: list[Example], threshold: float = 0.8, bands: int = 16) -> list[tuple[int, int, float]]:
    """Paires (i, j, similarité estimée) au-dessus du seuil. Le LSH (bandes de signature) évite de comparer
    toutes les paires : seules celles qui partagent au moins une bande identique sont vérifiées."""
    sigs = minhash([shingles(e.message) for e in examples])
    rows = sigs.shape[1] // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for b in range(bands):
            buckets[(b, sig[b * rows:(b + 1) * rows].tobytes())].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.add((members[a], members[b]))
    found = []
    for i, j in pairs:
        sim = float(np.mean(sigs[i] == sigs[j]))
        if sim >= threshold:
            found.append((i, j, sim))
    return sorted(found)


def dedup_near(examples: list[Example], threshold: float = 0.8) -> tuple[list[Example], int]:
    drop = {j for _, j, _ in near_duplicates(examples, threshold)}
    return [e for i, e in enumerate(examples) if i not in drop], len(drop)


# --- Décontamination -----------------------------------------------------------------------------------------

def ngrams(text: str, n: int) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contamination(train: list[Example], test: list[Example], n: int = 8, max_doc_freq: float = 0.01,
                  min_overlap: float = 0.5) -> list[tuple[Example, float]]:
    """Exemples de test dont au moins `min_overlap` des n-grammes (de n mots) figurent dans l'entraînement.

    Les grands laboratoires utilisent 13 mots pour des documents longs ; sur des messages de 15 à 40 mots,
    8 est un compromis. Les n-grammes présents dans plus de `max_doc_freq` des exemples d'entraînement
    (formules de politesse, phrases toutes faites) sont ignorés : ils ne révèlent aucune fuite. Et un seul
    n-gramme commun ne suffit pas : on regarde la PART du texte de test déjà vue.
    """
    counts: dict[str, int] = defaultdict(int)
    for e in train:
        for g in ngrams(e.message, n):
            counts[g] += 1
    limit = max(1, max_doc_freq * len(train))
    common = {g for g, c in counts.items() if c > limit}
    out = []
    for e in test:
        grams = ngrams(e.message, n) - common
        if grams:
            ratio = sum(g in counts for g in grams) / len(grams)
            if ratio >= min_overlap:
                out.append((e, ratio))
    return out
