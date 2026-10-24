"""Leçon 1.1 — Avant de fine-tuner : que donne le prompting seul ?

Le modèle de base, sans entraînement, sur le jeu de TEST : sans exemple (zero-shot), puis avec 3 et 8
exemples dans le prompt (few-shot). Ces chiffres sont la ligne de base que le fine-tuning devra battre.

    python examples/01_baselines.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge import config, infer, metrics, tickets  # noqa: E402

if __name__ == "__main__":
    train = tickets.load_split(config.DATA_DIR, "train")
    test = tickets.load_split(config.DATA_DIR, "test")
    model, tokenizer = infer.load(config.BASE_MODEL)

    # Exemples du prompt : un par catégorie, puis des cas « urgent » et « remboursement »
    by_cat = {c: next(e for e in train if e.categorie == c) for c in tickets.CATEGORIES}
    extra = [next(e for e in train if e.urgence == "haute"), next(e for e in train if e.remboursement),
             next(e for e in train if e.urgence == "basse" and e.produit != "aucun")]
    shots = {"zero-shot": None, "few-shot (3)": list(by_cat.values())[:3],
             "few-shot (8)": list(by_cat.values()) + extra}

    for name, examples in shots.items():
        outputs, seconds = infer.run_task(model, tokenizer, test, shots=examples)
        report = metrics.score(f"{config.BASE_MODEL.split('/')[-1]} {name}", test, outputs)
        print(report.row(), f" {seconds * 1000:.0f} ms/ex")
        for message, output, expected in report.errors[:2]:
            print(f"   ✗ {message[:70]!r}\n     → {output.strip()[:120]!r}\n     attendu {expected}")
