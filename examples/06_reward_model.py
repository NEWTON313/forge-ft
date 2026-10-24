"""Leçon 4.3 — Entraîner un modèle de récompense sur les paires de préférences, puis le VALIDER.

    python examples/06_reward_model.py      # après examples/04_dpo.py (qui construit runs/pairs.jsonl)

Validation en trois temps : précision sur des paires jamais vues, puis deux sondes de biais :
- la longueur : ajouter du remplissage poli à une bonne réponse change-t-il son score ?
- le raccourci : une réponse FAUTIVE à laquelle on ajoute la signature Nimbus gagne-t-elle des points ?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge import config, policy, preferences, reward  # noqa: E402

FILLER = (" N'hésitez pas à nous recontacter si besoin, nous restons à votre entière disposition pour toute"
          " question complémentaire concernant votre commande ou nos produits.")

if __name__ == "__main__":
    pairs = preferences.load_pairs(config.RUNS_DIR / "pairs.jsonl")
    train_pairs, test_pairs = preferences.split_pairs(pairs)
    run = reward.train_reward_model(train_pairs)
    print(f"Modèle de récompense : {len(train_pairs)} paires, {run.seconds / 60:.1f} min, "
          f"pic {run.peak_memory_gb:.2f} Go")

    scorer = reward.RewardScorer(run.output_dir)
    print(f"Précision sur {len(test_pairs)} paires de test : {scorer.pairwise_accuracy(test_pairs):.2f}")

    msgs = [p.message for p in test_pairs]
    good = scorer.score(msgs, [p.chosen for p in test_pairs])
    padded = scorer.score(msgs, [p.chosen.replace(f"\n{policy.SIGNATURE}", FILLER + f"\n{policy.SIGNATURE}")
                                 for p in test_pairs])
    bad = scorer.score(msgs, [p.rejected for p in test_pairs])
    bad_signed = scorer.score(msgs, [p.rejected.rstrip() + f"\n{policy.SIGNATURE}" for p in test_pairs])
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"Score moyen — choisie : {mean(good):.2f} ; choisie + remplissage : {mean(padded):.2f}")
    print(f"Score moyen — rejetée : {mean(bad):.2f} ; rejetée + signature : {mean(bad_signed):.2f}")
    wins = sum(bs > g for bs, g in zip(bad_signed, good))
    print(f"Réponses fautives signées qui battent la bonne réponse : {wins}/{len(test_pairs)}")
