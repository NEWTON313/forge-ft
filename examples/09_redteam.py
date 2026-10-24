"""Leçon 5.2 — Red-teaming : 40 attaques contre le modèle de base et contre le modèle aligné par DPO.

    python examples/09_redteam.py [dossier_adaptateur_dpo]     # par défaut runs/dpo-sigmoid-0.1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, infer, preferences, redteam  # noqa: E402
from forge.tickets import Example  # noqa: E402

if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else str(config.RUNS_DIR / "dpo-sigmoid-0.1")
    attacks = redteam.attacks()
    as_examples = [Example(f"rt{i}", a.message, "compte", "aucun", "haute", False, "redteam")
                   for i, a in enumerate(attacks)]
    results = {}
    for name, adapter_dir in (("base", None), ("DPO", adapter)):
        model, tokenizer = infer.load(config.BASE_MODEL, adapter=adapter_dir)
        replies = preferences.generate_replies(model, tokenizer, as_examples)
        results[name] = redteam.report(attacks, replies)
        print(f"\n{name} — taux de réussite des attaques : {results[name]['global']:.2f}")
        for key, value in results[name].items():
            if key != "global":
                print(f"   {key:<24} {value:.2f}")
        success = next(((a, r) for a, r in zip(attacks, replies) if redteam.attack_success(a, r)), None)
        if success:
            print(f"   ex. réussie ({success[0].technique}/{success[0].goal}) : {success[1][:160]!r}")
        del model
        torch.cuda.empty_cache()
