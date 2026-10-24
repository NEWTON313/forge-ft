"""Leçon 4.2 — Aligner les réponses aux clients sur la constitution Nimbus avec DPO.

    python examples/04_dpo.py                  # DPO (perte sigmoïde), β = 0.1
    python examples/04_dpo.py ipo 0.5          # autre perte, autre β

1. Le modèle de base répond à 500 messages d'entraînement : ses réponses non conformes deviennent les
   réponses REJETÉES, la réponse de référence conforme est la réponse CHOISIE.
2. DPO avec LoRA sur ces paires.
3. On compare la conformité des réponses AVANT / APRÈS sur 150 messages de test (gabarits jamais vus).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import align, config, infer, preferences, tickets  # noqa: E402


def show(name: str, scores: dict) -> None:
    rules = "  ".join(f"{k}={v:.2f}" for k, v in scores.items() if k not in ("conforme", "mots"))
    print(f"{name:<22} conforme={scores['conforme']:.2f}  mots={scores['mots']:.0f}  {rules}")


if __name__ == "__main__":
    loss_type = sys.argv[1] if len(sys.argv) > 1 else "sigmoid"
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    train = tickets.load_split(config.DATA_DIR, "train")[:500]
    test = tickets.load_split(config.DATA_DIR, "test")[:150]
    pairs_file = config.RUNS_DIR / "pairs.jsonl"

    model, tokenizer = infer.load(config.BASE_MODEL)
    before = preferences.generate_replies(model, tokenizer, test)
    show("base", preferences.compliance(test, before))
    if pairs_file.exists():
        pairs = preferences.load_pairs(pairs_file)
    else:
        replies = preferences.generate_replies(model, tokenizer, train)
        pairs, already_ok = preferences.build_pairs(train, replies)
        preferences.save_pairs(pairs, pairs_file)
        print(f"{len(pairs)} paires construites, {already_ok} réponses déjà conformes écartées")
    del model
    torch.cuda.empty_cache()

    run = align.train_dpo(pairs, loss_type=loss_type, beta=beta,
                          output_dir=str(config.RUNS_DIR / f"dpo-{loss_type}-{beta}"))
    print(f"DPO {loss_type} β={beta} : {run.seconds / 60:.1f} min, pic mémoire {run.peak_memory_gb:.2f} Go")
    for log in run.log_history[::4]:
        if "rewards/margins" in log and "loss" in log:
            print(f"  pas {log.get('step', '?')} : perte {log['loss']:.3f}, marge {log['rewards/margins']:.2f}, "
                  f"précision {log.get('rewards/accuracies', 0):.2f}")

    model, tokenizer = infer.load(config.BASE_MODEL, adapter=run.output_dir)
    after = preferences.generate_replies(model, tokenizer, test)
    show(f"DPO {loss_type} β={beta}", preferences.compliance(test, after))
    print("\nExemple :\n", test[0].message, "\n--- avant ---\n", before[0], "\n--- après ---\n", after[0])
