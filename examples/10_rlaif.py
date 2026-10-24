"""Leçon 4.4 — Constitutional AI : critique-révision par le modèle lui-même, puis RLAIF (DPO sur des
préférences produites par un juge automatique à partir de la constitution).

    python examples/10_rlaif.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import align, config, infer, preferences, rlaif, tickets  # noqa: E402


def show(name: str, examples, replies) -> None:
    s = preferences.compliance(examples, replies)
    rules = "  ".join(f"{k}={v:.2f}" for k, v in s.items() if k not in ("conforme", "mots"))
    print(f"{name:<28} conforme={s['conforme']:.2f}  mots={s['mots']:.0f}  {rules}")


if __name__ == "__main__":
    train = tickets.load_split(config.DATA_DIR, "train")[500:900]   # d'autres messages que pour examples/04
    test = tickets.load_split(config.DATA_DIR, "test")[:150]
    model, tokenizer = infer.load(config.BASE_MODEL)

    # 1. Critique et révision : le modèle relit et corrige son propre brouillon
    drafts = preferences.generate_replies(model, tokenizer, test)
    show("brouillons (base)", test, drafts)
    revised = rlaif.critique_and_revise(model, tokenizer, test, drafts)
    show("après auto-révision", test, revised)

    # 2. RLAIF : 4 réponses échantillonnées par message, le juge désigne la meilleure et la pire
    samples = rlaif.sample_replies(model, tokenizer, train, k=4)
    pairs = rlaif.ai_feedback_pairs(train, samples)
    print(f"Paires produites par le juge : {len(pairs)} sur {len(train)} messages")
    del model
    torch.cuda.empty_cache()

    run = align.train_dpo(pairs, output_dir=str(config.RUNS_DIR / "dpo-rlaif"))
    print(f"DPO sur retours automatiques : {run.seconds / 60:.1f} min")
    model, tokenizer = infer.load(config.BASE_MODEL, adapter=run.output_dir)
    show("DPO-RLAIF", test, preferences.generate_replies(model, tokenizer, test))
