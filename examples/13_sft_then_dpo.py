"""Leçon 4.2 — Les deux remèdes classiques quand DPO seul échoue : DPO + terme SFT, et SFT puis DPO.

    python examples/13_sft_then_dpo.py          # après examples/04_dpo.py (qui construit runs/pairs.jsonl)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import align, config, infer, preferences, redteam, sft, tickets  # noqa: E402
from forge.tickets import Example  # noqa: E402


def evaluate(name: str, test, adapter=None, merged_adapter=None) -> None:
    model, tokenizer = infer.load(config.BASE_MODEL, adapter=adapter, merged_adapter=merged_adapter)
    s = preferences.compliance(test, preferences.generate_replies(model, tokenizer, test))
    attacks = redteam.attacks()
    targets = [Example(f"rt{i}", a.message, "compte", "aucun", "haute", False, "rt") for i, a in enumerate(attacks)]
    attack = redteam.report(attacks, preferences.generate_replies(model, tokenizer, targets))["global"]
    rules = "  ".join(f"{k}={v:.2f}" for k, v in s.items() if k not in ("conforme", "mots"))
    print(f"{name:<22} conforme={s['conforme']:.2f}  mots={s['mots']:.0f}  {rules}  attaques réussies={attack:.2f}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train = tickets.load_split(config.DATA_DIR, "train")[:500]
    test = tickets.load_split(config.DATA_DIR, "test")[:150]
    pairs = preferences.load_pairs(config.RUNS_DIR / "pairs.jsonl")

    # Remède 1 : DPO + un terme de vraisemblance sur la réponse choisie
    run = align.train_dpo(pairs, loss_type="sigmoid+sft", output_dir=str(config.RUNS_DIR / "dpo-sigmoid+sft"))
    evaluate("DPO + SFT (perte)", test, adapter=run.output_dir)

    # Remède 2 : SFT sur les réponses de référence, puis DPO à partir du modèle SFT
    sft_run = sft.train_sft(train, method="lora", epochs=1, output_dir=str(config.RUNS_DIR / "sft-replies"),
                            formatter=preferences.to_reply_sft)
    evaluate("SFT seul", test, adapter=sft_run.output_dir)
    dpo_run = align.train_dpo(pairs, adapter=sft_run.output_dir, output_dir=str(config.RUNS_DIR / "sft-then-dpo"))
    evaluate("SFT puis DPO", test, adapter=dpo_run.output_dir, merged_adapter=sft_run.output_dir)
