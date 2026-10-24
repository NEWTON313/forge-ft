"""Leçon 4.1 — RL en ligne avec GRPO et des récompenses vérifiables, à partir du modèle de base.

    python examples/07_grpo.py [pas] [adaptateur_sft]      # 100 pas, à partir du modèle de base par défaut

Le modèle génère 4 réponses par message ; chacune est notée (JSON valide ? champs justes ?) ; le modèle est
poussé vers les réponses meilleures que la moyenne de leur groupe. Aucune réponse « modèle » n'est montrée :
le modèle n'apprend que des notes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, data_quality, infer, metrics, rl, tickets  # noqa: E402

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    start = sys.argv[2] if len(sys.argv) > 2 else None   # ex. runs/sft-lora : SFT puis GRPO
    train = data_quality.clean(tickets.load_split(config.DATA_DIR, "train")).kept
    test = tickets.load_split(config.DATA_DIR, "test")
    run = rl.train_grpo(train, max_steps=steps, adapter=start,
                        output_dir=str(config.RUNS_DIR / ("grpo-from-sft" if start else "grpo")))
    print(f"GRPO {steps} pas : {run.seconds / 60:.1f} min, pic mémoire {run.peak_memory_gb:.2f} Go")
    for log in run.log_history:
        if "reward" in log and log.get("step", 0) % 20 == 0:
            print(f"  pas {log['step']} : récompense moyenne {log['reward']:.2f} "
                  f"(JSON {log.get('rewards/json_reward/mean', 0):.2f}, champs {log.get('rewards/fields_reward/mean', 0):.2f})")
    torch.cuda.empty_cache()
    model, tokenizer = infer.load(config.BASE_MODEL, adapter=run.output_dir, merged_adapter=start)
    outputs, _ = infer.run_task(model, tokenizer, test)
    print(metrics.score(f"GRPO {steps} pas{' après SFT' if start else ''} test", test, outputs).row())
