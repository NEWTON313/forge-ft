"""Leçons 3.1 et 3.2 — SFT avec LoRA (ou QLoRA, DoRA, full) puis évaluation sur le jeu de test.

    python examples/03_sft_lora.py                 # LoRA r=16
    python examples/03_sft_lora.py qlora           # poids 4 bits + LoRA
    python examples/03_sft_lora.py dora
    python examples/03_sft_lora.py full --tiny     # tout le modèle (petit modèle : 6 Go ne suffisent pas pour 0,5 B)
    python examples/03_sft_lora.py lora --lr=1e-3 --epochs=1 --r=4 --seed=1    # hyperparamètres (leçon 3.3)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, data_quality, infer, metrics, sft, tickets  # noqa: E402

if __name__ == "__main__":
    method = next((a for a in sys.argv[1:] if not a.startswith("--")), "lora")
    model_name = config.TINY_MODEL if "--tiny" in sys.argv else config.BASE_MODEL
    options = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    hyper = {"learning_rate": float(options["lr"])} if "lr" in options else {}
    hyper |= {"epochs": float(options.get("epochs", 2)), "r": int(options.get("r", 16)),
              "seed": int(options.get("seed", config.SEED))}
    tag = "-".join(f"{k}{v}" for k, v in options.items())
    train = data_quality.clean(tickets.load_split(config.DATA_DIR, "train")).kept
    val = tickets.load_split(config.DATA_DIR, "val")
    test = tickets.load_split(config.DATA_DIR, "test")

    output_dir = str(config.RUNS_DIR / f"sft-{method}{'-' + tag if tag else ''}")
    run = sft.train_sft(train, val, method=method, model_name=model_name, output_dir=output_dir, **hyper)
    print(f"{method} : {run.trainable_params / 1e6:.2f} M paramètres entraînés sur {run.total_params / 1e6:.0f} M "
          f"({100 * run.trainable_params / run.total_params:.2f} %) — {run.seconds / 60:.1f} min, "
          f"pic mémoire {run.peak_memory_gb:.2f} Go, perte finale {run.train_loss:.3f}")
    for log in run.log_history:
        if "eval_loss" in log:
            print(f"  époque {log['epoch']:.0f} : perte de validation {log['eval_loss']:.4f}")

    torch.cuda.empty_cache()
    adapter = None if method == "full" else run.output_dir
    model, tokenizer = infer.load(run.output_dir if method == "full" else model_name, adapter=adapter)
    for name, split in (("val", val), ("test", test)):
        outputs, seconds = infer.run_task(model, tokenizer, split)
        print(metrics.score(f"{method} {tag} {name}", split, outputs).row(), f" {seconds * 1000:.0f} ms/ex")
