"""Leçon 2.2 — Données synthétiques : un modèle professeur (Qwen2.5-3B, chargé en 4 bits) écrit de nouveaux
messages, on les filtre, puis on ré-entraîne le modèle élève (0,5 B) avec LoRA et on compare sur le test.

    python examples/05_synthetic_data.py            # génère (≈ 1 h sur un GPU de 6 Go) puis entraîne
    python examples/05_synthetic_data.py --seed=1   # données déjà générées : entraîne avec une autre graine
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, data_quality, infer, metrics, sft, synthetic, tickets  # noqa: E402
from forge.sft import compute_dtype  # noqa: E402

if __name__ == "__main__":
    train = data_quality.clean(tickets.load_split(config.DATA_DIR, "train")).kept
    test = tickets.load_split(config.DATA_DIR, "test")
    out_file = config.DATA_DIR / "synthetic.jsonl"

    if not out_file.exists():
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype())
        teacher, tokenizer = infer.load(synthetic.TEACHER_MODEL, dtype=compute_dtype(), quantization_config=q,
                                        device_map={"": 0})
        candidates = synthetic.generate(teacher, tokenizer, n=600)
        shots = [next(e for e in train if e.categorie == c) for c in tickets.CATEGORIES]
        kept, rejected = synthetic.verify(teacher, tokenizer, candidates, shots)
        print(f"Professeur : {len(candidates)} messages, {len(kept)} gardés, rejets {rejected}")
        cleaned = data_quality.clean(kept)
        print("Nettoyage :", cleaned.summary())
        unique, dropped = data_quality.dedup_near(cleaned.kept, threshold=0.8)
        leaks = data_quality.contamination(unique, test)
        print(f"Quasi-doublons retirés : {dropped} ; exemples de test « contaminés » par le synthétique : {len(leaks)}")
        tickets.save_splits({"synthetic": unique}, config.DATA_DIR)
        for e in unique[:5]:
            print("  ", repr(e.message[:110]), e.target_json())
        del teacher
        torch.cuda.empty_cache()

    synthetic_examples = tickets.load_split(config.DATA_DIR, "synthetic")
    seed = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--seed=")), config.SEED))
    suffix = "" if seed == config.SEED else f"-seed{seed}"
    run = sft.train_sft(train + synthetic_examples, method="lora", seed=seed,
                        output_dir=str(config.RUNS_DIR / f"sft-lora-syn{suffix}"))
    print(f"LoRA gabarits + {len(synthetic_examples)} synthétiques : {run.seconds / 60:.1f} min")
    model, tokenizer = infer.load(config.BASE_MODEL, adapter=run.output_dir)
    outputs, _ = infer.run_task(model, tokenizer, test)
    print(metrics.score(f"lora + synthétique{suffix} test", test, outputs).row())
