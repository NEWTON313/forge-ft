"""Leçon 5.3 — Fusionner l'adaptateur LoRA dans le modèle, puis mesurer la quantization 8 et 4 bits.

    python examples/08_merge_quantize.py [dossier_adaptateur]     # par défaut runs/sft-lora
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, infer, merging, metrics, quantize, tickets  # noqa: E402

if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else str(config.RUNS_DIR / "sft-lora")
    merged_dir = str(config.RUNS_DIR / "merged")
    test = tickets.load_split(config.DATA_DIR, "test")

    # 1. Base + adaptateur, sans fusion
    model, tokenizer = infer.load(config.BASE_MODEL, adapter=adapter)
    outputs, s_adapter = infer.run_task(model, tokenizer, test)
    print(metrics.score("base + adaptateur", test, outputs).row(), f" {s_adapter * 1000:.0f} ms/ex")
    del model
    torch.cuda.empty_cache()

    # 2. Fusion : un modèle autonome, mêmes réponses, inférence sans surcoût d'adaptateur
    merging.merge_lora(config.BASE_MODEL, adapter, merged_dir)
    model, tokenizer = infer.load(merged_dir)
    merged_outputs, s_merged = infer.run_task(model, tokenizer, test)
    same = sum(a == b for a, b in zip(outputs, merged_outputs))
    print(metrics.score("modèle fusionné", test, merged_outputs).row(), f" {s_merged * 1000:.0f} ms/ex")
    print(f"Sorties identiques avant/après fusion : {same}/{len(test)}")
    del model
    torch.cuda.empty_cache()

    # 3. Quantization du modèle fusionné
    for result in quantize.compare(merged_dir, test):
        print(f"{result['bits']:>2} bits : poids {result['poids_mo']:.0f} Mo, {result['ms_par_exemple']:.0f} ms/ex, "
              f"exact {result['exact']:.2f}, JSON {result['json']:.2f}")
