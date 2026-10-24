"""Leçon 1.3 — Estimer la mémoire d'entraînement, puis la comparer à la mesure.

    python examples/02_budget.py

Affiche l'estimation (forge/budget.py) pour plusieurs modèles et méthodes. Les mesures réelles du cours
(pic de torch.cuda.max_memory_allocated) sont imprimées par examples/03_sft_lora.py à la fin de chaque run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.budget import GB, ModelShape, inference_memory, training_cost, training_memory  # noqa: E402

MODELS = {  # paramètres, dimension cachée, couches, vocabulaire (fiches des modèles)
    "SmolLM2-135M": ModelShape(135e6, 576, 30, 49152),
    "Qwen2.5-0.5B": ModelShape(494e6, 896, 24, 151936),
    "Llama-3.1-8B": ModelShape(8.03e9, 4096, 32, 128256),
    "Llama-3.1-70B": ModelShape(70.6e9, 8192, 80, 128256),
}

if __name__ == "__main__":
    print(f"{'Modèle':<15}{'Méthode':<10}{'Total':>9}  détail (Go)")
    for name, shape in MODELS.items():
        lora_params = shape.params * 0.0175  # LoRA r=16 sur toutes les couches linéaires : ~1,75 % ici
        setups = {
            "full": dict(trainable=shape.params, weights_dtype="bf16"),
            "lora": dict(trainable=lora_params, weights_dtype="bf16"),
            "qlora": dict(trainable=lora_params, weights_dtype="nf4"),
        }
        for method, kw in setups.items():
            est = training_memory(shape, batch=8, seq=256, checkpointing=True, **kw)
            print(f"{name:<15}{method:<10}{est.total / GB:>8.1f}  {est.describe()}")
    # Les runs du cours (fp32 : GPU sans bf16 ; batch 8, ~200 jetons ; TRL ne matérialise pas les logits)
    # comparés au pic mesuré par torch.cuda.max_memory_allocated sur une GTX 1060 (6 Go)
    print("\nEstimation contre mesure (Go) :")
    runs = [("Qwen2.5-0.5B", "LoRA", 8.8e6, "fp32", 2.87), ("Qwen2.5-0.5B", "QLoRA", 8.8e6, "nf4", 1.66),
            ("Qwen2.5-0.5B", "full", 494e6, "fp32", 8.41), ("SmolLM2-135M", "LoRA", 4.9e6, "fp32", 1.00),
            ("SmolLM2-135M", "full", 134.5e6, "fp32", 2.32)]
    for name, method, trainable, weights, measured in runs:
        est = training_memory(MODELS[name], trainable=trainable, batch=8, seq=200, weights_dtype=weights,
                              compute_dtype="fp32", checkpointing=True, full_logits=False)
        print(f"  {name:<14}{method:<7} estimé {est.total / GB:5.2f}   mesuré {measured:5.2f}")
    kv = inference_memory(MODELS["Llama-3.1-8B"], "bf16", batch=16, context=8192, kv_heads=8, head_dim=128)
    print(f"Inférence Llama-3.1-8B bf16, 16 requêtes × 8 192 jetons : {kv / GB:.1f} Go")
    hours, cost = training_cost(tokens=50e6, epochs=3, tokens_per_second=2500, price_per_hour=2.5)
    print(f"50 M jetons × 3 époques à 2 500 jetons/s : {hours:.0f} h, {cost:.0f} $ à 2,5 $/h")
