"""Fusion de modèles (leçon 5.3) : fusionner un adaptateur LoRA, puis combiner plusieurs fine-tunings.

Un fine-tuning modifie les poids : W_ft = W_base + Δ. Le « vecteur de tâche » Δ peut se combiner :
    linéaire : W = W_base + Σ λ_i Δ_i
    TIES     : par paramètre, on garde les plus grands Δ (élagage), on élit un signe majoritaire, puis on
               moyenne seulement les Δ de ce signe — pour limiter les interférences entre tâches.
Implémenté ici en quelques lignes pour le comprendre ; mergekit fait de même à grande échelle.
"""

from __future__ import annotations

import torch


def merge_lora(model_name: str, adapter_dir: str, output_dir: str, dtype=torch.float32):
    """W = W_base + (α/r)·B·A : l'adaptateur disparaît, l'inférence ne coûte plus rien de plus."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(output_dir)
    return merged


def task_vector(base: dict[str, torch.Tensor], tuned: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: tuned[k].float() - base[k].float() for k in base if k in tuned}


def merge_linear(base: dict[str, torch.Tensor], vectors: list[dict[str, torch.Tensor]], weights: list[float]
                 ) -> dict[str, torch.Tensor]:
    return {k: base[k].float() + sum(w * v[k] for w, v in zip(weights, vectors)) for k in base}


def merge_ties(base: dict[str, torch.Tensor], vectors: list[dict[str, torch.Tensor]],
               density: float = 0.2, scale: float = 1.0) -> dict[str, torch.Tensor]:
    """TIES-merging (Yadav et al., 2023) : Trim, Elect sign, disjoint merge."""
    out = {}
    for k in base:
        stacked = torch.stack([v[k] for v in vectors])                       # [tâches, ...]
        flat = stacked.reshape(len(vectors), -1)
        keep = max(1, int(density * flat.shape[1]))
        threshold = flat.abs().kthvalue(flat.shape[1] - keep + 1, dim=1, keepdim=True).values
        trimmed = torch.where(flat.abs() >= threshold, flat, torch.zeros_like(flat))  # 1. élagage
        sign = torch.sign(trimmed.sum(dim=0))                                         # 2. signe élu
        agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
        merged = (trimmed * agree).sum(0) / agree.sum(0).clamp(min=1)          # 3. moyenne disjointe
        out[k] = base[k].float() + scale * merged.reshape(base[k].shape)
    return out
