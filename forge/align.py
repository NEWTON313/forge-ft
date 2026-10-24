"""Alignement par préférences avec TRL (module 4) : DPO et ses variantes, modèle de récompense.

DPO n'a pas besoin de modèle de récompense ni de génération pendant l'entraînement : il augmente la
probabilité relative de la réponse choisie par rapport à la rejetée, en restant proche d'un modèle de
RÉFÉRENCE (le modèle de départ, gelé). Le paramètre β règle cette laisse : petit β, le modèle s'éloigne
davantage de la référence ; grand β, il reste prudent.
"""

from __future__ import annotations

import time

import torch

from . import config
from .compat import patch_trl_without_triton
from .preferences import Pair
from .sft import RunResult, lora_config, supports_bf16


def train_dpo(pairs: list[Pair], model_name: str = config.BASE_MODEL, adapter: str | None = None,
              output_dir: str | None = None, loss_type: str = "sigmoid", beta: float = 0.1, epochs: float = 1,
              learning_rate: float = 5e-5, batch_size: int = 4, grad_accum: int = 2, seed: int = config.SEED
              ) -> RunResult:
    """DPO avec LoRA. `adapter` : partir d'un adaptateur SFT existant (fusionné dans le modèle avant DPO).
    `loss_type` : une perte (« sigmoid », « ipo »…) ou plusieurs combinées (« sigmoid+sft » : DPO + un terme de
    vraisemblance sur la réponse choisie, qui l'empêche de devenir elle aussi improbable)."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    patch_trl_without_triton()
    output_dir = output_dir or str(config.RUNS_DIR / f"dpo-{loss_type}")
    bf16 = supports_bf16()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:  # SFT puis DPO : l'adaptateur SFT est fusionné, le DPO entraîne un nouvel adaptateur
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()

    args = DPOConfig(
        output_dir=output_dir, loss_type=loss_type.split("+"), beta=beta, num_train_epochs=epochs,
        learning_rate=learning_rate, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0.1, max_length=512, bf16=bf16, fp16=False,
        gradient_checkpointing=True, logging_steps=5, save_strategy="no", report_to="none", seed=seed,
    )
    # Avec un peft_config, pas de seconde copie du modèle : la référence est le modèle SANS adaptateur
    trainer = DPOTrainer(model=model, args=args, processing_class=tokenizer, peft_config=lora_config(),
                         train_dataset=Dataset.from_list([p.to_trl() for p in pairs]))
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = trainer.train()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return RunResult(f"dpo-{loss_type}", output_dir, trainable, total, seconds, peak, output.training_loss,
                     trainer.state.log_history)
