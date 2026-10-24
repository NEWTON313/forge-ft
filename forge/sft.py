"""Fine-tuning supervisé (SFT) avec TRL : full fine-tuning, LoRA, QLoRA, DoRA (module 3).

Une seule fonction, quatre méthodes : on compare ainsi mémoire, durée et qualité à données égales.
Sur un GPU sans bf16 natif (avant l'architecture Ampere), on entraîne en fp32 : TRL active bf16 par défaut,
et PyTorch l'ÉMULE alors sans prévenir — l'entraînement tourne, mais nettement plus lentement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from . import chat, config
from .compat import patch_trl_without_triton
from .tickets import Example


@dataclass
class RunResult:
    method: str
    output_dir: str
    trainable_params: int
    total_params: int
    seconds: float
    peak_memory_gb: float
    train_loss: float
    log_history: list[dict]


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05, dora: bool = False):
    from peft import LoraConfig

    # « all-linear » : toutes les projections linéaires (attention ET MLP), sauf la tête de sortie
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear",
                      use_dora=dora, task_type="CAUSAL_LM")


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)


def compute_dtype() -> torch.dtype:
    """bf16 sur les GPU récents ; sinon fp32. Pas de fp16 par défaut : les GPU Pascal grand public (GTX 10xx)
    calculent le fp16 natif bien plus lentement que le fp32, et le fp16 déborde plus facilement à l'entraînement."""
    return torch.bfloat16 if supports_bf16() else torch.float32


def train_sft(train: list[Example], val: list[Example] | None = None, method: str = "lora",
              model_name: str = config.BASE_MODEL, output_dir: str | None = None, epochs: float = 2,
              learning_rate: float | None = None, batch_size: int = 8, grad_accum: int = 1, r: int = 16,
              gradient_checkpointing: bool = True, max_steps: int = -1, seed: int = config.SEED,
              formatter=chat.to_prompt_completion) -> RunResult:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    patch_trl_without_triton()  # Windows/macOS : TRL sans Triton (voir forge/compat.py)
    output_dir = output_dir or str(config.RUNS_DIR / f"sft-{method}")
    bf16 = supports_bf16()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization = None
    if method == "qlora":  # poids gelés en 4 bits (NF4), adaptateurs LoRA entraînés en 16/32 bits
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, quantization_config=quantization)
    peft = None if method == "full" else lora_config(r=r, dora=method == "dora")
    # Taux d'apprentissage : ~1e-5 pour tout le modèle, ~1e-4 à 2e-4 pour des adaptateurs LoRA
    lr = learning_rate or (1e-5 if method == "full" else 2e-4)

    args = SFTConfig(
        output_dir=output_dir, num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0,  # 0.05 = 5 % des pas
        max_length=512, bf16=bf16, fp16=False, gradient_checkpointing=gradient_checkpointing,
        logging_steps=10, save_strategy="no", report_to="none", seed=seed,
        eval_strategy="epoch" if val else "no", per_device_eval_batch_size=batch_size,
    )
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tokenizer, peft_config=peft,
        train_dataset=Dataset.from_list([formatter(e) for e in train]),
        eval_dataset=Dataset.from_list([formatter(e) for e in val]) if val else None,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = trainer.train()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    trainer.save_model(output_dir)  # LoRA : seulement l'adaptateur (quelques Mo)
    tokenizer.save_pretrained(output_dir)
    return RunResult(method, output_dir, trainable, total, seconds, peak, output.training_loss,
                     trainer.state.log_history)
