"""Apprentissage par renforcement en ligne avec GRPO (leçon 4.1) : la version moderne et légère du « RL » de RLHF.

PPO (le RL de RLHF classique) entraîne en plus un réseau « critique » qui estime la valeur de chaque état.
GRPO (Group Relative Policy Optimization) s'en passe : pour chaque prompt, il génère un GROUPE de réponses,
les note, et pousse le modèle vers celles qui font mieux que la moyenne du groupe.

La récompense peut venir d'un modèle de récompense… ou d'une fonction VÉRIFIABLE, comme ici : le JSON
est-il valide, les champs sont-ils justes ? (RL avec récompenses vérifiables, « RLVR »)
"""

from __future__ import annotations

import time

import torch

from . import chat, config, metrics
from .compat import patch_trl_without_triton
from .sft import RunResult, lora_config, supports_bf16
from .tickets import Example


def _text(completion) -> str:
    return completion[-1]["content"] if isinstance(completion, list) else completion


def json_reward(completions, **kwargs) -> list[float]:
    """1 si la sortie est un JSON valide avec les bonnes clés et valeurs autorisées, sinon 0."""
    return [1.0 if metrics.parse(_text(c)) else 0.0 for c in completions]


def fields_reward(completions, target: list[str], **kwargs) -> list[float]:
    """Part des quatre champs justes (0 si sortie invalide). `target` vient d'une colonne du dataset."""
    import json

    out = []
    for c, t in zip(completions, target):
        data, expected = metrics.parse(_text(c)), json.loads(t)
        out.append(sum(data[k] == expected[k] for k in expected) / len(expected) if data else 0.0)
    return out


def train_grpo(train: list[Example], model_name: str = config.BASE_MODEL, output_dir: str | None = None,
               max_steps: int = 100, num_generations: int = 4, learning_rate: float = 1e-5,
               batch_size: int = 8, seed: int = config.SEED, adapter: str | None = None) -> RunResult:
    """`adapter` : partir d'un adaptateur SFT (fusionné avant GRPO), le schéma « SFT puis RL »."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    patch_trl_without_triton()
    output_dir = output_dir or str(config.RUNS_DIR / "grpo")
    bf16 = supports_bf16()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if bf16 else torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    dataset = Dataset.from_list([{"prompt": chat.prompt_messages(e.message), "target": e.target_json()}
                                 for e in train])
    args = GRPOConfig(
        output_dir=output_dir, max_steps=max_steps, learning_rate=learning_rate,
        # lot = prompts × générations
        per_device_train_batch_size=batch_size, num_generations=num_generations,
        max_completion_length=64, temperature=1.0, beta=0.0, bf16=bf16, fp16=False,
        gradient_checkpointing=True, logging_steps=5, save_strategy="no", report_to="none", seed=seed,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[json_reward, fields_reward], args=args,
                          train_dataset=dataset, processing_class=tokenizer, peft_config=lora_config())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = trainer.train()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    return RunResult("grpo", output_dir, trainable, total, seconds, peak, output.training_loss,
                     trainer.state.log_history)
