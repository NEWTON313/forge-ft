"""Modèle de récompense (leçon 4.3) : un modèle qui lit (prompt, réponse) et renvoie un score scalaire.

Entraînement : perte de Bradley-Terry, −log σ(r_choisie − r_rejetée). Le modèle n'apprend pas un score absolu,
seulement un ORDRE : les scores n'ont de sens que comparés entre eux, pour un même prompt.

Valider un modèle de récompense, ce n'est pas seulement mesurer sa précision sur des paires de test :
il faut chercher ses BIAIS (longueur, formules toutes faites…), car un modèle optimisé contre lui les
exploitera (« reward hacking »).
"""

from __future__ import annotations

import time

import torch

from . import config, policy
from .preferences import Pair
from .sft import RunResult, supports_bf16


def reward_messages(message: str, reply: str) -> list[dict]:
    return [{"role": "system", "content": policy.REPLY_SYSTEM}, {"role": "user", "content": message},
            {"role": "assistant", "content": reply}]


def train_reward_model(pairs: list[Pair], model_name: str = config.BASE_MODEL, output_dir: str | None = None,
                       epochs: float = 1, learning_rate: float = 1e-4, batch_size: int = 4,
                       seed: int = config.SEED) -> RunResult:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    output_dir = output_dir or str(config.RUNS_DIR / "reward")
    bf16 = supports_bf16()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Un modèle de langage + une tête linéaire qui produit UN nombre (num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, dtype=torch.bfloat16 if bf16 else torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = Dataset.from_list([{"chosen": reward_messages(p.message, p.chosen),
                                  "rejected": reward_messages(p.message, p.rejected)} for p in pairs])
    args = RewardConfig(output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
                        per_device_train_batch_size=batch_size, max_length=512, bf16=bf16, fp16=False,
                        gradient_checkpointing=True, logging_steps=10, save_strategy="no", report_to="none",
                        seed=seed)
    # La tête de score (« score ») est entraînée en entier : modules_to_save
    peft = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_CLS",
                      modules_to_save=["score"])
    trainer = RewardTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=dataset,
                            peft_config=peft)
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
    return RunResult("reward", output_dir, trainable, total, seconds, peak, output.training_loss,
                     trainer.state.log_history)


class RewardScorer:
    """Charge un modèle de récompense entraîné et note des (message, réponse)."""

    def __init__(self, adapter_dir: str, model_name: str = config.BASE_MODEL) -> None:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, dtype=torch.float32)
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, adapter_dir).eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def score(self, messages: list[str], replies: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(messages), batch_size):
            texts = [self.tokenizer.apply_chat_template(reward_messages(m, r), tokenize=False)
                     for m, r in zip(messages[i:i + batch_size], replies[i:i + batch_size])]
            batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
                                   add_special_tokens=False).to(self.model.device)
            scores += self.model(**batch).logits.squeeze(-1).float().tolist()
        return scores

    def pairwise_accuracy(self, pairs: list[Pair]) -> float:
        chosen = self.score([p.message for p in pairs], [p.chosen for p in pairs])
        rejected = self.score([p.message for p in pairs], [p.rejected for p in pairs])
        return sum(c > r for c, r in zip(chosen, rejected)) / len(pairs)
