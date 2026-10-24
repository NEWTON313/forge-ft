"""Génération par lots (évaluation) : modèle de base, avec ou sans adaptateur LoRA, avec ou sans exemples.

Décodage glouton (do_sample=False) : pour mesurer un modèle, on veut une sortie reproductible.
"""

from __future__ import annotations

import time

import torch

from . import chat, config
from .tickets import Example


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_name: str = config.BASE_MODEL, adapter: str | None = None, dtype=torch.float32,
         merged_adapter: str | None = None, **kwargs):
    """`merged_adapter` : un adaptateur fusionné d'abord (par exemple le SFT), avant d'appliquer `adapter` (le DPO)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # en génération par lots, le remplissage se met À GAUCHE
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    if merged_adapter or adapter:
        from peft import PeftModel

        if merged_adapter:
            model = PeftModel.from_pretrained(model, merged_adapter).merge_and_unload()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
    if "device_map" not in kwargs and "quantization_config" not in kwargs:
        model.to(device())
    model.eval()
    return model, tokenizer


def few_shot_messages(message: str, shots: list[Example]) -> list[dict]:
    """Prompting avec exemples : chaque exemple devient un échange utilisateur/assistant avant la question."""
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    for shot in shots:
        messages += [{"role": "user", "content": shot.message}, {"role": "assistant", "content": shot.target_json()}]
    return messages + [{"role": "user", "content": message}]


@torch.inference_mode()
def generate(model, tokenizer, messages_list: list[list[dict]], max_new_tokens: int = 64, batch_size: int = 16
             ) -> tuple[list[str], float]:
    """Renvoie les sorties et le temps moyen par exemple (secondes)."""
    outputs: list[str] = []
    start = time.perf_counter()
    for i in range(0, len(messages_list), batch_size):
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list[i:i + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        new_tokens = generated[:, batch["input_ids"].shape[1]:]
        outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return outputs, (time.perf_counter() - start) / max(len(messages_list), 1)


def run_task(model, tokenizer, examples: list[Example], shots: list[Example] | None = None, **kwargs):
    messages = [few_shot_messages(e.message, shots) if shots else chat.prompt_messages(e.message) for e in examples]
    return generate(model, tokenizer, messages, **kwargs)
