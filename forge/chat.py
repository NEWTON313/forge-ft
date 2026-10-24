"""Formats conversationnels et chat templates (leçon 2.3).

Un modèle « Instruct » a été entraîné sur des conversations mises en forme par SON chat template (balises de
rôles, jetons spéciaux de début et de fin de tour). Fine-tuner avec un autre format — ou oublier le jeton de
fin de tour — produit un modèle qui ne s'arrête plus de générer ou qui répond dans un format hybride.
Règle : on ne formate JAMAIS les conversations à la main, on passe par tokenizer.apply_chat_template.
"""

from __future__ import annotations

from . import config
from .tickets import Example


def prompt_messages(message: str, system: str | None = config.SYSTEM_PROMPT) -> list[dict]:
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": message}]


def to_prompt_completion(example: Example) -> dict:
    """Format « prompt / completion » de TRL : la perte ne porte que sur la complétion."""
    return {"prompt": prompt_messages(example.message),
            "completion": [{"role": "assistant", "content": example.target_json()}]}


def render(tokenizer, example: Example) -> str:
    """Le texte EXACT que voit le modèle à l'entraînement : à relire au moins une fois par projet."""
    answer = {"role": "assistant", "content": example.target_json()}
    messages = prompt_messages(example.message) + [answer]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def generation_prompt(tokenizer, message: str) -> str:
    """Le prompt d'inférence : même format, terminé par l'ouverture du tour de l'assistant."""
    return tokenizer.apply_chat_template(prompt_messages(message), tokenize=False,
                                         add_generation_prompt=True)
