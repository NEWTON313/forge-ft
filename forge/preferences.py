"""Données de préférences pour l'alignement (module 4) : des paires (prompt, réponse choisie, réponse rejetée).

Recette « sur la politique » (on-policy) : la réponse REJETÉE est une vraie réponse du modèle qu'on veut
corriger, quand elle enfreint la constitution ; la réponse CHOISIE est la réponse de référence conforme.
On apprend ainsi au modèle à s'éloigner de SES erreurs réelles, pas d'erreurs imaginaires.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import infer, policy
from .tickets import Example


@dataclass
class Pair:
    message: str
    chosen: str
    rejected: str
    broken_rules: list[str]
    example_id: str

    def to_trl(self) -> dict:
        """Format conversationnel de TRL : le chat template est appliqué par le trainer."""
        prompt = [{"role": "system", "content": policy.REPLY_SYSTEM},
                  {"role": "user", "content": self.message}]
        return {"prompt": prompt, "chosen": [{"role": "assistant", "content": self.chosen}],
                "rejected": [{"role": "assistant", "content": self.rejected}]}


def reply_messages(message: str) -> list[dict]:
    return [{"role": "system", "content": policy.REPLY_SYSTEM}, {"role": "user", "content": message}]


def to_reply_sft(example: Example) -> dict:
    """SFT sur les réponses de référence : rapproche le modèle des réponses « choisies » avant DPO."""
    return {"prompt": reply_messages(example.message),
            "completion": [{"role": "assistant", "content": policy.reference_reply(example)}]}


def generate_replies(model, tokenizer, examples: list[Example], max_new_tokens: int = 160, batch_size: int = 8
                     ) -> list[str]:
    outputs, _ = infer.generate(model, tokenizer, [reply_messages(e.message) for e in examples],
                                max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [o.strip() for o in outputs]


def build_pairs(examples: list[Example], replies: list[str]) -> tuple[list[Pair], int]:
    """Une paire par réponse NON conforme ; renvoie aussi le nombre de réponses déjà conformes."""
    pairs, already_ok = [], 0
    for example, reply in zip(examples, replies):
        verdict = policy.check(reply, example)
        broken = [name for name, ok in verdict.items() if not ok]
        if not broken:
            already_ok += 1
            continue
        pairs.append(Pair(example.message, policy.reference_reply(example), reply, broken, example.id))
    return pairs, already_ok


def compliance(examples: list[Example], replies: list[str]) -> dict[str, float]:
    """Part des réponses conformes, au total et règle par règle (la règle « safety » sur les seuls cas concernés)."""
    n = len(examples)
    out = {"conforme": sum(policy.compliant(r, e) for e, r in zip(examples, replies)) / n}
    for rule in policy.RULES:
        relevant = [(e, r) for e, r in zip(examples, replies)
                    if rule.name != "safety" or policy.DANGER.search(e.message)]
        out[rule.name] = sum(rule.check(r, e) for e, r in relevant) / max(len(relevant), 1)
    out["mots"] = sum(len(r.split()) for r in replies) / n
    return out


def save_pairs(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pairs(path: Path) -> list[Pair]:
    return [Pair(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_pairs(pairs: list[Pair], test_fraction: float = 0.15, seed: int = 0) -> tuple[list[Pair], list[Pair]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]
