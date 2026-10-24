"""Constitutional AI et RLAIF (leçon 4.4) : remplacer l'annotateur humain par un retour automatique.

Deux mécanismes de l'approche « Constitutional AI » (Anthropic, 2022) :
1. Critique et révision (phase supervisée) : le modèle relit sa réponse à la lumière d'un principe, la
   critique, puis la réécrit ; les révisions servent de données de SFT.
2. RLAIF (phase de préférences) : un juge automatique compare plusieurs réponses du modèle selon la
   constitution ; ses préférences remplacent celles d'annotateurs humains pour DPO ou un modèle de récompense.

Ici le juge est la vérification programmatique de forge/policy.py — un « vérificateur » — et, pour la
critique-révision, le modèle lui-même. Avec un modèle juge plus capable (par exemple Claude via l'API),
le même code s'applique à des principes non vérifiables par programme (ton, pertinence, honnêteté).
"""

from __future__ import annotations

import torch

from . import policy
from .preferences import Pair, reply_messages
from .tickets import Example

REVISE = """Voici le message d'un client et un brouillon de réponse.

Message : {message}

Brouillon : {draft}

Règles à respecter :
{constitution}

Réécris le brouillon pour qu'il respecte TOUTES les règles. Écris uniquement la réponse finale."""


def judge_score(reply: str, example: Example) -> float:
    """Nombre de règles respectées, et un léger bonus pour la concision (départage les ex aequo)."""
    verdict = policy.check(reply, example)
    return sum(verdict.values()) - len(reply.split()) / 1000


@torch.inference_mode()
def sample_replies(model, tokenizer, examples: list[Example], k: int = 4, temperature: float = 0.9,
                   max_new_tokens: int = 160, batch_size: int = 4) -> list[list[str]]:
    """k réponses échantillonnées par message (diversité nécessaire pour avoir quelque chose à comparer)."""
    out: list[list[str]] = []
    for i in range(0, len(examples), batch_size):
        prompts = [tokenizer.apply_chat_template(reply_messages(e.message), tokenize=False, add_generation_prompt=True)
                   for e in examples[i:i + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                   top_p=0.95, num_return_sequences=k, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [[t.strip() for t in texts[j * k:(j + 1) * k]] for j in range(len(prompts))]
    return out


def ai_feedback_pairs(examples: list[Example], samples: list[list[str]]) -> list[Pair]:
    """Meilleure contre pire réponse selon le juge, quand il les départage nettement (≥ 1 règle)."""
    pairs = []
    for example, replies in zip(examples, samples):
        scored = sorted(replies, key=lambda r: judge_score(r, example))
        worst, best = scored[0], scored[-1]
        if judge_score(best, example) - judge_score(worst, example) >= 1:
            broken = [n for n, ok in policy.check(worst, example).items() if not ok]
            pairs.append(Pair(example.message, best, worst, broken, example.id))
    return pairs


def critique_and_revise(model, tokenizer, examples: list[Example], drafts: list[str], batch_size: int = 8,
                        max_new_tokens: int = 160) -> list[str]:
    """Le modèle réécrit son propre brouillon à la lumière de la constitution."""
    from .infer import generate

    messages = [[{"role": "user", "content": REVISE.format(message=e.message, draft=d,
                                                           constitution=policy.constitution_text())}]
                for e, d in zip(examples, drafts)]
    outputs, _ = generate(model, tokenizer, messages, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [o.strip() for o in outputs]
