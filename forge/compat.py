"""Compatibilité TRL sans Triton (Windows, macOS, GPU anciens).

TRL 1.x calcule les log-probabilités des jetons avec un noyau Triton qui évite de matérialiser les logits
[jetons × vocabulaire]. Triton n'est livré avec PyTorch que sous Linux : ailleurs, l'import échoue en silence,
la fonction vaut None, et SFTTrainer, DPOTrainer ou GRPOTrainer plantent au premier pas
(« 'NoneType' object has no attribute 'apply' »).

`patch_trl_without_triton()` installe un équivalent en PyTorch pur : même calcul, découpé par paquets de
jetons et recalculé à la rétropropagation (checkpointing) pour limiter la mémoire. Plus lent que le noyau,
mais exact. Sous Linux avec Triton, la fonction ne fait rien.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
TOKEN_CHUNK = 512


def _chunk_stats(hidden, weight, bias, targets, temperature, softcap, logit_scale):
    logits = hidden.float() @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    logits = logits * logit_scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits / temperature
    log_z = torch.logsumexp(logits, dim=-1)
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return target_logit - log_z, entropy, torch.logsumexp(2 * log_probs, dim=-1), logits.mean(-1), \
        target_logit >= logits.max(-1).values


class TorchChunkedLogProb:
    """Même interface que trl.kernels.ChunkedLogProbFunction (méthode `apply`)."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None,
              logit_scale=1.0, outputs=OPTIONAL_OUTPUTS):
        parts = []
        for start in range(0, hidden.shape[0], TOKEN_CHUNK):
            args = (hidden[start:start + TOKEN_CHUNK], weight, bias, targets[start:start + TOKEN_CHUNK],
                    temperature, final_logit_softcapping, logit_scale)
            parts.append(checkpoint(_chunk_stats, *args, use_reentrant=False) if torch.is_grad_enabled()
                         else _chunk_stats(*args))
        log_probs, entropy, sq, mean, top1 = (torch.cat(x) if x[0].ndim else x[0] for x in zip(*parts)) \
            if parts else (hidden.new_zeros(0),) * 5
        return (log_probs,
                entropy if "entropy" in outputs else None,
                sq.detach() if "log_sum_sq_probs" in outputs else None,
                mean.detach() if "mean_logits" in outputs else None,
                top1 if "is_top1" in outputs else None)


def patch_trl_without_triton() -> bool:
    """Renvoie True si le correctif a été installé (Triton absent), False sinon."""
    import trl.trainer.utils as utils

    if utils._ChunkedLogProbFunction is not None:
        return False
    utils._ChunkedLogProbFunction = TorchChunkedLogProb
    return True
