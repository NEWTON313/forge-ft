"""Estimer la mémoire GPU et le coût d'un entraînement AVANT de le lancer (leçon 1.3).

Mémoire d'entraînement = poids + gradients + états de l'optimiseur + activations (+ logits).
Les ordres de grandeur suffisent pour décider : full fine-tuning, LoRA ou QLoRA ? Quelle carte ?
On les compare ensuite à la mesure réelle (torch.cuda.max_memory_allocated) : voir examples/02_budget.py.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "nf4": 0.5}
GB = 1024**3


@dataclass
class ModelShape:
    params: float          # nombre total de paramètres
    hidden: int            # dimension cachée
    layers: int            # nombre de blocs Transformer
    vocab: int             # taille du vocabulaire

    @classmethod
    def from_config(cls, config, params: float) -> ModelShape:
        return cls(params, config.hidden_size, config.num_hidden_layers, config.vocab_size)


@dataclass
class MemoryEstimate:
    weights: float
    gradients: float
    optimizer: float
    activations: float
    logits: float

    @property
    def total(self) -> float:
        return self.weights + self.gradients + self.optimizer + self.activations + self.logits

    def describe(self) -> str:
        parts = ", ".join(f"{k} {v / GB:.2f}" for k, v in vars(self).items())
        return f"{self.total / GB:.2f} Go ({parts})"


def training_memory(shape: ModelShape, trainable: float, batch: int, seq: int, weights_dtype: str = "bf16",
                    compute_dtype: str = "bf16", checkpointing: bool = False, full_logits: bool = True
                    ) -> MemoryEstimate:
    """Ordres de grandeur, en octets.

    - poids : tous les paramètres, dans leur format de stockage (nf4 pour QLoRA) ;
    - gradients : seulement les paramètres ENTRAÎNÉS, en fp32 par sécurité ;
    - AdamW : deux moments fp32 par paramètre entraîné (8 octets) ;
    - activations : ~ 34·b·s·h octets par couche en 16 bits (formule de Korthikanti et al., 2022, sans le
      terme d'attention quadratique, négligeable avec FlashAttention/SDPA), ~2·b·s·h avec le gradient
      checkpointing (on ne garde que l'entrée de chaque couche) ;
    - logits : b·s·vocab en fp32, et autant pour leur gradient — énorme quand le vocabulaire dépasse 100 000.
      TRL 1.x les calcule par morceaux, sur les seuls jetons de la réponse : full_logits=False pour l'ignorer.
    """
    act_bytes = BYTES[compute_dtype] / 2  # la formule est donnée pour 16 bits
    per_layer = (2 if checkpointing else 34) * batch * seq * shape.hidden * act_bytes
    return MemoryEstimate(
        weights=shape.params * BYTES[weights_dtype],
        gradients=trainable * 4,
        optimizer=trainable * 8,
        activations=per_layer * shape.layers,
        logits=2 * batch * seq * shape.vocab * 4 if full_logits else 0.0,
    )


def inference_memory(shape: ModelShape, dtype: str, batch: int, context: int, kv_heads: int,
                     head_dim: int) -> float:
    """Poids + cache KV (2 tenseurs par couche : clés et valeurs), en octets."""
    kv = 2 * shape.layers * batch * context * kv_heads * head_dim * BYTES["bf16"]
    return shape.params * BYTES[dtype] + kv


def training_cost(tokens: float, epochs: float, tokens_per_second: float,
                  price_per_hour: float) -> tuple[float, float]:
    """Durée (heures) et coût d'un entraînement, à partir d'un débit MESURÉ sur quelques pas."""
    hours = tokens * epochs / tokens_per_second / 3600
    return hours, hours * price_per_hour
