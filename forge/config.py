"""Réglages de Forge, le laboratoire de fine-tuning du cours. Tout ce qui se compare dans les expériences est ici."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FORGE_DATA_DIR", ROOT / "data"))
RUNS_DIR = Path(os.environ.get("FORGE_RUNS_DIR", ROOT / "runs"))  # adaptateurs, checkpoints, journaux

# Modèle de base : petit, open-weights (Apache 2.0), multilingue, avec un chat template.
# 0,5 milliard de paramètres : entraînable avec LoRA sur un GPU grand public de 6 Go.
BASE_MODEL = os.environ.get("FORGE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# Encore plus petit, pour les tests et le full fine-tuning sur petite carte
TINY_MODEL = os.environ.get("FORGE_TINY_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

SYSTEM_PROMPT = (
    "Tu tries les messages du support client de la boutique Nimbus. Réponds UNIQUEMENT par un objet JSON "
    'avec les clés "categorie" (retour, garantie, livraison, paiement, compte), "produit" (Chaise Axis, '
    'Bureau Lift, Lampe Halo, Mug Thermo ou aucun), "urgence" (basse, normale, haute) et '
    '"remboursement" (true si le client demande explicitement un remboursement, sinon false).'
)

SEED = 42
