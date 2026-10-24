# Forge — le laboratoire de fine-tuning et d'alignement

> Projet compagnon du cours gratuit **Fine-tuning et alignement de modèles** — https://iamaestro.academy/fr/courses/fine-tuning-alignment

Projet compagnon du cours IAMaestro **« Fine-Tuning & Alignement de Modèles Avancés »**.

Forge spécialise un petit modèle open-weights (**Qwen2.5-0.5B-Instruct**, Apache 2.0) sur deux tâches de
la boutique fictive *Nimbus*, et MESURE chaque étape :

- **trier** les messages du support (JSON : catégorie, produit, urgence, remboursement) — SFT, LoRA, QLoRA,
  DoRA, full fine-tuning, données synthétiques, GRPO ;
- **répondre** aux clients selon une constitution (concision, signature, pas de promesse de remboursement,
  pas de données sensibles, consigne de sécurité) — DPO, modèle de récompense, RLAIF, red-teaming.

Tout tourne sur un GPU grand public de 6 Go (les chiffres du cours viennent d'une GTX 1060 sous Windows).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
# 1. PyTorch, avec la version CUDA de votre carte (https://pytorch.org) — par exemple :
pip install torch --index-url https://download.pytorch.org/whl/cu126
# 2. Forge et ses dépendances (transformers, trl, peft, datasets, accelerate, bitsandbytes)
pip install -e ".[dev]"
```

Python 3.11 ou plus récent. Les modèles se téléchargent depuis le Hub Hugging Face au premier lancement
(≈ 1 Go pour Qwen2.5-0.5B, ≈ 2 Go pour le professeur Qwen2.5-3B en 4 bits).

**Windows et macOS** : TRL 1.15 calcule ses log-probabilités avec un noyau Triton, livré avec PyTorch sous
Linux uniquement. `forge/compat.py` installe automatiquement un équivalent en PyTorch pur (même résultat,
plus lent) : rien à faire de votre côté.

## Exécuter

```bash
pytest                                      # 16 tests rapides, sans GPU ni téléchargement
python examples/01_baselines.py             # ligne de base : prompting zero-shot et few-shot
python examples/02_budget.py                # mémoire estimée contre mémoire mesurée
python examples/03_sft_lora.py              # SFT LoRA (aussi : qlora, dora, full --tiny, --lr=… --r=…)
python examples/04_dpo.py                   # DPO sur la constitution Nimbus
python examples/05_synthetic_data.py        # données synthétiques par un modèle professeur
python examples/06_reward_model.py          # modèle de récompense et ses biais
python examples/07_grpo.py                  # RL en ligne (GRPO) avec récompenses vérifiables
python examples/08_merge_quantize.py        # fusion LoRA, quantization 8 et 4 bits
python examples/09_redteam.py               # 40 attaques, avant / après alignement
python examples/10_rlaif.py                 # critique-révision et RLAIF
python examples/11_general_regression.py runs/sft-lora     # capacités générales avant/après
python examples/12_compare_runs.py runs/sft-lora runs/...  # intervalles de confiance, test de McNemar
python examples/13_sft_then_dpo.py          # DPO + terme SFT, SFT puis DPO
```

Les adaptateurs et checkpoints s'écrivent dans `runs/` (modifiable par `FORGE_RUNS_DIR`). Le modèle de base
se change par `FORGE_BASE_MODEL`, le professeur des données synthétiques par `FORGE_TEACHER_MODEL`.

**Licences.** Qwen2.5-0.5B-Instruct et SmolLM2-135M-Instruct sont sous Apache 2.0. Le professeur par défaut,
Qwen2.5-3B-Instruct, est sous licence « qwen-research » (recherche) : pour un usage commercial des données
générées, choisissez un autre professeur (par exemple `FORGE_TEACHER_MODEL=Qwen/Qwen2.5-1.5B-Instruct`).

**Matériel.** Les chiffres du cours ont été mesurés sur une GTX 1060 6 Go (architecture Pascal, sans bf16 : tout
en fp32). Sur un GPU récent, `forge/sft.py` passe automatiquement en bf16 ; les durées seront bien plus courtes.
vLLM (`serving/`) demande Linux et un GPU de capacité de calcul 7.0 ou plus : ces commandes n'ont pas été
exécutées sur le matériel du cours.

## Résultats mesurés (GTX 1060, 300 messages de test aux formulations jamais vues)

| Tri des messages | Exactitude | JSON valide | Pic mémoire | Durée |
|---|---|---|---|---|
| Prompting, 8 exemples | 0,16 | 0,54 | — | — |
| LoRA r=16 (3 graines) | 0,65 / 0,65 / 0,74 | 1,00 | 2,87 Go | 10 min |
| QLoRA | 0,53 | 1,00 | 1,66 Go | 12 min |
| Full fine-tuning (1 époque) | 0,67 | 1,00 | 8,41 Go (débordement en RAM) | 7 min |
| LoRA + 352 messages synthétiques (3 graines) | 0,68 / 0,70 / 0,79 | 1,00 | 2,87 Go | 16 min |
| GRPO 100 pas depuis le modèle de base | 0,00 | 0,94 | 3,13 Go | 7 min |

| Réponses aux clients (constitution Nimbus) | Conformes | Attaques réussies |
|---|---|---|
| Modèle de base | 0,00 | 0,50 |
| DPO seul | 0,00 | 0,65 |
| DPO + terme SFT | 0,97 | 0,25 |
| SFT puis DPO | 0,96 | 0,25 |

## Contenu

| Fichier | Rôle | Leçons |
|---|---|---|
| `forge/tickets.py`, `data/` | Génération des données (gabarits d'entraînement / gabarits de test réservés) | 1.1, 2.1, 2.4 |
| `forge/metrics.py` | Parseur strict, exactitude, exactitude par champ | 1.4 |
| `forge/budget.py` | Estimation mémoire, cache KV, durée et coût | 1.3 |
| `forge/data_quality.py` | Nettoyage, données personnelles, MinHash, décontamination | 2.1, 2.4 |
| `forge/synthetic.py` | Données synthétiques par un professeur, avec contrôle de cohérence | 2.2 |
| `forge/chat.py` | Chat templates, format prompt/completion | 2.3 |
| `forge/sft.py`, `forge/compat.py` | SFT avec TRL : full, LoRA, QLoRA, DoRA ; compatibilité sans Triton | 3.1 – 3.3 |
| `configs/` | FSDP2 (Accelerate) et DeepSpeed ZeRO-3 | 3.4 |
| `serving/` | Commandes vLLM et client OpenAI | 5.4 |
| `forge/rl.py` | GRPO, récompenses vérifiables | 4.1 |
| `forge/general.py` | Test de fumée des capacités générales | 5.1 |
| `forge/policy.py`, `forge/preferences.py`, `forge/align.py` | Constitution, paires de préférences, DPO | 4.2 |
| `forge/reward.py` | Modèle de récompense, précision, sondes de biais | 4.3 |
| `forge/rlaif.py` | Critique-révision, préférences par un juge automatique | 4.4 |
| `forge/redteam.py` | Attaques et taux de réussite | 5.2 |
| `forge/merging.py`, `forge/quantize.py` | Fusion LoRA, merge linéaire/TIES, quantization | 5.3 |

## Licence

Le code est distribué sous licence MIT (voir `LICENSE`) : vous pouvez l'utiliser, le modifier et le redistribuer, y compris dans des projets professionnels. Les textes du cours restent protégés par le droit d'auteur. Les modèles téléchargés restent soumis à leurs propres licences.

Le cours complet, gratuit, en français et en anglais : https://iamaestro.academy/fr/courses/fine-tuning-alignment
