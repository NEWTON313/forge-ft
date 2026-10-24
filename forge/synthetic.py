"""Données synthétiques par distillation (leçon 2.2) : un modèle « professeur » plus grand écrit de nouveaux
messages de clients, à partir d'étiquettes TIRÉES AU HASARD. L'étiquette est donc connue avant le texte.

    1. tirer une étiquette (catégorie, produit, urgence, remboursement) ;
    2. demander au professeur un message qui correspond, avec une consigne de style variée ;
    3. FILTRER : produit vérifiable par expression régulière, et le professeur doit retrouver l'étiquette
       en relisant son propre message (contrôle de cohérence) — sinon on jette ;
    4. dédupliquer et décontaminer (data_quality), puis mélanger aux données existantes.

Le professeur ne voit jamais le jeu de test, et les descriptions de catégories viennent du cahier des
charges de la tâche, pas des exemples.
"""

from __future__ import annotations

import os
import random
import re

from . import config, infer, metrics
from .tickets import CATEGORIES, PRODUITS, Example

# Attention à la licence : Qwen2.5-3B-Instruct est publié sous la licence « qwen-research » (recherche),
# contrairement aux versions 0,5 B et 1,5 B (Apache 2.0). Pour un usage commercial des données générées,
# choisissez un professeur dont la licence le permet (FORGE_TEACHER_MODEL=Qwen/Qwen2.5-1.5B-Instruct, par exemple).
TEACHER_MODEL = os.environ.get("FORGE_TEACHER_MODEL", "Qwen/Qwen2.5-3B-Instruct")

CATEGORY_SPEC = {
    "retour": "le client veut renvoyer ou échanger un article, ou suit un retour déjà envoyé",
    "garantie": "un produit est en panne, cassé ou défectueux, ou le client s'interroge sur la garantie",
    "livraison": "un colis en retard, perdu, abîmé ou incomplet, un délai, une adresse ou un pays de livraison",
    "paiement": "un débit, une facture, un moyen de paiement, un code promo ou des frais facturés à tort",
    "compte": "la connexion, le mot de passe, les données personnelles, l'espace client ou la newsletter",
}
URGENCY_SPEC = {
    "basse": "une simple question d'information, ton calme",
    "normale": "un problème à régler, ton neutre",
    "haute": "le client est très mécontent ou signale un danger (brûlure, étincelles, blessure)",
}
STYLES = ["très court, sans formule de politesse", "poli et détaillé", "familier, avec une faute de frappe",
          "énervé", "avec le contexte de l'achat (date, usage)", "en une seule phrase"]

GENERATE = """Écris le message qu'un client envoie au support d'une boutique en ligne de mobilier de bureau.
- Sujet : {categorie} ({spec})
- Produit concerné : {produit}
- Ton et urgence : {urgence}
- {remboursement}
- Style : {style}
Écris uniquement le message du client, en français, sans titre ni guillemets."""


def sample_label(rng: random.Random) -> dict:
    categorie = rng.choice(CATEGORIES)
    produit = rng.choice(list(PRODUITS) + ["aucun"] * 2)
    urgence = rng.choice(["basse", "normale", "normale", "haute"])
    return {"categorie": categorie, "produit": produit, "urgence": urgence,
            "remboursement": urgence != "basse" and rng.random() < 0.25}


def generation_messages(label: dict, rng: random.Random) -> list[dict]:
    produit = label["produit"]
    produit_text = "aucun produit précis" if produit == "aucun" else f"{produit} (nomme-le explicitement)"
    remboursement = ("Le client demande explicitement à être remboursé." if label["remboursement"]
                     else "Le client ne demande PAS de remboursement.")
    prompt = GENERATE.format(categorie=label["categorie"], spec=CATEGORY_SPEC[label["categorie"]],
                             produit=produit_text, urgence=URGENCY_SPEC[label["urgence"]],
                             remboursement=remboursement, style=rng.choice(STYLES))
    return [{"role": "user", "content": prompt}]


def mentions(message: str) -> set[str]:
    """Produits nommés dans le message (contrôle par expression régulière, sans modèle)."""
    names = {"Chaise Axis": r"axis|chair-7", "Bureau Lift": r"\blift\b|desk-2", "Lampe Halo": r"halo|lamp-3",
             "Mug Thermo": r"thermo|mug-1"}
    return {p for p, pattern in names.items() if re.search(pattern, message, re.I)}


def generate(teacher, tokenizer, n: int, seed: int = config.SEED, batch_size: int = 8) -> list[tuple[dict, str]]:
    rng = random.Random(seed)
    labels = [sample_label(rng) for _ in range(n)]
    messages = [generation_messages(label, rng) for label in labels]
    outputs: list[str] = []
    for i in range(0, n, batch_size):
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages[i:i + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(teacher.device)
        generated = teacher.generate(**batch, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.95,
                                     pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i // batch_size) % 10 == 0:
            print(f"  génération : {len(outputs)}/{n}", flush=True)
    return [(label, out.strip().strip('"«» ')) for label, out in zip(labels, outputs)]


def verify(teacher, tokenizer, candidates: list[tuple[dict, str]], shots: list[Example]
           ) -> tuple[list[Example], dict]:
    """Garde un message si le produit nommé correspond ET si le professeur retrouve catégorie et
    remboursement."""
    rejected = {"produit": 0, "incoherent": 0}
    kept_candidates = []
    for label, message in candidates:
        named = mentions(message)
        if (label["produit"] == "aucun" and named) or (label["produit"] != "aucun" and named != {label["produit"]}):
            rejected["produit"] += 1
            continue
        kept_candidates.append((label, message))
    outputs, _ = infer.run_task(teacher, tokenizer,
                                [Example("", m, "retour", "aucun", "basse", False, "") for _, m in kept_candidates],
                                shots=shots, batch_size=8)
    kept = []
    for k, ((label, message), output) in enumerate(zip(kept_candidates, outputs)):
        guess = metrics.parse(output)
        if not guess or guess["categorie"] != label["categorie"] or guess["remboursement"] != label["remboursement"]:
            rejected["incoherent"] += 1
            continue
        kept.append(Example(f"syn-{k}", message, label["categorie"], label["produit"], label["urgence"],
                            label["remboursement"], "synthetique"))
    return kept, rejected
