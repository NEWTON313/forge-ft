"""Le jeu de données du cours : trier les messages du support de la boutique Nimbus (leçons 1.1 et 2.x).

Tâche : à partir du message d'un client, produire un JSON
    {"categorie": ..., "produit": ..., "urgence": ..., "remboursement": true|false}

Pourquoi cette tâche ? Elle se MESURE sans juge (JSON valide, exactitude par champ), elle est assez
difficile pour un petit modèle sans entraînement, et elle ressemble à un vrai besoin de production.

Les messages sont générés à partir de gabarits. Point essentiel : les gabarits du jeu de TEST ne sont
jamais utilisés pour l'entraînement (leçon 2.4) — sinon on mesurerait la mémoire du modèle, pas sa
capacité à généraliser.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

CATEGORIES = ["retour", "garantie", "livraison", "paiement", "compte"]
URGENCES = ["basse", "normale", "haute"]

# Chaque produit et les façons dont les clients le nomment (avec l'article : références, minuscules, surnoms)
PRODUITS = {
    "Chaise Axis": ["la chaise Axis", "ma chaise ergonomique Axis", "la CHAIR-7", "le fauteuil Axis",
                    "ma chaise de bureau Axis", "la chaise axis", "l'Axis"],
    "Bureau Lift": ["le bureau Lift", "mon bureau assis-debout", "le DESK-2", "le bureau électrique Lift",
                    "le bureau lift", "mon bureau réglable Lift"],
    "Lampe Halo": ["la lampe Halo", "la LAMP-3", "ma lampe de bureau Halo", "la lampe halo", "la Halo"],
    "Mug Thermo": ["le mug Thermo", "mon mug isotherme", "le MUG-1", "le mug thermo", "la tasse Thermo"],
}
# Signes de danger, propres à chaque produit : ils rendent le message urgent
SAFETY = {
    "Chaise Axis": [" Le pied s'est effondré pendant que j'étais assis.", " Mon fils s'est coupé sur une pièce."],
    "Bureau Lift": [" En plus ça sent le brûlé.", " J'ai vu des étincelles près du moteur."],
    "Lampe Halo": [" J'ai vu des étincelles près de la prise.", " Elle chauffe au point de brûler les doigts."],
    "Mug Thermo": [" Le couvercle a lâché et je me suis brûlé.", " Il fuit et j'ai renversé du café bouillant."],
}
CONTRACTIONS = {" de le ": " du ", " à le ": " au ", " de les ": " des ", " à les ": " aux "}


@dataclass(frozen=True)
class Template:
    categorie: str
    text: str                 # {p} = nom du produit ; absent = message sans produit
    info: bool = False        # simple question d'information → urgence basse
    refund: bool = False      # le client demande explicitement un remboursement


T = Template
# Pour chaque catégorie, les gabarits sont répartis : les 3 derniers sont RÉSERVÉS au test.
TEMPLATES: dict[str, list[Template]] = {
    "retour": [
        T("retour", "Je voudrais renvoyer {p}, ça ne me convient pas. Comment faire ?"),
        T("retour", "Bonjour, j'ai reçu {p} mais la couleur ne va pas du tout avec mon bureau. Je peux faire un retour ?"),
        T("retour", "Comment j'imprime l'étiquette de retour pour {p} ?", info=True),
        T("retour", "J'ai changé d'avis pour {p}, je souhaite être remboursé.", refund=True),
        T("retour", "Le retour de {p} est-il gratuit ?", info=True),
        T("retour", "J'ai renvoyé {p} il y a dix jours et je n'ai toujours pas été remboursé.", refund=True),
        T("retour", "Est-ce que je peux échanger {p} contre un autre modèle ?"),
        T("retour", "{p} ne rentre pas dans mon appartement, je renvoie tout. Remboursement svp.", refund=True),
        T("retour", "Combien de jours ai-je pour retourner {p} ?", info=True),
        T("retour", "Le livreur a récupéré {p} hier, quand serai-je remboursé ?", refund=True),
        T("retour", "Je ne garde pas {p}, vous pouvez organiser l'enlèvement ?"),
        T("retour", "Quelle est la procédure pour rendre {p} acheté le mois dernier ?", info=True),
    ],
    "garantie": [
        T("garantie", "{p} ne fonctionne plus après deux mois. Est-ce couvert par la garantie ?"),
        T("garantie", "{p} fait un bruit bizarre quand je l'utilise, c'est normal ?"),
        T("garantie", "Quelle est la durée de garantie de {p} ?", info=True),
        T("garantie", "Une pièce de {p} s'est cassée, pouvez-vous m'envoyer une pièce de rechange ?"),
        T("garantie", "{p} présente un défaut, je veux un remboursement.", refund=True),
        T("garantie", "La garantie de {p} couvre-t-elle l'usure normale ?", info=True),
        T("garantie", "Depuis la semaine dernière {p} ne marche qu'une fois sur deux."),
        T("garantie", "{p} a lâché au bout de trois mois, je dois l'envoyer où pour la réparation ?"),
        T("garantie", "{p} a un défaut de fabrication visible, que proposez-vous ?"),
        T("garantie", "{p} ne s'allume plus du tout, est-ce pris en charge ?"),
        T("garantie", "Est-ce que la garantie de {p} est valable si je l'ai acheté d'occasion ?", info=True),
        T("garantie", "Le mécanisme de {p} est cassé, je souhaite être remboursé intégralement.", refund=True),
    ],
    "livraison": [
        T("livraison", "Ma commande avec {p} n'est toujours pas arrivée, où en est-elle ?"),
        T("livraison", "Combien coûte la livraison express ?", info=True),
        T("livraison", "Le colis contenant {p} est arrivé abîmé, le carton était déchiré."),
        T("livraison", "Livrez-vous en Belgique ?", info=True),
        T("livraison", "Le suivi indique livré mais je n'ai rien reçu."),
        T("livraison", "Puis-je changer l'adresse de livraison de ma commande ?", info=True),
        T("livraison", "J'ai reçu un autre article à la place de {p}."),
        T("livraison", "Le transporteur a perdu mon colis, je veux être remboursé.", refund=True),
        T("livraison", "Il manque une partie de {p} dans le colis reçu ce matin."),
        T("livraison", "Quel est le délai de livraison pour {p} ?", info=True),
        T("livraison", "Ma livraison a déjà une semaine de retard, aucune nouvelle."),
        T("livraison", "Le colis est bloqué au point relais depuis plusieurs jours."),
    ],
    "paiement": [
        T("paiement", "J'ai été débité deux fois pour la même commande.", refund=True),
        T("paiement", "Peut-on payer en trois fois sans frais ?", info=True),
        T("paiement", "Mon paiement par carte a été refusé, pourquoi ?"),
        T("paiement", "Comment obtenir une facture pour l'achat de {p} ?", info=True),
        T("paiement", "Le code promo ne s'applique pas sur {p}."),
        T("paiement", "Je n'ai pas reçu le remboursement de ma commande annulée.", refund=True),
        T("paiement", "Acceptez-vous PayPal ?", info=True),
        T("paiement", "Le prix de {p} a baissé juste après mon achat, pouvez-vous me rembourser la différence ?",
          refund=True),
        T("paiement", "Ma carte a été débitée mais la commande n'apparaît pas dans mon compte."),
        T("paiement", "Puis-je payer {p} par virement ?", info=True),
        T("paiement", "On m'a facturé des frais de livraison alors que la livraison devait être offerte.",
          refund=True),
        T("paiement", "Le paiement en plusieurs fois a échoué au moment de valider."),
    ],
    "compte": [
        T("compte", "Je n'arrive plus à me connecter à mon compte."),
        T("compte", "Comment changer mon adresse e-mail ?", info=True),
        T("compte", "Je voudrais supprimer mon compte et mes données."),
        T("compte", "Je n'ai pas reçu l'e-mail pour réinitialiser mon mot de passe."),
        T("compte", "Comment voir l'historique de mes commandes ?", info=True),
        T("compte", "Quelqu'un a modifié les informations de mon compte sans mon accord."),
        T("compte", "Où trouver mes points de fidélité ?", info=True),
        T("compte", "Mon compte a été bloqué après plusieurs essais."),
        T("compte", "Je ne retrouve pas ma commande de {p} dans mon espace client."),
        T("compte", "Puis-je fusionner deux comptes ?", info=True),
        T("compte", "Je reçois des e-mails pour une commande que je n'ai jamais passée."),
        T("compte", "Comment me désinscrire de la newsletter ?", info=True),
    ],
}
N_TEST_TEMPLATES = 3

GREETINGS = ["", "", "Bonjour, ", "Bonjour,\n", "Hello, ", "Bonsoir, ", "Madame, Monsieur,\n", "Salut, "]
SIGNATURES = ["", "", "", "\nMerci.", "\nCordialement,\nCamille", "\nMerci d'avance !", "\nBonne journée.",
              "\n-- \nEnvoyé de mon téléphone"]
ANGER = [" C'est inadmissible !", " Je suis vraiment très mécontent.", " C'est la troisième fois que je vous écris !",
         " Sans réponse sous 24 h, je saisis une association de consommateurs.", " J'EXIGE une réponse rapide."]
DETAILS = ["", "", " Commande n° {n}.", " (commande {n})", " Ma commande : {n}.", " Je l'ai depuis trois semaines.",
           " J'habite à Lyon.", " Merci de faire vite, je travaille dessus tous les jours."]


@dataclass
class Example:
    id: str
    message: str
    categorie: str
    produit: str
    urgence: str
    remboursement: bool
    template: str  # identifiant du gabarit : sert à vérifier l'absence de fuite entre train et test

    def target(self) -> dict:
        return {"categorie": self.categorie, "produit": self.produit, "urgence": self.urgence,
                "remboursement": self.remboursement}

    def target_json(self) -> str:
        return json.dumps(self.target(), ensure_ascii=False)


def _noise(text: str, rng: random.Random) -> str:
    """Bruit réaliste : minuscules, ponctuation, fautes de frappe légères."""
    r = rng.random()
    if r < 0.15:
        text = text.lower()
    elif r < 0.25:
        text = text.replace(" ?", "?").replace(",", "")
    if rng.random() < 0.15 and len(text) > 20:  # une lettre doublée au hasard
        i = rng.randrange(5, len(text) - 5)
        if text[i].isalpha():
            text = text[:i] + text[i] + text[i:]
    return text


def make_example(template: Template, template_id: str, rng: random.Random, index: str) -> Example:
    produit = "aucun"
    text = template.text
    if "{p}" in text:
        produit = rng.choice(list(PRODUITS))
        name = rng.choice(PRODUITS[produit])
        text = text.replace("{p}", name)
        for long, short in CONTRACTIONS.items():
            text = text.replace(long, short)
        text = text[0].upper() + text[1:]
    urgence = "basse" if template.info else "normale"
    roll = rng.random()
    if not template.info and roll < 0.22:
        text += rng.choice(ANGER)
        urgence = "haute"
    elif not template.info and template.categorie == "garantie" and produit != "aucun" and roll < 0.40:
        text += rng.choice(SAFETY[produit])
        urgence = "haute"
    text += rng.choice(DETAILS).replace("{n}", f"NB-{rng.randint(10000, 99999)}")
    message = _noise(rng.choice(GREETINGS) + text + rng.choice(SIGNATURES), rng)
    return Example(f"{template_id}-{index}", message.strip(), template.categorie, produit, urgence,
                   template.refund, template_id)


def build_splits(n_train: int = 1200, n_val: int = 150, n_test: int = 300, seed: int = 0
                 ) -> dict[str, list[Example]]:
    """Train et validation partagent les gabarits d'entraînement ; le test n'utilise que les gabarits réservés."""
    rng = random.Random(seed)
    train_t, test_t = [], []
    for cat, templates in TEMPLATES.items():
        for i, t in enumerate(templates):
            (test_t if i >= len(templates) - N_TEST_TEMPLATES else train_t).append((f"{cat}-{i}", t))

    def sample(pool, n, prefix):
        return [make_example(t, tid, rng, f"{prefix}{k}") for k, (tid, t) in
                ((k, rng.choice(pool)) for k in range(n))]

    return {"train": sample(train_t, n_train, "tr"), "val": sample(train_t, n_val, "va"),
            "test": sample(test_t, n_test, "te")}


def save_splits(splits: dict[str, list[Example]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        with open(directory / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for e in examples:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def load_split(directory: Path, name: str) -> list[Example]:
    lines = (directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [Example(**json.loads(line)) for line in lines if line.strip()]
