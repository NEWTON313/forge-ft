"""Tests rapides, sans GPU ni téléchargement : données, métriques, qualité, budget, constitution, fusion."""

import json

import torch

from forge import budget, data_quality, merging, metrics, policy, redteam, tickets
from forge.compat import TorchChunkedLogProb


def test_splits_do_not_share_templates():
    splits = tickets.build_splits(n_train=300, n_val=50, n_test=100)
    train_t = {e.template for e in splits["train"]}
    assert not train_t & {e.template for e in splits["test"]}
    assert {e.template for e in splits["val"]} <= train_t


def test_targets_are_valid_json():
    for e in tickets.build_splits(n_train=200, n_val=0, n_test=50)["train"]:
        assert metrics.parse(e.target_json()) == e.target()


def test_parse_is_strict():
    good = '{"categorie": "retour", "produit": "aucun", "urgence": "basse", "remboursement": false}'
    assert metrics.parse("Voici : " + good)
    assert metrics.parse(good.replace("urgence", "urgency")) is None           # mauvaise clé
    assert metrics.parse(good.replace('"basse"', '"faible"')) is None          # valeur non autorisée
    assert metrics.parse(good.replace("false", '"false"')) is None    # chaîne au lieu de booléen
    assert metrics.parse(good.replace("false", "0")) is None          # entier au lieu de booléen
    assert metrics.parse(good.replace('"aucun"', '["aucun"]')) is None  # liste : pas de plantage


def test_score_counts_invalid_as_wrong():
    ex = tickets.build_splits(n_train=2, n_val=0, n_test=0)["train"]
    report = metrics.score("t", ex, [ex[0].target_json(), "pas de JSON"])
    assert report.json_valid == 0.5 and report.exact_match == 0.5


def test_clean_rejects_conflicts_and_duplicates():
    e = tickets.build_splits(n_train=1, n_val=0, n_test=0)["train"][0]
    dup = tickets.Example(**{**vars(e), "id": "x"})
    conflict = tickets.Example(**{**vars(e), "id": "y", "message": e.message.upper(), "urgence": "haute"})
    report = data_quality.clean([e, dup, conflict])
    assert "étiquettes_contradictoires" in report.rejected


def test_pii_masking():
    text = "Appelez-moi au 06 12 34 56 78 ou jean.dupont@mail.fr, IBAN FR76 3000 6000 0112 3456 7890 189"
    assert data_quality.mask_pii(text) == "Appelez-moi au [TELEPHONE] ou [EMAIL], IBAN [IBAN]"


def test_contamination_detects_leak_not_boilerplate():
    splits = tickets.build_splits(n_train=600, n_val=100, n_test=100)
    assert len(data_quality.contamination(splits["train"], splits["test"])) == 0
    assert len(data_quality.contamination(splits["train"], splits["val"])) > 30


def test_near_duplicates():
    base = tickets.build_splits(n_train=1, n_val=0, n_test=0)["train"][0]
    twin = tickets.Example(**{**vars(base), "id": "t", "message": base.message + " Merci."})
    assert data_quality.near_duplicates([base, twin], threshold=0.5)


def test_budget_orders_of_magnitude():
    shape = budget.ModelShape(8e9, 4096, 32, 128256)
    full = budget.training_memory(shape, trainable=8e9, batch=1, seq=512)
    qlora = budget.training_memory(shape, trainable=1.4e8, batch=1, seq=512, weights_dtype="nf4")
    assert full.total / budget.GB > 100 and qlora.total / budget.GB < 10


def test_policy_reference_replies_are_compliant():
    for e in tickets.build_splits(n_train=300, n_val=0, n_test=0)["train"]:
        assert policy.compliant(policy.reference_reply(e), e), e.message


def test_policy_detects_violations():
    e = tickets.Example("x", "Ma lampe Halo fait des étincelles.", "garantie", "Lampe Halo", "haute", False, "x")
    verdict = policy.check("Nous allons vous rembourser. Donnez votre numéro de carte.", e)
    assert not verdict["no_promise"] and not verdict["no_sensitive"] and not verdict["safety"]
    assert not verdict["signature"]


def test_redteam_attacks_cover_every_goal():
    attacks = redteam.attacks()
    assert len(attacks) == 40 and {a.goal for a in attacks} == set(redteam.GOALS)
    assert redteam.report(attacks, [policy.SIGNATURE] * 40)["global"] < 1


def test_compat_logprobs_match_reference():
    torch.manual_seed(0)
    h, w = torch.randn(600, 8, requires_grad=True), torch.randn(30, 8)
    t = torch.randint(0, 30, (600,))
    log_probs = TorchChunkedLogProb.apply(h, w, None, t, 1.0, 4096)[0]
    reference = torch.log_softmax(h @ w.t(), -1).gather(-1, t[:, None]).squeeze(-1)
    assert torch.allclose(log_probs, reference, atol=1e-5)


def test_ties_merge_keeps_agreeing_signs():
    base = {"w": torch.zeros(4)}
    v1, v2 = {"w": torch.tensor([1.0, 1.0, -1.0, 0.1])}, {"w": torch.tensor([1.0, -1.0, -1.0, 0.0])}
    merged = merging.merge_ties(base, [v1, v2], density=1.0)
    assert merged["w"][0] == 1.0 and merged["w"][2] == -1.0
    linear = merging.merge_linear(base, [v1, v2], [0.5, 0.5])
    assert torch.allclose(linear["w"], torch.tensor([1.0, 0.0, -1.0, 0.05]))


def test_examples_roundtrip(tmp_path):
    splits = tickets.build_splits(n_train=5, n_val=1, n_test=2)
    tickets.save_splits(splits, tmp_path)
    loaded = tickets.load_split(tmp_path, "train")
    assert [e.target() for e in loaded] == [e.target() for e in splits["train"]]
    assert json.loads(loaded[0].target_json())


def test_confidence_interval_and_mcnemar():
    low, high = metrics.wilson_interval(195, 300)
    assert 0.59 < low < 0.60 and 0.70 < high < 0.71
    a, b = metrics.Report("a"), metrics.Report("b")
    a.flags, b.flags = [True] * 10 + [False] * 10, [True] * 10 + [True] * 10
    only_a, only_b, p = metrics.mcnemar(a, b)
    assert (only_a, only_b) == (0, 10) and p < 0.01
