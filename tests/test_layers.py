from collections.abc import Sequence

import numpy as np

from grounding import AliasLayer, EmbeddingLayer, FuzzyLayer, Reference


def test_alias_layer_hits_and_misses(ports: Reference) -> None:
    layer = AliasLayer(ports)
    hit = layer.propose("antwerpen")
    assert hit is not None and hit.key == "antwerp" and hit.accepted
    assert layer.propose("antwrep") is None


def test_fuzzy_accepts_a_typo(ports: Reference) -> None:
    cand = FuzzyLayer(ports).propose("rotterdm")
    assert cand is not None
    assert cand.key == "rotterdam" and cand.accepted


def test_fuzzy_rejects_below_threshold_but_still_reports_best_guess(ports: Reference) -> None:
    cand = FuzzyLayer(ports).propose("rotterdam nl harbour")
    assert cand is not None
    assert cand.key == "rotterdam"
    assert not cand.accepted


def test_fuzzy_skips_very_short_queries(ports: Reference) -> None:
    assert FuzzyLayer(ports).propose("rtn") is None


def test_fuzzy_margin_rule_can_veto_a_high_score(ports: Reference) -> None:
    # Same query, same threshold: only the margin requirement changes the verdict.
    lenient = FuzzyLayer(ports, min_margin=0.0).propose("rotterdm")
    strict = FuzzyLayer(ports, min_margin=99.0).propose("rotterdm")
    assert lenient is not None and lenient.accepted
    assert strict is not None and not strict.accepted
    assert strict.key == lenient.key


def _fake_encoder(table: dict[str, list[float]]):
    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.array([table.get(t, [0.0, 0.0, 1.0]) for t in texts])

    return encode


def test_embedding_layer_uses_injected_encoder(ports: Reference) -> None:
    table = {form: [1.0, 0.0, 0.0] for form, key in ports.surface_index() if key == "antwerp"}
    table["anvers"] = [0.9, 0.1, 0.0]
    cand = EmbeddingLayer(ports, _fake_encoder(table)).propose("anvers")
    assert cand is not None
    assert cand.key == "antwerp" and cand.accepted


def test_embedding_layer_rejects_low_similarity(ports: Reference) -> None:
    table = {form: [1.0, 0.0, 0.0] for form, _ in ports.surface_index()}
    table["somewhere"] = [0.0, 1.0, 0.0]
    cand = EmbeddingLayer(ports, _fake_encoder(table)).propose("somewhere")
    assert cand is not None and not cand.accepted
