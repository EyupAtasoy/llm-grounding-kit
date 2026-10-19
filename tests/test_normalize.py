from grounding import Normalizer, fold


def test_fold_lowercases_and_strips_accents() -> None:
    assert fold("Constanța") == "constanta"
    assert fold("İzmir") == "izmir"


def test_fold_turns_punctuation_into_single_spaces() -> None:
    assert fold("  Hot-Rolled,  Coils!! ") == "hot rolled coils"
    assert fold("iron_ore") == "iron ore"


def test_normalizer_strips_one_prefix_and_one_suffix() -> None:
    norm = Normalizer(prefixes=("port of",), suffixes=("in bulk", "bulk"))
    assert norm("Port of Hamburg") == "hamburg"
    assert norm("coal in bulk") == "coal"


def test_normalizer_keeps_words_that_only_look_like_fillers() -> None:
    norm = Normalizer(prefixes=("port of",))
    assert norm("Port Klang") == "port klang"


def test_normalizer_never_reduces_a_filler_to_empty() -> None:
    norm = Normalizer(suffixes=("bulk",))
    assert norm("bulk") == "bulk"


def test_empty_input() -> None:
    assert Normalizer()("   ") == ""
