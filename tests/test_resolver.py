from grounding import UNKNOWN, AliasLayer, Reference, Resolver


def test_alias_wins_before_fuzzy_runs(ports: Reference) -> None:
    m = Resolver.build(ports).resolve("Port of RTM")
    assert m.key == "rotterdam"
    assert m.method == "alias"
    assert m.canonical == "Rotterdam"


def test_fuzzy_fills_in_when_alias_misses(ports: Reference) -> None:
    m = Resolver.build(ports).resolve("Constantsa")
    assert m.key == "constanta"
    assert m.method == "fuzzy"


def test_unknown_keeps_the_rejected_guess_for_review(ports: Reference) -> None:
    m = Resolver.build(ports).resolve("Rotterdam NL harbour")
    assert not m.is_known
    assert m.method == UNKNOWN
    assert m.suggestion == "rotterdam"


def test_out_of_reference_value_is_unknown(ports: Reference) -> None:
    m = Resolver.build(ports).resolve("Vladivostok")
    assert m.key is None and m.canonical is None


def test_empty_and_none_inputs(ports: Reference) -> None:
    resolver = Resolver.build(ports)
    assert resolver.resolve(None).method == UNKNOWN
    assert resolver.resolve("  ,  ").method == UNKNOWN


def test_alias_only_resolver_does_not_guess(ports: Reference) -> None:
    resolver = Resolver(reference=ports, layers=(AliasLayer(ports),))
    assert resolver.resolve("rotterdm").key is None


def test_resolve_many_preserves_order(ports: Reference) -> None:
    keys = [m.key for m in Resolver.build(ports).resolve_many(["klang", "nowhere", "rtm"])]
    assert keys == ["port_klang", None, "rotterdam"]
