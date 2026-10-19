import json
from pathlib import Path

import pytest

from grounding import Entry, Reference


def test_every_surface_form_is_indexed(ports: Reference) -> None:
    assert ports.lookup("rotterdam") == "rotterdam"
    assert ports.lookup("rtm") == "rotterdam"
    assert ports.lookup("port klang") == "port_klang"  # key with underscore folds too


def test_lookup_expects_normalised_input(ports: Reference) -> None:
    assert ports.lookup("RTM") is None


def test_duplicate_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        Reference.from_entries([Entry("a", "A"), Entry("a", "A again")])


def test_alias_shared_by_two_entries_is_rejected() -> None:
    with pytest.raises(ValueError, match="maps to both"):
        Reference.from_entries([Entry("a", "Alpha", ("x",)), Entry("b", "Beta", ("X",))])


def test_from_json_reads_entries_and_fillers(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(
        json.dumps(
            {
                "prefixes": ["port of"],
                "entries": [{"key": "hamburg", "canonical": "Hamburg", "aliases": ["hh"]}],
            }
        ),
        encoding="utf-8",
    )
    ref = Reference.from_json(path)
    assert len(ref) == 1
    assert ref.get("hamburg").aliases == ("hh",)
    assert ref.normalizer("Port of Hamburg") == "hamburg"
