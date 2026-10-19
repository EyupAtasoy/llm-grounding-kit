import pytest

from grounding import Entry, Normalizer, Reference


@pytest.fixture
def ports() -> Reference:
    return Reference.from_entries(
        [
            Entry("rotterdam", "Rotterdam", ("rtm", "europoort")),
            Entry("antwerp", "Antwerp", ("antwerpen",)),
            Entry("constanta", "Constanta", ("constantza",)),
            Entry("port_klang", "Port Klang", ("klang",)),
            Entry("novorossiysk", "Novorossiysk", ("novo",)),
            Entry("novigrad", "Novigrad"),
        ],
        normalizer=Normalizer(prefixes=("port of",)),
    )
