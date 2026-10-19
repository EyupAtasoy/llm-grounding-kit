"""Guards for the bundled synthetic data, so a typo in a JSON file fails CI."""

import json
from pathlib import Path

from grounding import Reference

DATA = Path(__file__).resolve().parent.parent / "data"


def test_references_load_and_eval_labels_point_at_real_keys() -> None:
    refs = {name: Reference.from_json(DATA / f"{name}.json") for name in ("ports", "cargoes")}
    lines = (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    assert len(cases) >= 50
    for case in cases:
        assert case["ref"] in refs
        if case["expected"] is not None:
            refs[case["ref"]].get(case["expected"])  # raises KeyError if mislabelled
