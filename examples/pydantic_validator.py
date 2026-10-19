"""Ground a field inside a Pydantic model, e.g. the schema an LLM fills via structured output.

    pip install pydantic
    python examples/pydantic_validator.py

The raw span is kept next to the canonical key: when the key is None the
record can still go to a review queue with what the model actually said.
"""

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from grounding import Reference, Resolver

PORTS = Resolver.build(
    Reference.from_json(Path(__file__).resolve().parent.parent / "data" / "ports.json")
)


class Shipment(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_port_raw: str
    load_port: str | None = None  # canonical key, or None when it could not be grounded

    @model_validator(mode="before")
    @classmethod
    def ground_load_port(cls, data: Any) -> Any:
        if isinstance(data, dict) and "load_port_raw" in data:
            return {**data, "load_port": PORTS.resolve(data["load_port_raw"]).key}
        return data


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # In a real pipeline this JSON comes back from the LLM.
    for payload in ('{"load_port_raw": "Constantsa"}', '{"load_port_raw": "Santos"}'):
        shipment = Shipment.model_validate_json(payload)
        print(shipment)
