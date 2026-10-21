import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


def test_portable_schemas_validate_original_fixtures_without_network():
    root = Path(__file__).parents[1]
    names = ["traffic-model", "neural-model", "ensemble-model"]
    schemas = {
        name: json.loads((root / "spec/v1" / f"{name}.schema.json").read_text()) for name in names
    }
    registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in schemas.values()
    )
    for name, fixture in [
        ("traffic-model", "traffic_model"),
        ("neural-model", "neural_model"),
        ("ensemble-model", "ensemble_model"),
    ]:
        schema = schemas[name]
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, registry=registry)
        validator.validate(
            json.loads((root / "go/testdata/conformance/v1" / f"{fixture}.json").read_text())
        )
