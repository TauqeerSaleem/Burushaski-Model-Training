from pathlib import Path

import yaml


def read_registry(path="configs/model_registry.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_specs(kind, names=None, registry_path="configs/model_registry.yaml"):
    registry = read_registry(registry_path)
    specs = registry.get(kind, {})
    if names:
        wanted = set(names)
        specs = {name: spec for name, spec in specs.items() if name in wanted}
    if not specs:
        raise ValueError(f"No {kind} models selected from {Path(registry_path)}")
    return specs

