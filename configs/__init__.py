from types import SimpleNamespace
import yaml


def load_yaml(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)
