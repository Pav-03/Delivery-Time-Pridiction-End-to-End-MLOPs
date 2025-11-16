import yaml
from pathlib import Path

def load_params( config_path: Path = None) ->dict:
    """load params.yaml with validation."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2]/"params.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f" params.yaml not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # validation-catches error early
    required = ["seed","users","restaurants"]
    missing = [key for key in required if key not in params]

    if missing:
        raise KeyError(f" params.yaml missing required keys: {missing}")
    
    return params