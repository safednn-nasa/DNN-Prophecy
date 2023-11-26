import json

from tensorflow import keras
from pathlib import Path
from prophecy.utils.paths import models_path, settings_path
from prophecy.data.objects import Settings


RULES = ['decision', 'accuracy']
FINGERPRINTS = ['features', 'activations']


def get_model(model: str, version: str) -> keras.Model:
    model_path = models_path / f"{model}{version}.h5"

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


def lookup_settings():
    res = {file.stem: file for file in settings_path.iterdir() if file.suffix == '.json'}

    if len(res) == 0:
        raise ValueError(f"No settings found in {settings_path} directory")

    return res


def load_settings(path: Path) -> Settings:
    with open(path, 'r') as f:
        settings_dict = json.load(f)

    new_dict = {}
    # parse settings dict by looking at fields and values
    for k, v in settings_dict.items():
        if k == 'rules' and v in RULES:
            new_dict[k] = v
        if k == 'fingerprint' and v in FINGERPRINTS:
            new_dict[k] = v

    return Settings(**new_dict)
