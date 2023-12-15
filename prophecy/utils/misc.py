import json

from tensorflow import keras
from pathlib import Path
from prophecy.utils.paths import models_path, settings_path, datasets_path
from prophecy.data.objects import Settings


RULES = ['decision', 'accuracy']
FINGERPRINTS = ['features', 'activations']


def get_model(model_path: Path) -> keras.Model:
    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


def lookup_datasets():
    datasets = {file.stem: file for file in datasets_path.iterdir() if file.is_dir()}

    if len(datasets) == 0:
        raise ValueError(f"No datasets found in {datasets_path} directory")

    return datasets


def lookup_models():
    models = {file.stem: file for file in models_path.iterdir() if file.suffix == '.h5'}

    if len(models) == 0:
        raise ValueError(f"No models found in {models_path} directory")

    return models


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
