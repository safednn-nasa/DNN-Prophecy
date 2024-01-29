import json
from collections import Counter
from typing import Any

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from pathlib import Path
from prophecy.utils.paths import models_path, settings_path
from prophecy.data.objects import Settings


RULES = ['decision', 'accuracy']
FINGERPRINTS = ['features', 'activations']


def get_model(model_path: Path) -> keras.Model:
    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


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


def sanity_check(ruleset: list, clf: DecisionTreeClassifier, by_class: bool = False) -> bool:
    # total_rules == total_leaves
    if by_class:
        # TODO: fix this as it is not working properly
        leaves_counter = Counter([clf.classes_[np.argmax(leaf_values)] for leaf_values in clf.tree_.value])

        # filter leaves_values by target class
        rules_counter = Counter([rule['label'] for rule in ruleset])

        for label, count in rules_counter.items():
            if count != leaves_counter[label]:
                print(f"Label {label} has {count} rules and {leaves_counter[label]} leaves")
                return False

        return True

    return len(ruleset) == clf.get_n_leaves()
