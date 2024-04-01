from collections import Counter

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from pathlib import Path


def get_model(model_path: str) -> keras.Model:
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


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


def read_split(x: str, y: str):
    x_path = Path(x)
    y_path = Path(y)

    if not x_path.exists():
        raise ValueError(f"{x_path} does not exist")

    if not y_path.exists():
        raise ValueError(f"{y_path} does not exist")

    if x_path.suffix == '.npy':
        x_data = np.load(x_path)
    elif x_path.suffix == '.csv':
        # TODO: add case for files with no headers
        x_data = pd.read_csv(x_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {x_path.suffix}")

    if y_path.suffix == '.npy':
        y_data = np.load(y_path)
    elif y_path.suffix == '.csv':
        # TODO: add case for files with no headers
        # TODO: does not handle the case where the labels are not integers
        y_data = pd.read_csv(y_path, delimiter=',').to_numpy()
    else:
        raise ValueError(f"Unsupported file format: {y_path.suffix}")

    return x_data, y_data
