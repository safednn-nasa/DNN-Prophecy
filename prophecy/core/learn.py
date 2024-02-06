import numpy as np
import pickle

from sklearn import tree
from typing import Dict
from pathlib import Path


def learn_rules(labels: np.array, fingerprints: dict, activations: bool, save_path: Path = None) \
        -> Dict[str, tree.DecisionTreeClassifier]:
    """
        Run Dec-Tree Learning using Train Data to learn rules after each layer in terms of neuron features of
        activations

    :param labels: list of labels
    :param fingerprints: dict of fingerprints
    :param activations: boolean indicating whether to use activations or features
    :param save_path: save path for the resulting estimator
    :return:
    """

    classifiers = {}

    if activations:
        # skip input layer
        # TODO: why input rules are learned only for features?
        fingerprints = {layer: fingerprint for layer, fingerprint in fingerprints.items() if layer != 'input'}

    for layer, fingerprint in fingerprints.items():
        print("Inputs: (neuron signature (On/Off activations) dataset)(labels dataset)")

        #for output, values in fingerprint.items():
        print(fingerprint.shape, labels.shape)

        if len(fingerprint.shape) > 2:
            print(f"Reshaping fingerprint for layer {layer}")
            fingerprint = np.array([x.flatten() for x in fingerprint])

        #print(f"Invoking Dec-tree classifier based on neuron {output}. for layer {layer}")
        basic_estimator = tree.DecisionTreeClassifier()
        basic_estimator.fit(fingerprint, labels)
        #classifiers[f"{layer}_{output}"] = basic_estimator
        classifiers[layer] = basic_estimator

        if save_path and save_path.exists():
            model_path = save_path / f"{layer}.pkl"

            with model_path.open(mode='wb') as mf:
                pickle.dump(basic_estimator, mf)

    return classifiers
