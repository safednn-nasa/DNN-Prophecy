import numpy as np

from sklearn import tree
from typing import Dict

from prophecy.data.dataset import Dataset


def get_val_rules(dec_labels: list, dataset: Dataset) -> tree.DecisionTreeClassifier:
    """
        Run Dec-Tree Learning using Train Data to learn rules after each layer in terms of neuron values

    :param dec_labels: list of labels
    :param dataset: Dataset
    :return:
    """

    print("DECISION RULES:")
    dec_labels = np.array(dec_labels)
    print("Invoking Dec-tree classifier based on INPUT FEATURES.")
    print("Inputs: (neuron signature (Values) dataset)(labels dataset)")
    fingerprint_inp = dataset.splits['train'].features.to_numpy()
    print(fingerprint_inp.shape, dec_labels.shape)
    dec_basic_estimator = tree.DecisionTreeClassifier()
    dec_basic_estimator.fit(fingerprint_inp, dec_labels)

    return dec_basic_estimator


def get_act_rules(dec_labels: list, fingerprints: dict) -> Dict[str, tree.DecisionTreeClassifier]:
    """
        Run Dec-Tree Learning using Train Data to learn rules after each layer in terms of neuron activations

    :param dec_labels: list of labels
    :param fingerprints: dict of fingerprints
    :return:
    """

    print("DECISION RULES:")
    dec_labels = np.array(dec_labels)
    classifiers = {}

    for layer, fingerprint in fingerprints.items():
        print(f"Invoking Dec-tree classifier based on neuron activations for layer {layer}")
        print("Inputs: (neuron signature (On/Off activations) dataset)(labels dataset)")

        for output, values in fingerprint.items():
            print(values.shape, dec_labels.shape)
            dec_basic_estimator = tree.DecisionTreeClassifier()
            dec_basic_estimator.fit(values, dec_labels)
            classifiers[f"{layer}_{output}"] = dec_basic_estimator

    return classifiers
