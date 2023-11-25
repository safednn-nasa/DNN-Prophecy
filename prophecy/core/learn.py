import numpy as np

from sklearn import tree
from typing import Dict

from prophecy.data.dataset import Dataset


def learn_val_rules(labels: np.array, dataset: Dataset) -> tree.DecisionTreeClassifier:
    """
        Run Dec-Tree Learning using Train Data to learn rules after each layer in terms of neuron values

    :param labels: np.array of decision/accuracy labels
    :param dataset: Dataset
    :return:
    """

    print("Inputs: (neuron signature (Values) dataset)(labels dataset)")
    fingerprint_inp = dataset.splits['train'].features.to_numpy()
    print(fingerprint_inp.shape, labels.shape)
    basic_estimator = tree.DecisionTreeClassifier()
    basic_estimator.fit(fingerprint_inp, labels)

    return basic_estimator


def learn_act_rules(dec_labels: list, fingerprints: dict) -> Dict[str, tree.DecisionTreeClassifier]:
    """
        Run Dec-Tree Learning using Train Data to learn rules after each layer in terms of neuron activations

    :param dec_labels: list of labels
    :param fingerprints: dict of fingerprints
    :return:
    """

    dec_labels = np.array(dec_labels)
    classifiers = {}

    for layer, fingerprint in fingerprints.items():
        print("Inputs: (neuron signature (On/Off activations) dataset)(labels dataset)")

        for output, values in fingerprint.items():
            print(values.shape, dec_labels.shape)
            print(f"Invoking Dec-tree classifier based on neuron {output}. for layer {layer}")
            basic_estimator = tree.DecisionTreeClassifier()
            basic_estimator.fit(values, dec_labels)
            classifiers[f"{layer}_{output}"] = basic_estimator

    return classifiers
