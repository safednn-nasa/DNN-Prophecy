import keras
import pandas as pd
import numpy as np

from typing import Tuple
from keras import backend
from keras.src.engine.keras_tensor import KerasTensor

from prophecy.data.dataset import Dataset
from prophecy.core.evaluate import get_eval_labels


def extract_rules(model: keras.Model, dataset: Dataset, split: str) -> Tuple[np.array, np.array]:
    """
        Rule Extraction from every layer (input through output).
        Each rule is of the form pre(x) = > P(F(x)), pre: neuron constraints at the chosen layer.
        P is a property of the output of model.

        Three types of Rules:
            1. Decision Rules: P(F(x)) is true iff F(x) = L
            2. Rules for correct behavior: P(F(x)) is true iff F(x) = L and L = L_ideal.
            3. Rules for incorrect behavior: P(F(x)) is true iff F(x) != L_ideal.
    :return:
    """

    eval_labels = get_eval_labels(model, dataset, split)
    # TODO: for test in the previous code the eval_labels are computed with (model.predict(x_val)).argmax(axis=1)
    # that yields different results than the get_eval_labels function
    accuracy_labels = []
    decision_labels = []
    print(f"{split.upper()} LABELS:", eval_labels.shape)
    match_count = 0
    mismatch_count = 0

    for idx in range(0, len(eval_labels)):
        # if eval_labels[idx] == int(dataset.splits[split].labels.iloc[idx]['label']):
        if eval_labels[idx] == dataset.splits[split].labels[idx]:
            match_count = match_count + 1
            decision_labels.append(eval_labels[idx])
            accuracy_labels.append(0)
        else:
            mismatch_count = mismatch_count + 1
            decision_labels.append(1000)
            accuracy_labels.append(1000)  # Misclassified

    print(f"{split.upper()} ACCURACY:", (match_count / (match_count + mismatch_count)) * 100.0)

    return np.array(accuracy_labels), np.array(decision_labels)


def get_layer_fingerprint(model_input: KerasTensor, layer: keras.layers.Layer, features: pd.DataFrame):
    """

    :param model_input: model input
    :param layer: keras.layers.Layer
    :param features: pd.DataFrame
    :return: tuple of (neuron activations, neuron values)
    """
    func = backend.function(model_input, [layer.output])

    x_tensor = backend.constant(features)
    outputs = func(x_tensor)

    return (outputs[0] > 0.0).astype('int'), outputs[0]


def get_model_fingerprints(model: keras.Model, dataset: Dataset, split: str):
    """
        Collect the neuron values and activations of Train Data and Test Data after each layer
    :return:
    """

    # check split is valid
    if split not in ['train', 'val']:
        raise ValueError(f"Invalid split {split}")

    fingerprints = {}

    for layer in model.layers:
        act, val = get_layer_fingerprint(model.input, layer, dataset.splits[split].features)
        fingerprints[layer.name] = {'act': act, 'val': val}
        print(f"Fingerprint after {layer.name}. ({act.shape} inputs, {val.shape} neurons)")

    return fingerprints
