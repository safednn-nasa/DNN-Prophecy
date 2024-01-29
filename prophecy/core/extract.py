import keras
import pandas as pd
import numpy as np

from pathlib import Path
from keras import backend
from keras.src.engine.keras_tensor import KerasTensor

from prophecy.core.learn import learn_rules
from trustbench.core.dataset import Dataset
from prophecy.data.objects import Settings
from prophecy.utils.misc import sanity_check
from prophecy.core.evaluate import get_eval_labels
from prophecy.core.helpers import (get_all_invariants_val, get_all_invariants, impure_rules,
                                   describe_invariants_all_labels)


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


class RuleExtractor:
    def __init__(self, model: keras.Model, dataset: Dataset, settings: Settings):
        self.model = model
        self.dataset = dataset
        self.settings = settings
        self.labels = {'train': {}, 'val': {}}
        self.fingerprints = {'train': {}, 'val': {}}

    @property
    def train_labels(self):
        if len(self.labels['train']) == 0:
            self.get_labels('train')

        return self.labels['train']

    @property
    def val_labels(self):
        if len(self.labels['val']) == 0:
            self.get_labels('val')

        return self.labels['val']

    @property
    def train_fingerprints(self):
        if len(self.fingerprints['train']) == 0:
            self.get_model_fingerprints('train')

        return self.fingerprints['train']

    @property
    def val_fingerprints(self):
        if len(self.fingerprints['val']) == 0:
            self.get_model_fingerprints('val')

        return self.fingerprints['val']

    def __call__(self, path: Path = None, *args, **kwargs):
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
        print(f"{self.settings.rules.upper()} RULES:")
        print(f"Invoking Dec-tree classifier based on {self.settings.fingerprint.upper()}.")

        # TODO: fix this
        if self.settings.fingerprint == 'features':
            fingerprints_tr = {_l: _f[self.settings.fingerprint] for _l, _f in self.train_fingerprints.items()}
            fingerprints_tst = [_f[self.settings.fingerprint] for _l, _f in self.val_fingerprints.items()]
        else:
            fingerprints_tr = {_l: _f[self.settings.fingerprint] for _l, _f in self.train_fingerprints.items() if _l != 'input'}
            fingerprints_tst = [_f[self.settings.fingerprint] for _l, _f in self.val_fingerprints.items() if _l != 'input']

        learners = learn_rules(labels=self.train_labels[self.settings.rules], fingerprints=fingerprints_tr,
                               activations=self.settings.fingerprint == 'activations', save_path=path)

        results = {}
        is_mis = True if self.settings.rules == 'accuracy' else False

        for i, (layer, learner) in enumerate(learners.items(), 1):
            # TODO: get the tree and for every input just call predict and get the output
            print(f"\nRULES FROM LAYER {layer.upper()} IN TERMS OF {self.settings.fingerprint.upper()}\n")
            invariants = get_all_invariants_val(learner) if self.settings.fingerprint == 'features' else \
                get_all_invariants(learner)
            print(f"InV {i-1}")
            impure_rules(invariants)

            desc = describe_invariants_all_labels(invariants, i, list(fingerprints_tr.values()), fingerprints_tst,
                                                  self.train_labels[self.settings.rules],
                                                  self.val_labels[self.settings.rules], ALL=True, MIS=is_mis)
            results[layer] = desc

            #if not sanity_check(desc, learner):
            #    raise ValueError(f"Sanity check failed for layer {layer}. "
            #                     f"#rules: {len(desc)} | #leaves: {learner.get_n_leaves()}")

        return results

    def get_labels(self, split: str):
        """
            Collect the labels for the rules
        :param split: train/val
        :return:
        """

        eval_labels = get_eval_labels(self.model, self.dataset, split)

        print(f"{split.upper()} LABELS:", eval_labels.shape)

        self.labels[split]['accuracy'] = np.array([]).astype(int)
        self.labels[split]['decision'] = np.array([]).astype(int)

        match_count = 0
        mismatch_count = 0

        for idx in range(0, len(eval_labels)):
            # if eval_labels[idx] == int(dataset.splits[split].labels.iloc[idx]['label']):
            if eval_labels[idx] == self.dataset.splits[split].labels[idx]:
                match_count = match_count + 1
                self.labels[split]['decision'] = np.append(self.labels[split]['decision'], eval_labels[idx])
                self.labels[split]['accuracy'] = np.append(self.labels[split]['accuracy'], 0)
            else:
                mismatch_count = mismatch_count + 1
                self.labels[split]['decision'] = np.append(self.labels[split]['decision'], 1000)
                self.labels[split]['accuracy'] = np.append(self.labels[split]['accuracy'], 1000)  # Misclassified

        print(f"{split.upper()} ACCURACY:", (match_count / (match_count + mismatch_count)) * 100.0)

    def get_model_fingerprints(self, split: str):
        """
            Collect the neuron values and activations of Train Data and Test Data after each layer
        :return:
        """

        # check split is valid
        if split not in self.fingerprints:
            raise ValueError(f"Invalid split {split}")

        resize_shape = self.dataset.get_resize_shape()

        if resize_shape:
            features = self.dataset.splits[split].resize(resize_shape)
        else:
            features = self.dataset.splits[split].features.to_numpy()

        self.fingerprints[split]['input'] = {'activations': None, 'features': features}

        for layer in self.model.layers:

            if resize_shape:
                input_features = self.dataset.splits[split].resize(resize_shape)
            else:
                input_features = self.dataset.splits[split].features.to_numpy()

            activations, features = get_layer_fingerprint(self.model.input, layer, input_features)
            self.fingerprints[split][layer.name] = {'activations': activations, 'features': features}
            print(f"Fingerprint after {layer.name}. ({activations.shape} inputs, {features.shape} neurons)")
