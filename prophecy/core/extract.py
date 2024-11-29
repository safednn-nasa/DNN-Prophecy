from typing import Union

import keras
import sys
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from keras import backend
from keras.src.engine.keras_tensor import KerasTensor

from prophecy.core.learn import learn_rules
from prophecy.core.evaluate import get_eval_labels
from prophecy.core.helpers import (get_all_invariants_val, impure_rules, describe_invariants_all_labels)


def get_layer_fingerprint(model_input: KerasTensor, layer: keras.layers.Layer,
                          features: Union[pd.DataFrame, np.ndarray]):
    """

    :param model_input: model input
    :param layer: keras.layers.Layer
    :param features: pd.DataFrame
    :return: tuple of (neuron activations, neuron values)
    """
    func = backend.function(model_input, [layer.output])

    if features.shape[0] > 5000 and isinstance(features, np.ndarray):
        batch_size = 256
        num_samples = len(features)
        num_batches = int(np.ceil(num_samples / batch_size))

        activations_list = []
        values_list = []

        for i in tqdm(range(num_batches), desc=f"Processing {layer.name}", file=sys.stdout):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            batch_features = features[start_idx:end_idx, :]
            x_tensor = backend.constant(batch_features)
            outputs = func(x_tensor)

            activations_list.append((outputs[0] > 0.0).astype('int'))
            values_list.append(outputs[0])

        return np.concatenate(activations_list, axis=0), np.concatenate(values_list, axis=0)
    else:
        x_tensor = backend.constant(features)
        outputs = func(x_tensor)

        return (outputs[0] > 0.0).astype('int'), outputs[0]


class Extractor:
    def __init__(self, model: keras.Model, train_features: pd.DataFrame, train_labels: np.ndarray,
                 val_features: pd.DataFrame, val_labels: np.ndarray, only_dense: bool = False, skip_rules: bool = False,
                 balance: bool = False, only_activation: bool = False, confidence: bool = False, random_state: int = 42, type: int = 1, 
                 inptype: int = 0, acts: bool = False, **kwargs):
        self.model = model
        self.features = {'train': train_features, 'val': val_features}
        self.labels = {'train': train_labels, 'val': val_labels}
        self.clf_labels = {'train': None, 'val': None}
        self.fingerprints = {'train': None, 'val': None}
        self._skip_rules = skip_rules
        self._balance = balance
        self._confidence = confidence
        self.layers = []
        self.only_activation = only_activation
        self.only_dense = only_dense
        self.random_state = random_state
        self.type = type
        self.inptype = inptype
        self.acts = acts
                   
        if only_dense and only_activation:
            print("Dense layers and associated activation layers are considered for fingerprinting")
            include_next = False

            for layer in model.layers:
                if layer.name.startswith('dense'):
                    include_next = True
                    self.layers.append(layer)
                elif layer.name.startswith('activation') and include_next:
                    include_next = False
                    self.layers.append(layer)
        elif only_dense:
            print("Only dense layers are considered for fingerprinting")
            self.layers = [layer for layer in model.layers if 'dense' in layer.name]
        elif only_activation:
            print("Only activation layers of dense layers are considered for fingerprinting")
            self.layers = [layer for layer in model.layers if 'activation' in layer.name]
        else:
            self.layers = model.layers

        print(f"Layers to be considered for fingerprinting: {[layer.name for layer in self.layers]}")

    @property
    def clf_train_labels(self):
        if self.clf_labels['train'] is None:
          if self.type == 1:
            self.get_labels('train')
          if self.type == 0:
            self.get_labels_dec('train')
          if self.type == 2:
            label_arr = self.labels['train']
            self.clf_labels['train'] = label_arr.astype(int)
        return self.clf_labels['train']

    @property
    def clf_val_labels(self):
        if self.clf_labels['val'] is None:
          if self.type == 1:
            self.get_labels('val')
          if self.type == 0:
            self.get_labels_dec('val')
          if self.type == 2:
            label_arr = self.labels['val']
            self.clf_labels['val'] = label_arr.astype(int)
        return self.clf_labels['val']

    @property
    def train_fingerprints(self):
        if self.fingerprints['train'] is None:
            self.get_model_fingerprints('train')

        return self.fingerprints['train']

    @property
    def val_fingerprints(self):
        if self.fingerprints['val'] is None:
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

        #if self._balance or self._confidence:
        #self.get_labels('train')

        if self.type == 1:
          self.get_labels('train')
        if self.type == 0:
          self.get_labels_dec('train')
        if self.type == 2:
          label_arr = self.labels['train']
          self.clf_labels['train'] = label_arr.astype(int)

        print(f"Invoking Dec-tree classifier based on FEATURES")
      
        fingerprints_tr = {_l: _f for _l, _f in self.train_fingerprints.items()}

        learners = learn_rules(labels=self.clf_train_labels, fingerprints=fingerprints_tr,
                               activations=self.acts, save_path=path, random_state=self.random_state)


        if self._skip_rules:
            return {}

        return self._extract(learners, list(fingerprints_tr.values()))

    def _extract(self, learners: dict, fingerprints_tr: list):
        results = []
      
        print("FINGERPRINTS:")
        for indx in range(0, 50):
          print(fingerprints_tr[0][indx])
      
        for layer_count, (layer_name, learner) in enumerate(learners.items(), 1):
            # TODO: get the tree and for every input just call predict and get the output
            print(f"\nRULES FROM LAYER {layer_name.upper()} IN TERMS OF FEATURES\n")
            invariants = get_all_invariants_val(learner)
            print(f"InV {layer_count-1}")
            impure_rules(invariants)

            desc = describe_invariants_all_labels(invariants, layer_count, layer_name, fingerprints_tr,
                                                  list(self.val_fingerprints.values()), self.clf_train_labels,
                                                  self.clf_val_labels, ALL=True, MIS=True)
            results.extend(desc)

        return results
   
          
    def get_labels_dec(self, split: str):
        """
        Collect the labels for the rules
        :param split: train/val
        :return:
        """
        # TODO: this method contains code for confidence based filtering of labels which was disabled
        # eval_labels, confidences = get_eval_labels(self.model, self.features[split], split_name=split)
        eval_labels = get_eval_labels(self.model, self.features[split], split_name=split)

         # Initialize empty lists for decision and accuracy
        accuracy_list = []

        match_count = 0
        mismatch_count = 0

        # Iterate over labels and confidence together using enumerate
        # for idx, (label, confidence) in enumerate(zip(eval_labels, confidences)):
        for idx, label in enumerate(eval_labels):
            # Default values for decision and accuracy is misclassified
            accuracy = 1000

            # Check if confidence is within the specified range
            # if self._confidence and confidence < minimum_confidence:
            #     mismatch_count += 1
            #     pass  # Leave decision and accuracy as default
            if label == self.labels[split][idx]:
                match_count += 1
                accuracy = label
            else:
                mismatch_count += 1
                pass  # Leave decision and accuracy as default

            accuracy_list.append(accuracy)

        self.clf_labels[split] = np.array(accuracy_list).astype(int)

        print(f"{split.upper()} ACCURACY:", (match_count / (match_count + mismatch_count)) * 100.0)
        
        
      
        # get the number of samples in each class
        unique, counts = np.unique(self.clf_labels[split], return_counts=True)
        print(f"{split.upper()} LABELS COUNT:", dict(zip(unique, counts)))

        # randomly drop labels to balance the classes
        if split == 'train' and self._balance:
            print(f"Balancing {split.upper()} labels")
            counts = dict(zip(unique, counts))
            print("Counts:", counts)
            # get the class with the maximum number of samples
            max_class = max(counts, key=counts.get)
            # get the class with the minimum number of samples
            min_class = min(counts, key=counts.get)
            # get the indexes of the samples in the maximum class
            max_class_idx = np.where(self.clf_labels[split] == max_class)[0]
            min_class_idx = np.where(self.clf_labels[split] == min_class)[0]

            selected_max_class_idx = np.random.choice(max_class_idx, size=counts[min_class], replace=False)
            new_ids = np.append(selected_max_class_idx, min_class_idx)
            print("Length of new_ids:", len(new_ids))
            self.clf_labels[split] = self.clf_labels[split][new_ids]
            self.features[split] = self.features[split][new_ids]

  
    def get_labels(self, split: str):
        """
            Collect the labels for the rules
        :param split: train/val
        :return:
        """
        # TODO: this method contains code for confidence based filtering of labels which was disabled
        # eval_labels, confidences = get_eval_labels(self.model, self.features[split], split_name=split)
        eval_labels = get_eval_labels(self.model, self.features[split], split_name=split)

        # compute outliers by looking at the confidence
        # minimum_confidence = Q1 - 1.5 * IQR
        # q1 = np.percentile(confidences, 25)
        # q3 = np.percentile(confidences, 75)
        # iqr = q3 - q1
        # minimum_confidence = q1 - 1.5 * iqr

        print(f"{split.upper()} LABELS:", eval_labels.shape)

        # Initialize empty lists for decision and accuracy
        accuracy_list = []

        match_count = 0
        mismatch_count = 0

        # Iterate over labels and confidence together using enumerate
        # for idx, (label, confidence) in enumerate(zip(eval_labels, confidences)):
        for idx, label in enumerate(eval_labels):
            # Default values for decision and accuracy is misclassified
            accuracy = 1000

            # Check if confidence is within the specified range
            # if self._confidence and confidence < minimum_confidence:
            #     mismatch_count += 1
            #     pass  # Leave decision and accuracy as default
            if label == self.labels[split][idx]:
                match_count += 1
                accuracy = 0
            else:
                mismatch_count += 1
                pass  # Leave decision and accuracy as default

            accuracy_list.append(accuracy)

        self.clf_labels[split] = np.array(accuracy_list).astype(int)

        print(f"{split.upper()} ACCURACY:", (match_count / (match_count + mismatch_count)) * 100.0)
        # get the number of samples in each class
        unique, counts = np.unique(self.clf_labels[split], return_counts=True)
        print(f"{split.upper()} LABELS COUNT:", dict(zip(unique, counts)))

        # randomly drop labels to balance the classes
        if split == 'train' and self._balance:
            print(f"Balancing {split.upper()} labels")
            counts = dict(zip(unique, counts))
            print("Counts:", counts)
            # get the class with the maximum number of samples
            max_class = max(counts, key=counts.get)
            # get the class with the minimum number of samples
            min_class = min(counts, key=counts.get)
            # get the indexes of the samples in the maximum class
            max_class_idx = np.where(self.clf_labels[split] == max_class)[0]
            min_class_idx = np.where(self.clf_labels[split] == min_class)[0]

            selected_max_class_idx = np.random.choice(max_class_idx, size=counts[min_class], replace=False)
            new_ids = np.append(selected_max_class_idx, min_class_idx)
            print("Length of new_ids:", len(new_ids))
            self.clf_labels[split] = self.clf_labels[split][new_ids]
            self.features[split] = self.features[split][new_ids]

    def get_model_fingerprints(self, split: str):
        """
            Collect the neuron values and activations of Train Data and Test Data after each layer
        :return:
        """

        # check split is valid
        if split not in self.fingerprints:
            raise ValueError(f"Invalid split {split}")

        if not (self.only_activation or self.only_dense):
            if isinstance(self.features[split], pd.DataFrame):
                self.fingerprints[split] = {'input': self.features[split].to_numpy()}
            else:
                self.fingerprints[split] = {'input': self.features[split]}
        else:
            self.fingerprints[split] = {}

        if (self.inptype == 1):
            self.fingerprints[split]['current'] = self.features
            return
          
        for layer in self.layers:
            print(f"\nFingerprinting {split.upper()} data after {layer.name} layer")
            activations, features = get_layer_fingerprint(self.model.input, layer, self.features[split])
            if (self.acts == False):
              self.fingerprints[split][layer.name] = features
            else:
              self.fingerprints[split][layer.name] = activations
            print(f"Fingerprint after {layer.name}. ({activations.shape} inputs, {features.shape} neurons)")
