import sys
from abc import abstractmethod

import keras
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from ast import literal_eval
from typing import Tuple, Union
from tqdm import tqdm
from pathlib import Path

from prophecy.core.helpers import check_pattern


class RulesProve:
    #RulesProve(model=model, onnx_model_nm=onnx_model, layer_nm = top_rule_layer, neurons=rule_neurons_list, sig=rule_sig_list,features=train_features, labels=train_labels)
    def __init__(self, model: keras.Model, onnx_model: str, layer_nm: str, neurons: list(), sig: list(), features: pd.DataFrame, labels: np.ndarray):
        self.model = model
        self.onnx_path = onnx_model
        self.layer_nm = layer_nm
        self.neurons = neurons
        self.sig = sig
        self.features = features
        self.labels = labels
        

    def __call__(self, **kwargs) -> list:
        results = []

        if isinstance(self.features, pd.DataFrame):
            iterator = self.features.iterrows()
        else:
            iterator = enumerate(self.features)

        for inp_idx, sample in tqdm(iterator, file=sys.stdout):
            outcome = self.eval(inp_idx, sample)
            results.append({'idx': inp_idx, 'outcome': outcome})

        return results

    @abstractmethod
    def eval(self, index: int, row: Union[pd.Series, np.ndarray]):
        pass

    @property
    def model_rep(self):
        if self._model_rep is None:
            self._model_rep = {}
            # Get the model fingerprints

            total_samples = self.features.shape[0]

            if isinstance(self.features, np.ndarray) and total_samples > 5000:
                self.get_batched_model_rep(total_samples)
            else:
                for layer in self.model.layers:
                    if layer.name not in self.target_layers:
                        continue

                    func_dense = keras.backend.function(self.model.input, [layer.output])
                    inp_tensor = keras.backend.constant(self.features)
                    op = func_dense(inp_tensor)
                    self._model_rep[layer.name] = (func_dense, inp_tensor, op)

        return self._model_rep

    def get_batched_model_rep(self, total_samples: int,  batch_size=256):
        for layer in self.model.layers:
            if layer.name not in self.target_layers:
                continue

            layer_inputs = []
            layer_outputs = []
            func_dense = keras.backend.function(self.model.input, [layer.output])

            for start in tqdm(range(0, total_samples, batch_size), desc=f"Processing {layer.name}", file=sys.stdout):
                end = min(start + batch_size, total_samples)
                batch_features = self.features[start:end]
                inp_tensor = keras.backend.constant(batch_features)
                layer_inputs.append(inp_tensor)
                op = func_dense(inp_tensor)
                layer_outputs.append(op)

            self._model_rep[layer.name] = (func_dense, tf.concat(layer_inputs, axis=0),
                                           np.concatenate(layer_outputs, axis=1))


class RulesDetector(BaseDetector):
    def __init__(self, ruleset: pd.DataFrame, **kw):
        print("DETECT CORRECT, INCORRECT, UNCERTAIN on UNSEEN DATA")
        super().__init__(name="rules", **kw)
        # parse the ruleset
        ruleset['neurons'] = ruleset['neurons'].apply(literal_eval)
        ruleset['signature'] = ruleset['signature'].apply(literal_eval)
        self._target_layers = list(ruleset['layer'].unique())

        self.correct_rules = ruleset[ruleset['kind'] == 'correct']
        self.incorrect_rules = ruleset[ruleset['kind'] == 'incorrect']

    @property
    def target_layers(self):
        return self._target_layers

    def eval(self, index: int, row: Union[pd.Series, np.ndarray]):
        # print(sample.to_list())
        corr_layer, corr_cover, found = self.eval_rules(index, self.correct_rules)
        inc_layer, inc_cover, found = self.eval_rules(index, self.incorrect_rules)
        corr_cnt = len(corr_layer)
        inc_cnt = len(inc_layer)
        stats = {'idx': index, 'corr': corr_cnt, 'inc': inc_cnt, 'corr_cover': corr_cover, 'inc_cover': inc_cover,
                 'corr_layer': corr_layer, 'inc_layer': inc_layer}

        if corr_cnt == inc_cnt:
            stats['eval'] = 'uncertain'
        elif corr_cnt > inc_cnt:
            stats['eval'] = 'correct'
        elif inc_cnt > corr_cnt:
            stats['eval'] = 'incorrect'

        self.stats.append(stats)

        return stats['eval']

    def eval_rules(self, inp_idx: int, ruleset: pd.DataFrame) -> Tuple[list, bool, bool]:
        layers = []
        cover = False
        found = False

        for layer, rows in ruleset.groupby('layer', sort=False):
            found = False
            for i, row in rows.iterrows():
                # TODO: check if this applies for other settings
                func_dense, inp_tensor, op = self.model_rep['dense'] if layer == 'input' else self.model_rep[layer]
                found = check_pattern(op[0][inp_idx], row['neurons'], row['signature'])

                if found:
                    cover = True
                    layers.append(layer)
                    break

        return layers, cover, found


class ClassifierDetector(BaseDetector):
    def __init__(self, learners_path: Path, only_pure: bool = False, **kw):
        super().__init__(name="classifier", **kw)
        self.learners_path = learners_path
        self._classifiers = {}
        self._only_pure = only_pure

    @property
    def target_layers(self):
        if len(self._target_layers) == 0:
            self._target_layers = list(self.classifiers.keys())

        return self._target_layers

    @property
    def classifiers(self):
        if self._classifiers == {}:
            for classifier_path in self.learners_path.iterdir():
                if classifier_path.is_file() and classifier_path.suffix == '.pkl':
                    self._classifiers[classifier_path.stem] = pickle.load(open(classifier_path, 'rb'))

        return self._classifiers

    def eval(self, index: int, row: pd.Series):
        corr_layer, inc_layer = self.eval_classifiers(index, row)
        corr_cnt = len(corr_layer)
        inc_cnt = len(inc_layer)
        stats = {'idx': index, 'corr': corr_cnt, 'inc': inc_cnt, 'corr_layer': corr_layer, 'inc_layer': inc_layer}

        if corr_cnt == inc_cnt:
            stats['eval'] = 'uncertain'

        elif corr_cnt > inc_cnt:
            stats['eval'] = 'correct'

        elif inc_cnt > corr_cnt:
            stats['eval'] = 'incorrect'

        self.stats.append(stats)

        return stats['eval']

    def eval_classifiers(self, inp_idx: int, features: Union[pd.Series, np.ndarray]) -> Tuple[list, list]:
        corr_layer = []
        inc_layer = []

        for layer, classifier in self.classifiers.items():
            #print("LAYER:", layer.name)
            predict_method = classifier.predict_proba if self._only_pure else classifier.predict

            if layer == 'input':
                if isinstance(features, pd.Series):
                    prediction = predict_method([features.to_numpy()])
                else:
                    if len(features.shape) > 2:
                        prediction = predict_method([features.flatten()])
                    else:
                        prediction = predict_method([features])
            else:
                _, _, op = self.model_rep[layer]

                if len(op[0][inp_idx].shape) > 2:
                    prediction = predict_method([op[0][inp_idx].flatten()])
                else:
                    prediction = predict_method([op[0][inp_idx]])

            if self._only_pure:
                if prediction[0][0] not in [0.0, 1.0]:
                    print(f"Uncertain prediction for layer {layer} with probabilities {prediction[0]}")
                    continue
                # get the class with the highest probability
                label = classifier.classes_[np.argmax(prediction[0])]
            else:
                label = prediction[0]

            corr_layer.append(layer) if label == 0 else inc_layer.append(layer)

        return corr_layer, inc_layer
