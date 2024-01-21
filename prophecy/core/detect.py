from abc import abstractmethod

import keras
import numpy as np
import pandas as pd
import pickle

from ast import literal_eval
from typing import Tuple
from tqdm import tqdm
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from prophecy.data.dataset import Dataset
from prophecy.data.objects import Predictions, Evaluation
from prophecy.core.evaluate import predict_unseen
from prophecy.core.helpers import check_pattern


class BaseDetector:
    def __init__(self, name: str, model: keras.Model, dataset: Dataset):
        self.name = name
        self.model = model
        self.dataset = dataset
        self._model_rep = None
        self._predictions = None

    def __call__(self, **kwargs) -> dict:
        evaluation = Evaluation()

        for inp_idx, sample in tqdm(self.dataset.splits['unseen'].features.iterrows()):
            self.eval(evaluation, inp_idx, sample)

        results = {
            "unseen_correct": self.predictions.correct,
            "unseen_incorrect": self.predictions.incorrect
        }
        results.update(evaluation.to_dict())
        results.update(self.stats(evaluation))
        results.update(evaluation.performance())

        return results

    @abstractmethod
    def eval(self, evaluation: Evaluation, index: int, row: pd.Series):
        pass

    @abstractmethod
    def stats(self, evaluation: Evaluation) -> dict:
        pass

    @property
    def predictions(self) -> Predictions:
        if self._predictions is None:
            self._predictions = predict_unseen(self.model, self.dataset.splits['unseen'])

        return self._predictions

    @property
    def model_rep(self):
        if self._model_rep is None:
            self._model_rep = {}
            # Get the model fingerprints

            for layer in self.model.layers:
                func_dense = keras.backend.function(self.model.input, [layer.output])
                inp_tensor = keras.backend.constant(self.dataset.splits['unseen'].features)
                op = func_dense(inp_tensor)
                self._model_rep[layer.name] = (func_dense, inp_tensor, op)

        return self._model_rep


class RulesDetector(BaseDetector):
    def __init__(self, ruleset: pd.DataFrame, **kw):
        print("DETECT CORRECT, INCORRECT, UNCERTAIN on UNSEEN DATA")
        super().__init__(name="rules", **kw)
        # parse the ruleset
        ruleset['neurons'] = ruleset['neurons'].apply(literal_eval)
        ruleset['signature'] = ruleset['signature'].apply(literal_eval)

        self.correct_rules = ruleset[ruleset['kind'] == 'correct']
        self.incorrect_rules = ruleset[ruleset['kind'] == 'incorrect']

    def eval(self, evaluation: Evaluation, index: int, row: pd.Series):
        # print(sample.to_list())
        corr_layer, corr_cover, found = self.eval_rules(index, self.correct_rules)
        inc_layer, inc_cover, found = self.eval_rules(index, self.incorrect_rules)
        corr_cnt = len(corr_layer)
        inc_cnt = len(inc_layer)
        stats = {'idx': index, 'corr': corr_cnt, 'inc': inc_cnt, 'corr_layer': corr_layer, 'inc_layer': inc_layer}

        # print("INPUT:", inp_indx , "CORR CNT:", corr_cnt, "INCORR CNT:", inc_cnt)
        if corr_cnt == inc_cnt:
            stats['eval'] = 'uncertain'
            evaluation.uncertain += 1
            # if self.dataset.splits['unseen'].labels[inp_idx] == labels[inp_idx]:
            #    false_neg_cor = false_neg_cor + 1
            #    true_neg_inc = true_neg_inc + 1
            # else:
            #    false_neg_inc = false_neg_inc + 1
            #    true_neg_cor = true_neg_cor + 1
        # TODO: the evaluation should be done with sklearn
        if corr_cnt > inc_cnt:
            stats['eval'] = 'correct'
            evaluation.tot_corr += 1
            pred = evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                              pred_label=self.predictions.labels[index], is_pos=True)
            stats['pred'] = pred

        if inc_cnt > corr_cnt:
            stats['eval'] = 'incorrect'
            evaluation.tot_inc += 1
            pred = evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                              pred_label=self.predictions.labels[index], is_pos=False)
            stats['pred'] = pred

        print(stats)
        # save whether the sample was covered or not
        evaluation.outputs.append(1) if corr_cover or inc_cover else evaluation.outputs.append(0)

    def stats(self, evaluation: Evaluation) -> dict:
        true_covered = sum(evaluation.outputs) - evaluation.uncertain

        return {
            "covered": true_covered,
            "coverage": round((true_covered / len(self.dataset.splits['unseen'].features)) * 100.0, 2)
        }

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
    def classifiers(self):
        if self._classifiers == {}:
            for classifier_path in self.learners_path.iterdir():
                if classifier_path.is_file() and classifier_path.suffix == '.pkl':
                    self._classifiers[classifier_path.stem] = pickle.load(open(classifier_path, 'rb'))

        return self._classifiers

    def eval(self, evaluation: Evaluation, index: int, row: pd.Series):
        corr_layer, inc_layer = self.eval_classifiers(index, row)
        corr_cnt = len(corr_layer)
        inc_cnt = len(inc_layer)
        stats = {'idx': index, 'corr': corr_cnt, 'inc': inc_cnt, 'corr_layer': corr_layer, 'inc_layer': inc_layer}

        if corr_cnt == inc_cnt:
            stats['eval'] = 'uncertain'
            evaluation.uncertain += 1

        if corr_cnt > inc_cnt:
            stats['eval'] = 'correct'
            evaluation.tot_corr += 1
            pred = evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                              pred_label=self.predictions.labels[index], is_pos=True)
            stats['pred'] = pred

        if inc_cnt > corr_cnt:
            stats['eval'] = 'incorrect'
            evaluation.tot_inc += 1
            pred = evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                              pred_label=self.predictions.labels[index], is_pos=False)
            stats['pred'] = pred

        print(stats)

    def eval_classifiers(self, inp_idx: int, row: pd.Series) -> Tuple[list, list]:
        corr_layer = []
        inc_layer = []

        for layer, classifier in self.classifiers.items():
            predict_method = classifier.predict_proba if self._only_pure else classifier.predict

            if layer == 'input':
                prediction = predict_method([row.to_numpy()])
            else:
                _, _, op = self.model_rep[layer]
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

    def stats(self, evaluation: Evaluation) -> dict:
        return {}
