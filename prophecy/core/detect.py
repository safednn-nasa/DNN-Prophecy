from abc import abstractmethod

import keras
import pandas as pd
import pickle

from ast import literal_eval
from typing import Tuple
from tqdm import tqdm
from pathlib import Path

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
        corr_cnt, corr_cover, found = self.eval_rules(index, self.correct_rules)
        inc_cnt, inc_cover, found = self.eval_rules(index, self.incorrect_rules)

        # print("INPUT:", inp_indx , "CORR CNT:", corr_cnt, "INCORR CNT:", inc_cnt)
        if corr_cnt == inc_cnt:
            # print("UNCERTAIN:")
            evaluation.uncertain += 1
            # if self.dataset.splits['unseen'].labels[inp_idx] == labels[inp_idx]:
            #    false_neg_cor = false_neg_cor + 1
            #    true_neg_inc = true_neg_inc + 1
            # else:
            #    false_neg_inc = false_neg_inc + 1
            #    true_neg_cor = true_neg_cor + 1
        # TODO: the evaluation should be done with sklearn
        if corr_cnt > inc_cnt:
            # print("CORRECT")
            evaluation.tot_corr += 1
            evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                       pred_label=self.predictions.labels[index], is_pos=True)

        if inc_cnt > corr_cnt:
            # print("INCORRECT")
            evaluation.tot_inc += 1
            evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                       pred_label=self.predictions.labels[index], is_pos=False)

        # save whether the sample was covered or not
        evaluation.outputs.append(1) if corr_cover or inc_cover else evaluation.outputs.append(0)

    def stats(self, evaluation: Evaluation) -> dict:
        true_covered = sum(evaluation.outputs) - evaluation.uncertain

        return {
            "covered": true_covered,
            "coverage": round((true_covered / len(self.dataset.splits['unseen'].features)) * 100.0, 2)
        }

    def eval_rules(self, inp_idx: int, ruleset: pd.DataFrame) -> Tuple[int, bool, bool]:
        counter = 0
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
                    counter += 1
                    break

        return counter, cover, found


class ClassifierDetector(BaseDetector):
    def __init__(self, learners_path: Path, **kw):
        super().__init__(name="classifier", **kw)
        self.learners_path = learners_path
        self._classifiers = {}

    @property
    def classifiers(self):
        if self._classifiers == {}:
            for classifier_path in self.learners_path.iterdir():
                if classifier_path.is_file() and classifier_path.suffix == '.pkl':
                    self._classifiers[classifier_path.stem] = pickle.load(open(classifier_path, 'rb'))

        return self._classifiers

    def eval(self, evaluation: Evaluation, index: int, row: pd.Series):
        corr_cnt, inc_cnt = self.eval_classifiers(index, row)

        if corr_cnt == inc_cnt:
            # print("UNCERTAIN:")
            evaluation.uncertain += 1

        if corr_cnt > inc_cnt:
            # print("CORRECT")
            evaluation.tot_corr += 1
            evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                       pred_label=self.predictions.labels[index], is_pos=True)

        if inc_cnt > corr_cnt:
            # print("INCORRECT")
            evaluation.tot_inc += 1
            evaluation(true_label=self.dataset.splits['unseen'].labels[index],
                       pred_label=self.predictions.labels[index], is_pos=False)

    def eval_classifiers(self, inp_idx: int, row: pd.Series) -> Tuple[int, int]:
        corr_cnt = 0
        inc_cnt = 0

        for layer, classifier in self.classifiers.items():
            if layer == 'input':
                pred_label = classifier.predict([row.to_numpy()])
            else:
                _, _, op = self.model_rep[layer]
                pred_label = classifier.predict([op[0][inp_idx]])
                #print(classifier.decision_path([op[0][inp_idx]]))
                #print(classifier.apply([op[0][inp_idx]]))

            #label_type = type(self.dataset.splits['unseen'].labels[inp_idx])

            #if type(pred_label[0]) is not label_type:
            #    raise TypeError(f"Predicted label {pred_label[0]} is not of type {label_type}")
            #print(pred_label[0], self.dataset.splits['unseen'].labels[inp_idx])
            #if pred_label[0] == self.dataset.splits['unseen'].labels[inp_idx]:
            #    corr_cnt += 1
            #else:
            #    inc_cnt += 1

            if pred_label[0] == 0:
                corr_cnt += 1
            else:
                inc_cnt += 1

        return corr_cnt, inc_cnt

    def stats(self, evaluation: Evaluation) -> dict:
        return {}
