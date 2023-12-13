import keras
import pandas as pd

from ast import literal_eval
from typing import Tuple

from prophecy.data.dataset import Dataset
from prophecy.core.evaluate import get_unseen_labels
from prophecy.core.helpers import check_pattern


class Detector:
    def __init__(self, model: keras.Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset
        self._model_rep = {}

    @property
    def model_rep(self):
        if len(self._model_rep) == 0:
            self.get_model_rep()

        return self._model_rep

    def __call__(self, ruleset: pd.DataFrame) -> dict:
        print("DETECT CORRECT, INCORRECT, UNCERTAIN on UNSEEN DATA")

        # parse the ruleset
        ruleset['neurons'] = ruleset['neurons'].apply(literal_eval)
        ruleset['signature'] = ruleset['signature'].apply(literal_eval)
        correct_rules = ruleset[ruleset['kind'] == 'correct']
        incorrect_rules = ruleset[ruleset['kind'] == 'incorrect']
        labels, tot_corr_unseen, tot_inc_unseen = get_unseen_labels(self.model, self.dataset.splits['unseen'])

        tot_corr = 0
        tot_inc = 0
        uncertain = 0
        true_pos_cor = 0
        false_pos_cor = 0
        true_neg_cor = 0
        false_neg_cor = 0

        true_pos_inc = 0
        false_pos_inc = 0
        true_neg_inc = 0
        false_neg_inc = 0

        covered = 0

        for inp_idx, sample in self.dataset.splits['unseen'].features.iterrows():
            print(sample.to_list())
            corr_cnt, corr_cover, found = self.eval_rules(inp_idx, correct_rules)
            inc_cnt, inc_cover, found = self.eval_rules(inp_idx, incorrect_rules)

            # print("INPUT:", inp_indx , "CORR CNT:", corr_cnt, "INCORR CNT:", inc_cnt)
            if corr_cnt == inc_cnt:
                print("UNCERTAIN:")
                uncertain += 1
                if self.dataset.splits['unseen'].labels[inp_idx] == labels[inp_idx]:
                    false_neg_cor = false_neg_cor + 1
                    true_neg_inc = true_neg_inc + 1
                else:
                    false_neg_inc = false_neg_inc + 1
                    true_neg_cor = true_neg_cor + 1

            if corr_cnt > inc_cnt:
                print("CORRECT")
                tot_corr += 1

                if self.dataset.splits['unseen'].labels[inp_idx] == labels[inp_idx]:
                    true_pos_cor += 1
                    true_neg_inc += 1
                else:
                    false_pos_cor += 1
                    false_neg_inc += 1

            if inc_cnt > corr_cnt:
                print("INCORRECT")
                tot_inc += 1

                if self.dataset.splits['unseen'].labels[inp_idx] != labels[inp_idx]:
                    true_pos_inc += 1
                    true_neg_cor += 1
                else:
                    false_pos_inc += 1
                    false_neg_cor += 1

            if corr_cover or inc_cover:
                covered += 1

        total_tps = true_pos_cor + true_pos_inc
        total_fps = false_pos_cor + false_pos_inc
        total_tns = true_neg_cor + true_neg_inc
        total_fns = false_neg_cor + false_neg_inc
        retrieved_instances = total_tps + total_fps
        relevant_instances = total_tps + total_fns
        total_precision = (total_tps / retrieved_instances) if retrieved_instances > 0 else 0
        total_recall = (total_tps / relevant_instances) if relevant_instances > 0 else 0

        return {
            "unseen_correct": tot_corr_unseen,
            "unseen_incorrect": tot_inc_unseen,
            "covered": covered,
            "coverage": round((covered / len(self.dataset.splits['unseen'].features))*100.0, 2),
            "uncertain": uncertain,
            "tot_pred_correct": tot_corr,
            "tot_pred_incorrect": tot_inc,
            "true_pos_cor": true_pos_cor,
            "false_pos_cor": false_pos_cor,
            "precision_cor": round(float(true_pos_cor / (true_pos_cor + false_pos_cor)) * 100.0, 2),
            "recall_cor": round(float(true_pos_cor / tot_corr_unseen) * 100.0, 2),
            "true_pos_inc": true_pos_inc,
            "false_pos_inc": false_pos_inc,
            "precision_inc": round(float(true_pos_inc / (true_pos_inc + false_pos_inc)) * 100.0, 2),
            "recall_inc": round(float(true_pos_inc / tot_inc_unseen) * 100.0, 2),
            "total_tps": total_tps,
            "total_fps": total_fps,
            "total_tns": total_tns,
            "total_fns": total_fns,
            "total_precision": round(total_precision * 100.0, 2),
            "total_recall": round(total_recall * 100.0, 2),
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

    def get_model_rep(self):
        """
            Get the model fingerprints
        :return: None
        """

        for layer in self.model.layers:
            func_dense = keras.backend.function(self.model.input, [layer.output])
            inp_tensor = keras.backend.constant(self.dataset.splits['unseen'].features)
            op = func_dense(inp_tensor)
            self._model_rep[layer.name] = (func_dense, inp_tensor, op)
