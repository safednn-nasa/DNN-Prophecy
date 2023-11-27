import keras
import numpy as np

from typing import Tuple

from prophecy.data.dataset import Dataset, Split


def get_eval_labels(model: keras.Model, dataset: Dataset, split: str):
    """
        Evaluate the model on the given features and labels
    :param model: The model to evaluate
    :param dataset: The dataset to evaluate on
    :param split: The split to evaluate on
    :return: labels
    """

    # check split is valid
    if split not in ['train', 'val']:
        raise ValueError(f"Invalid split: {split}")

    predictions = model.predict(dataset.splits[split].features)

    labels = []
    cnt_0 = 0
    cnt_1 = 1

    for i in range(0, len(predictions)):
        if predictions[i][0] > 0.5:
            cnt_1 = cnt_1 + 1
            labels.append(1)
        else:
            cnt_0 = cnt_0 + 1
            labels.append(0)

    print(f"{split.upper()}: Label 0:", cnt_0, "Label 1:", cnt_1)

    return np.array(labels)


def get_unseen_labels(model: keras.Model, dataset_split: Split) -> Tuple[np.array, int, int]:
    unseen_ops = model.predict(dataset_split.features)
    unseen_labels = []

    cnt_0 = 0
    cnt_1 = 1
    tot_corr_unseen = 0
    tot_inc_unseen = 0

    for i in range(0, len(unseen_ops)):
        if unseen_ops[i][0] > 0.5:
            cnt_1 = cnt_1 + 1
            unseen_labels.append(1)

            if dataset_split.labels[i] == 1:
                tot_corr_unseen = tot_corr_unseen + 1
            else:
                tot_inc_unseen = tot_inc_unseen + 1
        else:
            cnt_0 = cnt_0 + 1
            unseen_labels.append(0)
            if dataset_split.labels[i] == 0:
                tot_corr_unseen = tot_corr_unseen + 1
            else:
                tot_inc_unseen = tot_inc_unseen + 1

    print("UNSEEN: Label 0:", cnt_0, "Label 1:", cnt_1)
    print("UNSEEN: ACT CORR:", tot_corr_unseen, ", ACT INCORR:", tot_inc_unseen)

    return np.array(unseen_labels), tot_corr_unseen, tot_inc_unseen
