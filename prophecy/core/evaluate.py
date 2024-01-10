import keras
import numpy as np

from typing import Tuple

from prophecy.data.dataset import Dataset, Split
from prophecy.data.objects import Predictions


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


def predict_unseen(model: keras.Model, dataset_split: Split) -> Predictions:
    unseen_ops = model.predict(dataset_split.features)
    predictions = Predictions()

    cnt_0 = 0
    # TODO: why this count is set to one?
    cnt_1 = 1

    for i in range(0, len(unseen_ops)):
        if unseen_ops[i][0] > 0.5:
            cnt_1 += + 1
            predictions.labels.append(1)

            if dataset_split.labels[i] == 1:
                predictions.correct += 1
            else:
                predictions.incorrect += 1
        else:
            cnt_0 += 1
            predictions.labels.append(0)
            if dataset_split.labels[i] == 0:
                predictions.correct += 1
            else:
                predictions.incorrect += 1

    print("UNSEEN: Label 0:", cnt_0, "Label 1:", cnt_1)
    print("UNSEEN: ACT CORR:", predictions.correct, ", ACT INCORR:", predictions.incorrect)

    return predictions
