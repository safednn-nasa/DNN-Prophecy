import keras
import numpy as np

from prophecy.data.dataset import Dataset


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
