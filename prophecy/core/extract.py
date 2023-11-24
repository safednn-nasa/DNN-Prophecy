import keras
import numpy as np
import pandas as pd


from prophecy.data.dataset import Dataset
from prophecy.core.evaluate import get_eval_labels


def extract_rules(model: keras.Model, dataset: Dataset, split: str):
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
    #TODO: for test in the previous code the eval_labels are computed with (model.predict(x_val)).argmax(axis=1)
    # that yields different results than the get_eval_labels function
    acc_labels = []
    dec_labels = []
    print(f"{split.upper()} LABELS:", eval_labels.shape)
    match_count = 0
    mismatch_count = 0

    for idx in range(0, len(eval_labels)):
        # if eval_labels[idx] == int(dataset.splits[split].labels.iloc[idx]['label']):
        if eval_labels[idx] == dataset.splits[split].labels[idx]:
            match_count = match_count + 1
            dec_labels.append(eval_labels[idx])
            acc_labels.append(0)
        else:
            mismatch_count = mismatch_count + 1
            dec_labels.append(1000)
            acc_labels.append(1000) #Misclassified

    print(f"{split.upper()} ACCURACY:", (match_count/(match_count + mismatch_count)) * 100.0)
