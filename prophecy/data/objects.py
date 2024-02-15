import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Settings:
    fingerprint: str
    rules: str


@dataclass
class Rule:
    kind: str
    layer: int
    neurons: list
    signature: list
    support: int
    label: int

    def __str__(self):
        return (f"KIND: {self.kind}, LAYERS: {self.layer}, NEURONS: {self.neurons}, SIGNATURE: {self.signature}, "
                f"SUPPORT: {self.support}, LABEL: {self.label}")


@dataclass
class Performance:
    coverage: float
    precision: float
    recall: float
    f1: float

    def to_dict(self, prefix: str = '') -> dict:
        if prefix:
            prefix = f"{prefix}_"

        return {f"{prefix}coverage": self.coverage,
                f"{prefix}precision": self.precision,
                f"{prefix}recall": self.recall,
                f"{prefix}f1": self.f1}


@dataclass
class Predictions:
    labels: list = field(default_factory=lambda: [])
    correct: int = 0
    incorrect: int = 0


@dataclass
class Evaluation:
    tot_corr: int = 0
    tot_inc: int = 0
    uncertain: int = 0
    true_pos: int = 0
    false_pos: int = 0
    true_neg: int = 0
    false_neg: int = 0
    outputs: list = field(default_factory=lambda: [])

    def __call__(self, true_label: Any, pred_label: Any, is_pos: bool) -> str:
        if is_pos:
            if true_label != pred_label:
                self.true_pos += 1
                return 'tp'
            else:
                self.false_pos += 1
                return 'fp'
        else:
            # TODO: explain this condition
            if true_label == pred_label:
                self.true_neg += 1
                return 'tn'
            else:
                self.false_neg += 1
                return 'fn'

    @property
    def retrieved(self):
        return self.true_pos + self.false_pos

    @property
    def relevant(self):
        return self.true_pos + self.false_neg

    @property
    def precision(self):
        return (self.true_pos / self.retrieved) if self.retrieved > 0 else 0

    @property
    def recall(self):
        return (self.true_pos / self.relevant) if self.relevant > 0 else 0

    @property
    def f1(self):
        recall_precision = self.precision + self.recall
        return (2 * self.precision * self.recall) / recall_precision if recall_precision > 0 else 0

    @property
    def mcc(self):
        covar = self.true_pos * self.true_neg - self.false_pos * self.false_neg
        denom = np.sqrt((self.true_pos + self.false_pos) * (self.true_pos + self.false_neg) *
                        (self.true_neg + self.false_pos) * (self.true_neg + self.false_neg))

        if denom == 0:
            print(f"Warning: MCC denominator is zero. True Pos: {self.true_pos}, False Pos: {self.false_pos}, "
                  f"True Neg: {self.true_neg}, False Neg: {self.false_neg}")

        return covar / denom if denom > 0 else 0

    def performance(self):
        return {
            "precision": round(self.precision * 100.0, 2),
            "recall": round(self.recall * 100.0, 2),
            "f1": round(self.f1 * 100.0, 2),
            "mcc": round(self.mcc, 3)
        }

    def to_dict(self):
        return {
            "uncertain": self.uncertain,
            "tot_pred_correct": self.tot_corr,
            "tot_pred_incorrect": self.tot_inc,
            "tps": self.true_pos,
            "fps": self.false_pos,
            "tns": self.true_neg,
            "fns": self.false_neg,
        }
