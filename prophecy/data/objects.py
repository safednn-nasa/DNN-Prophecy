from dataclasses import dataclass, field


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

