from dataclasses import dataclass


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

    def to_dict(self, prefix: str = '') -> dict:
        if prefix:
            prefix = f"{prefix}_"

        return {f"{prefix}coverage": self.coverage,
                f"{prefix}precision": self.precision,
                f"{prefix}recall": self.recall}
