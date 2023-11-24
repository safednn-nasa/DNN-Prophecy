import pandas as pd
from dataclasses import dataclass
from pathlib import Path


from prophecy.utils.paths import datasets_path


@dataclass
class Split:
    name: str
    _features: str
    _labels: str
    _path: Path = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = path / self.name

    @property
    def features(self):
        return pd.read_csv(str(self.path / self._features), delimiter=',', encoding='utf-8')

    @property
    def labels(self):
        return pd.read_csv(str(self.path / self._labels), delimiter=',', encoding='utf-8')


@dataclass
class Train(Split):
    name: str = 'train'
    _features: str = 'x.csv'
    _labels: str = 'y.csv'


@dataclass
class Val(Split):
    name: str = 'val'
    _features: str = 'x.csv'
    _labels: str = 'y.csv'


@dataclass
class Unseen(Split):
    name: str = 'unseen'
    _features: str = 'data.csv'
    _labels: str = 'data.csv'


class Dataset:
    def __init__(self, name):
        self.name = name
        self.path = datasets_path / name

        if not self.path.exists():
            raise ValueError(f"Dataset {self.path} does not exist")

        self.splits = {'train': Train(), 'val': Val(), 'unseen': Unseen()}

        for k, v in self.splits.items():
            v.path = self.path
