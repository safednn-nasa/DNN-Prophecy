import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path


from prophecy.utils.paths import datasets_path


@dataclass
class Split:
    name: str
    _features_file: str
    _labels_file: str
    headers: bool = True
    _features: pd.DataFrame = None
    _labels: np.ndarray = None
    _path: Path = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = path / self.name

    @property
    def features(self):
        if self._features is None:
            self._features = pd.read_csv(str(self.path / self._features_file), delimiter=',', encoding='utf-8',
                                         header=None if not self.headers else 'infer')

        return self._features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.loadtxt(str(self.path / self._labels_file), dtype=int)
            #self._labels = pd.read_csv(str(self.path / self._labels_file), delimiter=',', encoding='utf-8', header=None,
            #                           names=['label'])

        return self._labels


@dataclass
class Train(Split):
    name: str = 'train'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = False


@dataclass
class Val(Split):
    name: str = 'val'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = True


@dataclass
class Unseen(Split):
    name: str = 'unseen'
    _features_file: str = 'data.csv'
    _labels_file: str = 'data.csv'
    headers: bool = True


class Dataset:
    def __init__(self, name):
        self.name = name
        self.path = datasets_path / name

        if not self.path.exists():
            raise ValueError(f"Dataset {self.path} does not exist")
        # TODO: normalize datasets to have the same format
        self.splits = {'train': Train(), 'val': Val(), 'unseen': Unseen()}

        for k, v in self.splits.items():
            v.path = self.path
