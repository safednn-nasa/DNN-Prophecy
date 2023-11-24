import pandas as pd

from prophecy.utils.paths import models_path, datasets_path
from tensorflow import keras
from dataclasses import dataclass


@dataclass
class DatasetSplit:
    name: str
    features: str
    labels: str


@dataclass
class TrainSplit(DatasetSplit):
    name: str = 'train'
    features: str = 'x.csv'
    labels: str = 'y.csv'


@dataclass
class ValSplit(DatasetSplit):
    name: str = 'val'
    features: str = 'x.csv'
    labels: str = 'y.csv'


@dataclass
class UnseenSplit(DatasetSplit):
    name: str = 'unseen'
    features: str = 'data.csv'
    labels: str = 'data.csv'


def get_model(model: str, version: str) -> keras.Model:
    model_path = models_path / f"{model}{version}.h5"

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


def get_dataset(dataset: str, split: DatasetSplit, features: bool = True) -> pd.DataFrame:
    if features:
        dataset_path = datasets_path / dataset / split.name / split.features
    else:
        dataset_path = datasets_path / dataset / split.name / split.labels

    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist")

    return pd.read_csv(str(dataset_path), delimiter=',', encoding='utf-8')
