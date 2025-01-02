# ProphecyPlus
This repository provides the tool for the approach published in:

Property Inference for Deep Neural Networks.
Authors: Divya Gopinath, Hayes Converse, Corina S. Pasareanu, Ankur Taly.
Published in ASE'19 proceedings. Preprint available at: https://arxiv.org/abs/1904.13215


## Installation
Prophecy is implemented in Python 3.10. To install the required packages, run:

```shell
#Optional: Create a virtual environment
$ python3.10 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Usage

#### Example Notebooks
The examples folder contains a number of notebooks using Prophecy.

### Analyze Command

Extract rules from the provided training and validation datasets.


```shell
python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir analyze [-h] -tx TRAIN_FEATURES -ty TRAIN_LABELS \
						-vx VAL_FEATURES -vy VAL_LABELS [-odl] [-oal] [-sr] [-b] [-c] [-rs RANDOM_STATE] \
						[-layer_name] [-inptype] [-type] [-acts] [-top]
```

#### Arguments

- -tx, --train_features (required): Path to the training features.
- -ty, --train_labels (required): Path to the training labels.
- -vx, --val_features (required): Path to the validation features.- -vy, --val_labels (required): Path to the validation labels.
- -odl, --only-dense-layers: Consider only dense layers.
- -oal, --only-activation-layers: Include the activation layers associated with the dense layers.
- -sr, --skip-rules: Skip rules extraction.
- -b, --balance: Balance classes in the dataset for training the classifiers.
- -c, --confidence: Adjust labels in the dataset for training the classifiers with confidence.
- -rs, --random-state: Random state for reproducibility (default: 42).
- -layer_name --layer name: Name of a specific dense or activation layer for which rules need to be extracted.
- -inptype, --type of input: 0: model, 1: an array of neuron values/activations.
- -type, --type of rules to be extracted: 0: rules based on model output (such as predicted labels), \
  1: rules for correct vs incorrect model behavior, 2: rules for correct classification to each label and incorrect classification, \
  3: rules for labels provided as an input array.
- -acts, --mathematical form of rules: True: on/off neuron activations, False: neuron Values.
- -top, --number of rules to be extracted: True: rules with the highest train recall, False: all rules.

#### Examples

- Extract rules for the given keras model based on the train and validation data provided as .npy files based on the model's predictions (ex. distinct labels) for activation and dense layers, in terms of on/off neuron activation values, only those that have the highest recall on the train data. 

```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir analyze -tx /path/to/train_features.npy -ty /path/to/train_labels.npy \
-vx /path/to/val_features.npy -vy /path/to/val_labels.npy -odl -oal -type 0 -acts True -top True
```


