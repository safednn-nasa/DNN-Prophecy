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

### Analyze Command

Extract rules and train classifiers from the provided training and validation datasets.


```shell
python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir analyze [-h] -tx TRAIN_FEATURES -ty TRAIN_LABELS \
						-vx VAL_FEATURES -vy VAL_LABELS [-odl] [-oal] [-sr] [-b] [-c] [-rs RANDOM_STATE]
```

#### Arguments

- -tx, --train_features (required): Path to the training features.
- -ty, --train_labels (required): Path to the training labels.
- -vx, --val_features (required): Path to the validation features.
- -vy, --val_labels (required): Path to the validation labels.
- -odl, --only-dense-layers: Consider only dense layers.
- -oal, --only-activation-layers: Include the activation layers associated with the dense layers.
- -sr, --skip-rules: Skip rules extraction.
- -b, --balance: Balance classes in the dataset for training the classifiers.
- -c, --confidence: Adjust labels in the dataset for training the classifiers with confidence.
- -rs, --random-state: Random state for reproducibility (default: 42).


#### Examples

- Train classifiers from a model and extract rules only for activation/dense layers

```shell
$ python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir analyze -tx /path/to/train_features.csv -ty /path/to/train_labels.csv \
-vx /path/to/val_features.csv -vy /path/to/val_labels.csv -odl -oal 
```

### Infer Command

Run inference using the specified model and test dataset.

```shell
$ python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir infer [-h] -tx TEST_FEATURES -ty TEST_LABELS {rules,classifiers} ...
```

#### Subcommands
- rules: Detect rule violations on the test data.
- classifiers: Classify test data using pre-trained classifiers.


#### Common Arguments:
- -tx, --test_features (required): Path to the test features.
- -ty, --test_labels (required): Path to the test labels.


#### rules Arguments:
- -t, --threshold: Sets the F1-threshold for selecting rules (default: 0.0).

#### classifiers Arguments:
- -op, --only-pure: Consider only classifications with 100% probability.

#### Examples
- Evaluate a given model on unseen data with the extracted rules
```shell
$ python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir infer -tx /path/to/test_features.csv -ty /path/to/test_labels.csv rules
```

- Evaluate a given model on unseen data with the trained classifiers
```shell
$ python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir infer -tx /path/to/test_features.csv -ty /path/to/test_labels.csv classifiers 
```
