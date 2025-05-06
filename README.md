# ProphecyPlus
This repository provides the tool for the approach first published in:

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
The examples folder contains a number of notebooks using Prophecy on different models and datasets.

### Analyze Command

Extract rules from the provided training and validation datasets.


```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir analyze [-h] -tx TRAIN_FEATURES -ty TRAIN_LABELS \
						-vx VAL_FEATURES -vy VAL_LABELS [-odl] [-oal] [-sr] [-b] [-c] [-rs RANDOM_STATE] \
						[-layer_name] [-inptype] [-type] [-acts] [-top]
```

#### Arguments

- -m: Pre-trained model in keras (.h5) format
- -wd: Working directory path
- -tx, -ty: Datasets with model input data and labels (.npy) 
- -vx, -vy: Datasets with model input data and labels (.npy) for statistical validation
- 
- ##### Layer/s to be used for activations collection
- -odl: only dense layers (name starting with text 'dense')
- -oal: includes the activation layers associated with the dense layers.
- -layer_name: name of a specific layer
- 
- ##### Type of input data (provided in -tx and -vx)
- -inptype: 0: model inputs (eg. images), 1: array of neuron activations
- 
- ##### Short-cut for post-cond properties
- -type: 0:rules w.r.t model output,eg.rules for every predicted label, 1:rules for correct vs incorrect classification, 2:rules for correct classification per label and incorrect classification, 3:rules w.r.t labels in -ty
- 
- ##### Mathematical form of rules
- -acts: True:on/off neuron activations, False:neuron Values
- -top: --number of rules to be extracted: True: rules with the highest train recall, False: all rules
- -sr: --skip-rules: Skip rules extraction
- -b: --balance: Balance classes in the dataset for training the classifiers
- -c: --confidence: Adjust labels in the dataset for training the classifiers with confidence
- -rs: --random-state: Random state for reproducibility (default: 42)

#### Examples

- Extract rules from a classification model, model.h5, using the train dataset. Each rule corresponds to a distinct label predicted by the model. Rules extracted from the activation and dense layers and in terms of on/off neuron activation values. Only those that have the highest recall on the train data for each label are stored in the output file in wd.

```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir analyze -tx /path/to/train_features.npy -ty /path/to/train_labels.npy \
-vx /path/to/val_features.npy -vy /path/to/val_labels.npy -odl -oal -type 0 -acts True -top True
```

### Prove Command

Attempt to prove rules extracted by Prophecy (invokes Marabou solver https://github.com/NeuralNetworkVerification/Marabou).

```shell
$ python -m prophecy.main -m /path/to/model.h5' -wd /path/to/workdir prove [-h]
                     -mp MARABOU_PATH -onx ONNX_PATH -onx_map ONNX_MAP
                     -tx TRAIN_FEATURES [-ty TRAIN_LABELS] [-vx VAL_FEATURES]
                     [-vy VAL_LABELS] [-label LAB] [-min_const MIN_CONST]
                     [-pred PRED] [-cp CP]
```

#### Arguments
- -mp: Path to Marabou folder
- -onx: Path to model in ONNX form
- -onx_map: Map between the layers of .h5 and .onnx models
- -label: Label for which the rule is chosen. Selects the top rule for given label
- -pred: True:classification output constraints
- -min_const: True:classification output constraints specifying that label has the minimum value
- -cp CONSTS_file: Path to a file specifiying the output constraints (ex. regression outputs)
- -tx: data used to constrain network input variables
- -tx, -vx: in-distribution data to calculate coverage of regions with proofs

#### Example
- For the given label 0, select the rule with the highest recall on the train dataset. Invoke Marabou using the onnx version of the model and attempt to prove the query Vx \sigma(x) => F(x) = label
```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir prove -mp /path/to/marabou\_build\_dir -onx /path/to/onnx_model.onnx -onx_map h5_onnx_map.npy -tx /path/to/train_features.npy -vx /path/to/val_features.npy -label 0 -pred True
```

### Monitor Command

Monitor model's behavior on unseen inputs and classify them as correctly classified, mis-classified or uncertain.

```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir monitor [-h] -tx TEST_FEATURES -ty TEST_LABELS {rules,classifiers} ...
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
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir monitor -tx /path/to/test_features.npy -ty /path/to/test_labels.npy rules
```

- Evaluate a given model on unseen data with the trained classifiers
```shell
$ python -m prophecy.main -m /path/to/model.h5 -wd /path/to/workdir monitor -tx /path/to/test_features.npy -ty /path/to/test_labels.npy classifiers 
```
