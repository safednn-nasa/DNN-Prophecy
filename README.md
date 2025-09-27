# DNN-Prophecy
DNN-Prophecy is a tool for automatically inferring formal properties of deep neural networks.
It extracts rules based on neuron activations (values or on/off
statuses) as preconditions that imply a desirable output property specified by the user,
e.g., the prediction being a certain class.

The approach was first published in:
Property Inference for Deep Neural Networks.
Authors: Divya Gopinath, Hayes Converse, Corina S. Pasareanu, Ankur Taly.
Published in ASE'19 proceedings. Preprint available at: https://arxiv.org/abs/1904.13215

Note: Since the DNN-Prophecy software results are precondition rules that are generated based upon user selected, entered postcondition outputs, the accuracy of the resulting rules directly relates to the quality/quantity of user-provided postcondition outputs. Further, the DNN-Prophecy software has no capability to analyze user specific application domains so the software lacks the ability to assess the accuracy of the resulting precondition rules. DNN-Prophecy is a research software tool intended only to aid user understanding of DNN models.

## INSTALLATION
#### Prophecy is implemented in Python 3.10. 
#### It invokes the Marabou solver (https://github.com/NeuralNetworkVerification/Marabou), which currently supports Python 3.8, 3.9, 3.10 and 3.11.

To install the required packages, run:

```shell
#Optional: Create a virtual environment
$ python3.10 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## USAGE

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
- ##### Layer/s to be used for activations collection
- -odl: only dense layers (name starting with text 'dense')
- -oal: includes the activation layers associated with the dense layers.
- -layer_name: name of a specific layer
- ##### Type of input data (provided in -tx and -vx)
- -inptype: 0: model inputs (eg. images), 1: array of neuron activations
- ##### Short-cut for post-cond properties
- -type: 0:rules w.r.t model output,eg.rules for every predicted label, 1:rules for correct vs incorrect classification, 2:rules for correct classification per label and incorrect classification, 3:rules w.r.t labels in -ty
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
                     -tx TRAIN_FEATURES [-vx VAL_FEATURES] [-label LAB] [-min_const MIN_CONST]
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
## LICENSE
* * * * * * * * * * * * * * 

Notices:

Copyright © 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT. 

 

* * * * * * * * * * * * * * 
