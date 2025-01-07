Uses Prophecy to extract rules for correct vs incorrect behavior.
Image models and datasets: MNIST,CIFAR
Tabular model and dataset: PD (PIMA diabetes)

The rules are then used to classify runtime inputs as "CORRECTLY CLASSIFIED", "INCORRECTLY CLASSIFIED", "UNCERTAIN".
The following Prophecy command line arguments are used at runtime.

###Infer Command

Run inference using the specified model and test dataset.

```shell
$ python -m prophecy.main -m /path/to/model.pth -wd /path/to/workdir infer [-h] -tx TEST_FEATURES -ty TEST_LABELS {rules,classifiers} ...
```

####Subcommands
- rules: Detect rule violations on the test data.
- classifiers: Classify test data using pre-trained classifiers.


####Common Arguments:
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
