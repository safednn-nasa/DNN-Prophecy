# Prophecy
Property Inference from Deep Neural Networks

Work based on the following paper:

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
```
usage: prophecy.py [-h] [--model MODEL] [--version VERSION] [--dataset DATASET] [--settings SETTINGS] command
```

## Examples

- Extract rules from a model

```shell
$ python -m prophecy.main --model PD --version 1 --dataset PD --settings feat_acc extract
```
- Detect violations of a model on unseen data
```shell
$ python -m prophecy.main --model PD --version 1 --dataset PD --settings feat_acc detect
```
