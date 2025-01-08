import sys
from abc import abstractmethod

import keras
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from ast import literal_eval
from typing import Tuple, Union
from tqdm import tqdm
from pathlib import Path

from prophecy.core.helpers import check_pattern, get_suffix_cluster


class RulesProve:
    #RulesProve(model=model, onnx_model_nm=onnx_model, layer_nm = top_rule_layer, neurons=rule_neurons_list, sig=rule_sig_list,features=train_features, labels=train_labels)
    def __init__(self, model: keras.Model, onnx_model_nm: str, layer_nm: str, neurons: list, sig: list, features: pd.DataFrame, labels: np.ndarray):
        self.model = model
        self.onnx_path = onnx_model_nm
        self.layer_nm = layer_nm
        self.neurons = neurons
        self.sig = sig
        self.features = features
        self.labels = labels
        
    def get_bounds() -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
        print("MIN AND MAX BOUNDS OF INPUT VARIABLES BASED ON TRAIN DATA")
        x_train = self.features
        x_train_flat = []
        for indx in range(0,len(x_train)):
            x_train_flat.append(x_train[indx].flatten())
        x_train_flat = np.array(x_train_flat)
        length = len(x_train_flat[0])
        
        x_train_min = np.zeros(length)
        x_train_max = np.zeros(length)
        for indx in range(0,length):
            x_train_min[indx] = np.min(x_train_flat[:,indx])
            x_train_max[indx] = np.max(x_train_flat[:,indx])

        print("TRAIN MIN:", x_train_min)
        print("TRAIN MAX:", x_train_max)

        print("GET FINGERPRINTS FOR TRAIN DATA AFTER LAYER:", self.layer_nm)
        func_layer = None
        for layer in self.model.layers:
            if layer.name == self.layer_nm:
                func_layer = keras.backend.function(self.model.input, [layer.output])
        fingerprint_layer = []
        if (func_layer != None):
            fingerprint_layer = func_layer(self.features)

        print("GET INDICES OF INPUTS SATISFYING RULE")
        fingerprints = fingerprint_layer[0]
        if (len(self.neurons) == len(self.sig)):
            fngprnt = (fingerprints > 0.0).astype('int')
            indices = get_suffix_cluster(self.neurons, self.sig, fngprnt)
        else:
            indices = get_suffix_cluster(self.neurons, self.sig, fingerprints, VAL=True)
        print("indices:", len(indices))

        x_train3 = []
        fngprnt3 = []
        inp_ex = []
        finger_ex = []
        for indx in range(0, len(indices)):
            if (indx == 0):
                inp_ex.append(x_train_flat[indices[indx]])
                finger_ex.append(fingerprints[indices[indx]])
            x_train3.append(x_train_flat[indices[indx]])
            fngprnt3.append(fingerprints[indices[indx]])
        x_train3 = np.array(x_train3)
        fngprnt3 = np.array(fngprnt3)

        print("GET MIN,MAX BOUNDS OF INPUTS SATISFYING RULE")
        x_train_min3 = np.zeros(length)
        x_train_max3 = np.zeros(length)
        for indx in range(0,length):
          x_train_min3[indx] = np.min(x_train3[:,indx])
          x_train_max3[indx] = np.max(x_train3[:,indx])

        print(x_train_min3)
        print(x_train_max3)

        print("GET MIN,MAX BOUNDS OF NEURONS SATISFYING RULE")
        fngprnt_min3 = np.zeros(len(fngprnt3[0]))
        fngprnt_max3 = np.zeros(len(fngprnt3[0]))
        for indx in range(0,len(fngprnt3[0])):
            fngprnt_min3[indx] = np.min(fngprnt3[:,indx])
            fngprnt_max3[indx] = np.max(fngprnt3[:,indx])
        print(fngprnt_min3)
        print(fngprnt_max3)
        
        print("INPUT EXAM:", inp_ex[0])
        print("FINGERPRINT EXAM:", finger_ex[0])

        return (x_train_min, x_train_max, x_train_min_layer, x_train_max_layer, fngprnt_min_layer, fngprnt_max_layer, inp_ex[0], fngr_ex[0])

    def solve_marabou():
        
    def __call__(self, **kwargs) -> str:
        results = ""

        (x_train_min, x_train_max, x_train_min_layer, x_train_max_layer, fngprnt_min_layer, fngprnt_max_layer, inp_ex, fngr_ex) = self.get_bounds()

        
        return results
