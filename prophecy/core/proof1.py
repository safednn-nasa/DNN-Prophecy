import sys
import os
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

import time
#import func_timeout

from prophecy.core.helpers import check_pattern, get_suffix_cluster

sys.path.append('/content/drive/MyDrive/Marabou')
from maraboupy import Marabou
from maraboupy.MarabouCore import *
from maraboupy.MarabouPythonic import *


class RulesProve:
    def __init__(self, model: keras.Model, onnx_model_nm: str, onnx_map_nm: str, layer_nm: str, neurons: list, sig: list, features: pd.DataFrame, labels: np.ndarray, lab: int, iter: int):
        self.model = model
        self.onnx_path = onnx_model_nm
        self.onnx_map = onnx_map_nm
        self.layer_nm = layer_nm
        self.neurons = neurons
        self.sig = sig
        self.features = features
        self.labels = labels
        self.lab = lab
        self.iter = iter
        
    def get_bounds(self) -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
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

        return (x_train_min, x_train_max, x_train_min3, x_train_max3, fngprnt_min3, fngprnt_max3, inp_ex[0], finger_ex[0])
        
    def __call__(self, **kwargs) -> bool:
        
        (x_train_min, x_train_max, x_train_min_layer, x_train_max_layer, fngprnt_min_layer, fngprnt_max_layer, inp_ex, finger_ex) = self.get_bounds()

        results = False
        
        onnx_model_nm=self.onnx_path
        h5_onnx_map = np.genfromtxt(self.onnx_map, delimiter=',', dtype=str)
        print("h5_onnx_map:", np.shape(h5_onnx_map))
        onnx_layer_nm = None
        for indx1 in range(0,len(h5_onnx_map)):
            if (h5_onnx_map[indx1][0] == self.layer_nm):
                onnx_layer_nm = h5_onnx_map[indx1][1]
                break
        if (onnx_layer_nm != None):
            print("onnx layer name:",onnx_layer_nm)
        else:
            print("could not find the onnx layer mapped to the h5 layer", self.layer_nm)
            
        #onnx_layer_nm="dense_14_1/Identity:0"
        lab=self.lab
     
        #options1 = Marabou.createOptions(verbosity = 1,numWorkers=1,timeoutInSeconds=90,snc=True)
        options1 = Marabou.createOptions(verbosity = 1,timeoutInSeconds=120)
        filename = onnx_model_nm
        network_a = Marabou.read_onnx(filename)

        print("INPUT VARS")
        invars = network_a.inputVars[0][0].flatten()
        print(invars)
    
        for indx in range(0,len(invars)):
            i = invars[indx]
            v = Var(i)
            if (self.iter ==  2):
                network_a.setLowerBound(i,inp_ex[indx])
                network_a.setUpperBound(i,inp_ex[indx])
            if ((self.iter == 0) or (self.iter == 1)):
               network_a.setLowerBound(i,x_train_min_layer[i])
               network_a.setUpperBound(i,x_train_max_layer[i])
            

        print("LAYER VARS")
        neurons_layer = network_a.layerNameToVariables[onnx_layer_nm][0]
        print(np.shape(neurons_layer))
    
        for indx in range(0, len(neurons_layer)):
            neuron_indx = neurons_layer[indx] - neurons_layer[0]
            if (self.iter > 0):
                network_a.setLowerBound(neurons_layer[indx], finger_ex[neuron_indx] - 0.1)
                network_a.setUpperBound(neurons_layer[indx], finger_ex[neuron_indx] + 0.1)
            if (self.iter == 0):
                network_a.setLowerBound(neurons_layer[indx], fngprnt_min_layer[neuron_indx])
                network_a.setUpperBound(neurons_layer[indx], fngprnt_max_layer[neuron_indx])
            

        print("OUTPUT VARS")
        outvars = network_a.outputVars[0].flatten()
        print(outvars)

        rule_label = lab
        #prove = True
        sat_lbls = []
        unsat_lbls = []
        for label in range(0,  len(outvars)):
            if (label == rule_label):
                continue
            label_var = Var(outvars[label])
            for indx in range(0,  len(outvars)):
                v = Var(outvars[indx])
                if (indx == label):
                    continue
                network_a.addConstraint(label_var >= v + 0.001)
                print(v, ":",indx)


            sat_unsat = None
            vals = None
            stats = None
            
            sat_unsat,vals,stats = network_a.solve(options = options1)
           
            print("sat_unsat:", sat_unsat)
            
            if (sat_unsat == 'sat'):
                print("SAT for label:", label)
                print("vals:", vals)
               # prove = False
                sat_lbls.append(label)
            if (sat_unsat == 'unsat'):
                print("UNSAT for label:", label)
                unsat_lbls.append(label)
        

        if (len(sat_lbls) == 0):
            print("RULE PROVED!!")
            results = True
            return results
            
        if (len(unsat_lbls) > 0):
            print("Rule Proved for the following labels.")
            for indx1 in range(0, len(unsat_lbls)):
                print("LABEL:", unsat_lbls[indx1])
        if (len(sat_lbls) > 0):
            print("Rule NOT proved for the following labels.")
            for indx1 in range(0, len(sat_lbls)):
                print("LABEL:", sat_lbls[indx1])
        
        return results
