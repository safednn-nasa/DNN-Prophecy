import os
import numpy as np
import prophecy.core.Marabou.build.bin.Marabou
#from maraboupy.MarabouCore import *
#from maraboupy.MarabouPythonic import *

class SolveMarabou:
  def __init__(self, onnx_model_nm: str, onnx_layer_nm: str, x_train_min_layer: np.array, x_train_max_layer: np.array, fngprnt_min_layer: np.array, fngprnt_max_layer: np.array, lab: int):
    self.options = Marabou.createOptions(verbosity = 1, numWorkers=1, numBlasThreads=1,snc=True)
    self.filename = onnx_model_nm
    self.network_a = Marabou.read_onnx(self.filename)
    self.x_train_min_layer = x_train_min_layer
    self.x_train_max_layer = x_train_max_layer
    self.fngprnt_min_layer = fngprnt_min_layer
    self.fngprnt_max_layer = fngprnt_max_layer
    self.lab = lab
    self.onnx_layer_nm = onnx_layer_nm

  def __call__(self, **kwargs):
    print("INPUT VARS")
    invars = self.network_a.inputVars[0][0].flatten()
    print(invars)
    
    for indx in range(0,len(invars)):
      i = invars[indx]
      v = Var(i)
      self.network_a.setLowerBound(i,self.x_train_min_layer[i])
      self.network_a.setUpperBound(i,self.x_train_max_layer[i])
      #network_a.setLowerBound(i,inp_ex[0][indx])
      #network_a.setUpperBound(i,inp_ex[0][indx])

    print("LAYER VARS")
 #   onnx_layer_nm = "dense_14_1/Identity:0"
    neurons = self.network_a.layerNameToVariables[self.onnx_layer_nm][0]
    print(np.shape(neurons))
    
    for indx in range(0, len(neurons)):
      neuron_indx = neurons[indx] - neurons[0]
      self.network_a.setLowerBound(neurons[indx], self.fngprnt_min_layer[neuron_indx])
      self.network_a.setUpperBound(neurons[indx], self.fngprnt_max_layer[neuron_indx])
      #network_a.setLowerBound(dense_14_neurons[indx], finger_ex[0][neuron_indx] - 0.1)
      #network_a.setUpperBound(dense_14_neurons[indx], finger_ex[0][neuron_indx] + 0.1)

    print("OUTPUT VARS")
    outvars = self.network_a.outputVars[0].flatten()
    print(outvars)

    rule_label = self.lab
    prove = True
    for label in range(0,  len(outvars)):
      if (label == rule_label):
        continue
      label_var = Var(outvars[label])
      for indx in range(0,  len(outvars)):
        v = Var(outvars[indx])
        if (indx == label):
          continue
        self.network_a.addConstraint(label_var >= v + 0.001)
        print(v, ":",indx)
        
      sat_unsat,vals,stats = self.network_a.solve(options = self.options)
      print("sat_unsat:", sat_unsat)
    
      if (sat_unsat == 'sat'):
        print("SAT for label:", label)
        print("vals:", vals)
        prove = False
        break
      else:
        print("UNSAT for label:", label)

    if (prove == True):
      print("Rule Proved!!")
