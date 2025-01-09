import os
import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouCore import *
from maraboupy.MarabouPythonic import *

class SolveMarabou:
  def _init_(self, onnx_model_nm: str):
    self.options = Marabou.createOptions(verbosity = 1, numWorkers=1, numBlasThreads=1,snc=True)
    self.filename = onnx_model_nm
    self.network_a = Marabou.read_onnx(self.filename)

  def _call_
    print("INPUT VARS")
    self.invars = self.network_a.inputVars[0][0].flatten()
    print(invars)


for indx in range(0,len(invars)):
    i = invars[indx]
    v = Var(i)
    network_a.setLowerBound(i,x_train_min3[i])
    network_a.setUpperBound(i,x_train_max3[i])
    #network_a.setLowerBound(i,inp_ex[0][indx])
    #network_a.setUpperBound(i,inp_ex[0][indx])


print("LAYER VARS MAP")
print(network_a.layerNameToVariables)

dense_14_neurons = network_a.layerNameToVariables["dense_14_1/Identity:0"][0]
print(np.shape(dense_14_neurons))
print(len(dense_14_neurons))
print(dense_14_neurons[0])




for indx in range(0, len(dense_14_neurons)):
    neuron_indx = dense_14_neurons[indx] - dense_14_neurons[0]

    network_a.setLowerBound(dense_14_neurons[indx], fngprnt_min3[neuron_indx])
    network_a.setUpperBound(dense_14_neurons[indx], fngprnt_max3[neuron_indx])
    #network_a.setLowerBound(dense_14_neurons[indx], finger_ex[0][neuron_indx] - 0.1)
    #network_a.setUpperBound(dense_14_neurons[indx], finger_ex[0][neuron_indx] + 0.1)



print("OUTPUT VARS")
outvars = network_a.outputVars[0].flatten()
print(outvars)

rule_label = 0
prove = True
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

  sat_unsat,vals, stats = network_a.solve(options = options)

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
