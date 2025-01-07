import os
import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouCore import *
from maraboupy.MarabouPythonic import *

options = Marabou.createOptions(verbosity = 1, numWorkers=1, numBlasThreads=1,snc=True)

filename = "./resources/onnx/acasxu/ACASXU_experimental_v2a_1_1.onnx"
network_a = Marabou.read_onnx(filename)

#-0.295234 -0.5      -0.5      -0.5      -0.5
print("INPUT VARS")
invars = network_a.inputVars[0][0].flatten()
print(invars)
for i in invars:
    v = Var(i)
    if (i == 0):
      network_a.setLowerBound(i,x_train_min3[0])
      network_a.setUpperBound(i,x_train_max3[0])
      #network_a.setLowerBound(i,0.100096)
      #network_a.setUpperBound(i,0.100096)
    if (i == 1):
      network_a.setLowerBound(i,x_train_min3[1])
      network_a.setUpperBound(i,x_train_max3[1])
      #network_a.setLowerBound(i,-0.5)
      #network_a.setUpperBound(i,-0.5)
    if (i == 2):
      network_a.setLowerBound(i,x_train_min3[2])
      network_a.setUpperBound(i,x_train_max3[2])
      #network_a.setLowerBound(i,-0.5)
      #network_a.setUpperBound(i,-0.5)
    if (i == 3):
      network_a.setLowerBound(i,x_train_min3[3])
      network_a.setUpperBound(i,x_train_max3[3])
      #network_a.setLowerBound(i,-0.5)
      #network_a.setUpperBound(i,-0.5)
    if (i == 4):
      network_a.setLowerBound(i,x_train_min3[4])
      network_a.setUpperBound(i,x_train_max3[4])
      #network_a.setLowerBound(i,-0.5)
      #network_a.setUpperBound(i,-0.5)

print("LAYER VARS MAP")
print(network_a.layerNameToVariables)



dense_3_neurons = network_a.layerNameToVariables["relu_3"][0]
print(np.shape(dense_3_neurons))
print(len(dense_3_neurons))
print(dense_3_neurons[0])


inp = [0.10179728,0,1.3244011,0.04412816,0.57699263,0,0,0,0,0,0,0,0,0,0,0.39995432,0,2.1758285,0,0,0,0,3.7059731,1.4501623,0,0,0.11942634,0,0,1.348339,0,0.16858709,0,0,0,0,0.20186378,0.2259273,0,0,0.05198997,1.1146446,0,0,0,0,0,0,0,0.25532976]


for indx in range(0, len(dense_3_neurons)):
    neuron_indx = dense_3_neurons[indx] - dense_3_neurons[0]
   # print(neuron_indx)
    network_a.setLowerBound(dense_3_neurons[indx], fngprnt_min3[neuron_indx])
    network_a.setUpperBound(dense_3_neurons[indx], fngprnt_max3[neuron_indx])
    #network_a.setLowerBound(dense_3_neurons[indx], inp[neuron_indx] - 0.1)
    #network_a.setUpperBound(dense_3_neurons[indx], inp[neuron_indx] + 0.1)


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
