import os
import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouCore import *
from maraboupy.MarabouPythonic import *

if __name__ == '__main__':
  options = Marabou.createOptions(verbosity = 1, numWorkers=1, numBlasThreads=1,snc=True)
  filename = "./resources/onnx/cnn_max_mninst2.onnx"
  network_a = Marabou.read_onnx(filename)
  exit()
