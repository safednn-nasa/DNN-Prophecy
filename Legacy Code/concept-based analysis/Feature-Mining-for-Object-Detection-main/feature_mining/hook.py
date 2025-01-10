import torch
import numpy as np
import os

from torch import Tensor
from torch.nn import Module


from typing import Tuple, List, Dict, Union


class ForwardHook(object):
    def __init__(self, module:Module):
        super(ForwardHook).__init__()

        self._module_input:List[Tensor] = []
        self._module_output:Tensor = None
        
        self.module = module
        self._module_handle = None
    
    @property
    def module_input(self)->List[Tensor]:
        return self._module_input
    
    @property
    def module_output(self)->Tensor:
        assert self._module_output is not None
        return self._module_output

    def hook(self, fn=None):
        def _hook_fn(module:Module, input:Tuple[Tensor], output:Tensor):
            if fn is not None:
                output = fn(module, input, output)
            # based on the document of PyTorch, the input is a tuple
            # of Tensor (may be for multiple inputs function)
            # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
            self._module_input = [i.clone() for i in input]
            # different from input, output is a tensor
            self._module_output = output.clone()
            return output

        if self._module_handle is not None:
            self.remove()
        self._module_handle = self.module.register_forward_hook(_hook_fn)

        return self
            
    def remove(self):
        if self._module_handle is not None:
            self._module_handle.remove()
            self._module_handle = None
        
        self._module_input = []
        self._module_output = None