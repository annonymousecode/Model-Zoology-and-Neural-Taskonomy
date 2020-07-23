# fast.ai flatten_model implementation
import torch.nn as nn
from typing import Collection

ModuleList = Collection[nn.Module]

class Module(nn.Module):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self): super().__init__()
    def __init__(self): pass
    
class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p:nn.Parameter): self.val = p
    def forward(self, x): return x

def children(m:nn.Module)->ModuleList:
    "Get children of `m`."
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of children modules in `m`."
    return len(children(m))

def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children

flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)),[]) if num_children(m) else [m]