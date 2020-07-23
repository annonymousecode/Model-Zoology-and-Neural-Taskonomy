import torch.backends.cudnn as cudnn
import torch.utils.data
import torch

import matplotlib.pyplot as plt
import numpy as np
import sys, os

def set_device_and_seed(device, seed=None):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        cudnn.benchmark = True
        if seed != None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    print('Using Device[{}]: {}'.format(device, torch.cuda.get_device_name(device)))
    
    
