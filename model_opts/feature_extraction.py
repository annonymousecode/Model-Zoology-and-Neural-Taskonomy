import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import torch.nn as nn
import torch

import pandas as pd
import numpy as np

image_transforms = {
    'imagenet_from_numpy': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
    'imagenet': transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
}

def get_image_transforms():
    return(image_transforms)

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)


# Method 1: Flatten model; extract features by layer

from fastai.torch_core import flatten_model

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.out = output.clone().detach().requires_grad_(True).cuda()
    def close(self):
        self.hook.remove()
    def extract(self):
        return self.out
    
def get_layer_names(layers):
    layer_names = []
    for layer in layers:
        layer_name = str(layer).split('(')[0]
        layer_names.append(layer_name + '-' + str(sum(layer_name in string for string in layer_names) + 1))
    return layer_names

def get_features_by_layer(model, target_layer, img_tensor):
    features = SaveFeatures(target_layer)
    model(img_tensor)
    features.close()
    return features.extract()


# Method 2: Hook all layers simultaneously; remove duplicates


def extract_feature_maps(model, inputs):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            layer_key = class_name + '-' + str(sum(class_name in key for key in list(features.keys())) + 1)
            features[layer_key] = output.cpu().detach()

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
    
    features = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
    
    return(features)

def remove_duplicate_feature_maps(feature_maps, return_matches = False):
    matches = []
    layer_names = list(feature_maps.keys())
    for i in range(len(layer_names)):
        for j in range(i+1,len(layer_names)):
            layer1 = feature_maps[layer_names[i]].flatten()
            layer2 = feature_maps[layer_names[j]].flatten()
            if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                if layer_names[j] not in matches:
                    matches.append(layer_names[j])

    for match in matches:
        feature_maps.pop(match)
    
    if return_matches:
        return(feature_maps, matches)
    
    if not return_matches:
        return(feature_maps)
    
def get_all_feature_maps(model, inputs, flatten=True, numpy=True):
    feature_maps = extract_feature_maps(model, inputs)
    feature_maps = remove_duplicate_feature_maps(feature_maps)
    
    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)
            
    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy()
            
    return feature_maps