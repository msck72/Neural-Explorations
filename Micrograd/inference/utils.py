import torch
import torchvision.models as models
import torch.nn as nn
import re

import inference_tensor
import conv_cpp
import pool_layers
from resnet import ResNet18

def build_resnet18_from_torch():
    torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    torch_model.eval()

    model = ResNet18()
    # stem_layer, block1_layer, block2_layer, block3_layer, block4_layer, fc_layer = {}, {}, {}, {}, {}, {}
    # for name, param in torch_model.state_dict().items():
    #     print(name)
    #     if(name == 'conv1.weight'):
    #         stem_layer[name] = param
    #     elif(name.startswith('layer1')):
    #         block1_layer[name] = param
    #     elif(name.startswith('layer2')):
    #         block2_layer[name] = param
    #     elif(name.startswith('layer3')):    
    #         block3_layer[name] = param
    #     elif(name.startswith('layer4')):
    #         block4_layer[name] = param
    #     elif(name.startswith('fc')):
    #         fc_layer[name] = param
    # print("Stem layer:")
    # for k, v in stem_layer.items():
    #     model.set.conv.set_values(v.detach().numpy().tolist())

    # print("\nBloack 1 layer:")
    for k, v in torch_model.state_dict().items():
        # print(f'k = {k}')
        if('num_batches_tracked' in k):
            continue
        ls = k.split('.')
        if(len(ls) == 2):
            if('conv' in k):
                model.stem.conv.set_values(v.detach().numpy().tolist())
            elif('bn' in k):
                getattr(model.stem.bn_layer, f"{ls[-1]}").set_values(v.detach().numpy().tolist())
            continue
        ln = int(ls[1])
        if('conv' in k):
            print(f"model.blocks[{ln}]", f"conv{1 if 'conv1' in k else 2}")
            getattr(model.blocks[ln], f"conv{1 if 'conv1' in k else 2}").set_values(v.detach().numpy().tolist())
        elif('bn' in k):
            print(f"model.blocks[{ln}]", f"bn_layers[{0 if '1' in ls[2] else 1}].{ls[-1]}")
            getattr(model.blocks[ln].bn_layers[0 if '1' in ls[2] else 1], f"{ls[-1]}").set_values(v.detach().numpy().tolist())
    # print("\nBloack 2 layer:")
    # for k, v in block2_layer.items():
    #     print(k)
    # print("\nBloack 3 layer:")
    # for k, v in block3_layer.items():
    #     print(k)
    # print("\nBloack 4 layer:")
    # for k, v in block4_layer.items():
    #     print(k)
    # print("\nFully connected layer:")
    # for k, v in fc_layer.items():
    #     print(k)

    # model = ResNet18()



    # return model


build_resnet18_from_torch()
