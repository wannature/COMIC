"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


from models.ResNetFeature import *
from utils.util import *
from os import path
from collections import OrderedDict
import torch
from torchvision.models import resnet50,resnet101


def load_model(model, pretrain):
    print("Loading Backbone pretrain model from {}......".format(pretrain))
    model_dict = model.state_dict()
    pretrain_dict = torch.load(pretrain)
    new_dict = OrderedDict()

    for k, v in pretrain_dict.items():
        if k.startswith("module"):
            k = k[7:]
        if "fc" not in k and "classifier" not in k:
            new_dict[k] = v

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print("Backbone model has been loaded......")

    return model
        
def create_model(feat_dim=512, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')

    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(num_ftrs,feat_dim),
                            nn.LogSoftmax(dim=1))
    model = model.cuda()


    # resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=None)
    # resnet50 = load_model(resnet50, pretrain='/hdd8//pretrained_model/resnet50-0676ba61.pth')
    # num_ftrs = resnet50.fc_add.in_features
    # resnet50.fc_add = nn.Sequential(nn.Linear(num_ftrs,80),
    #                         nn.LogSoftmax(dim=1))
    return model
