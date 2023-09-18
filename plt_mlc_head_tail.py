import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from utils.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay,shot_mAP
from loss.AsymmetricLoss import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from utils.util import source_import, update, shot_acc
from utils import dataloader
from torchvision.models import resnet50
import yaml
from torch import nn
import numpy as np
from models.MltMultiLabelNetwork import MLTModel
parser = argparse.ArgumentParser(description='PyTorch MLT_COCO Training')
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--xERM', default=False, action='store_true') 
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
args = parser.parse_args()
with open(args.cfg) as f:
    config = yaml.load(f)
training_opt = config['training_opt']
os.environ['CUDA_VISIBLE_DEVICES'] = str(training_opt["gpu_ids"])
sampler_dic=None
sampler_defs = training_opt['sampler']
if sampler_defs:
    if sampler_defs['type'] == 'ClassAwareSampler':
        sampler_dic = {
            'sampler': source_import(sampler_defs['def_file']).get_sampler(),
            'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
        }
    elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                    'ClassPrioritySampler']:
        sampler_dic = {
            'sampler': source_import(sampler_defs['def_file']).get_sampler(),
            'params': {k: v for k, v in sampler_defs.items() \
                        if k not in ['type', 'def_file']}
        }
def main():
    
    args.do_bottleneck_head = False
    train_dataloader,val_dataloader = dataloader.load_data(training_opt,sampler_dic)
    # Setup model
    print('creating model...')
    training_model = MLTModel(config, train_dataloader, val_dataloader, test=False)
    training_model.train()

    print('done\n')



if __name__ == '__main__':
    main()
