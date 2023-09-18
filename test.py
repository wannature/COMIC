import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from utils.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay,shot_mAP, mixup_data, mixup_criterion

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from utils.util import source_import, update, shot_acc
from utils import dataloader
from loss.ReflectiveLabelCorrectorLoss import SPLC
from torchvision.models import resnet50,resnet101
import yaml
from torch import nn
import numpy as np
from torch.autograd import Variable
#Losses
from loss../loss/PriorFocalModifierLoss.py import ./loss/PriorFocalModifierLoss.py
from loss.AsymmetricLoss import AsymmetricLoss
from loss.FocalLoss import FocalLoss
from loss.Cross_entropy_loss import CrossEntropyLoss
from loss.HillLoss import Hill

parser = argparse.ArgumentParser(description='PyTorch MLT_COCO Training')
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--xERM', default=False, action='store_true') 
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
args = parser.parse_args()
with open(args.cfg) as f:
    config = yaml.load(f)
training_opt = config['training_opt']
os.environ['CUDA_VISIBLE_DEVICES'] = str(training_opt["gpu_ids"])

def main():
    model_path=""
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(num_ftrs,80),
                            nn.LogSoftmax(dim=1))
    weights = torch.load(model_path)   
    weights = weights['state_dict']
    model.load_state_dict(weights)   
    model = model.cuda()
    args.do_bottleneck_head = False
    sampler_dic=None
    train_dataloader,val_dataloader = dataloader.load_data(training_opt,sampler_dic)
    # Setup model
    print('creating model...')
   
    


    print('done\n')


    validate_multi (per_class_labels,val_loader, model)



def validate_multi(per_class_labels,val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for step, (inputs, labels, indexes) in enumerate(val_loader):
        target = labels
        # compute output
        with torch.no_grad():
            output_regular = Sig(model(inputs.cuda())).cpu()
        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())
    print (per_class_labels)
    mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot\
        = shot_mAP(per_class_labels,torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    # mAP_ema_total, mAP_ema_many_shot, mAP_ema_median_shot,mAP_ema_many_shot\
    #      = shot_mAP(per_class_labels,torch.cat(targets).numpy(), torch.cat(preds_ema).numpy(),many_shot_thr=100, low_shot_thr=20)
    # print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))

   
    print("mAP score total  shot {:.2f}, many  shot {:.2f}, median shot {:.2f}, low shot {:.2f}".format(mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot))
    return  mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot


if __name__ == '__main__':
    main()
