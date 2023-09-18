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
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(num_ftrs,80),
                            nn.LogSoftmax(dim=1))
    model = model.cuda()


    print('done\n')


    train_multi_label_coco(model, train_dataloader,val_dataloader)


def train_multi_label_coco(model, train_loader, val_loader):

    # set optimizer
    Epochs = 40
    Stop_epoch = 40
    weight_decay = 1e-5
    learning_rate= 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    # criterion=./loss/PriorFocalModifierLoss.py(distribution_path=training_opt["distribution_path"], \
    #     co_occurrence_matrix=training_opt["co_occurrence_matrix"])
    # criterion=FocalLoss ()
    # criterion=CrossEntropyLoss(use_sigmoid=True)
    spls_loss=SPLC(batch_size=32)
    # criterion=Hill()
    parameters = add_weight_decay(model, weight_decay)

    optimizer = torch.optim.SGD(params=parameters, lr=learning_rate,weight_decay=weight_decay)
    # (params=parameters, lr=learning_rate, weight_decay=weight_decay)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)

    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=steps_per_epoch, epochs=Epochs,
    #                                     pct_start=0.2)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for step, (inputs, labels, indexes) in enumerate(train_loader):
            inputs = inputs.cuda()
            target =labels.cuda()  

            with torch.set_grad_enabled(True):  # mixed precision
                outputs  = model(inputs).float()  # sigmoid will be done in loss !


            #mixup
            # inputs, targets_a, targets_b, lam = mixup_data(inputData, target, 1.)
            # inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            # loss_func = mixup_criterion(targets_a, targets_b, lam)
            # with torch.set_grad_enabled(True):  # mixed precision
            #     outputs  = model(inputs).float()  # sigmoid will be done in loss !
            # loss = loss_func(criterion, outputs)
            loss = criterion(outputs, target)
            loss_sp=0.2*spls_loss(outputs,target,epoch)
            # loss_sp=0
            loss+=loss_sp


            
            # loss_sp=spls_loss(outputs,target,epoch)
            
            # loss_sp=0
            
            # print (loss_sp)
            model.zero_grad() 
            
            loss.backward()
            # loss.backward()
            optimizer.step()
            # optimizer.step()
            # scheduler.step()
            # store information
            if step % 10 == 0:
                trainInfoList.append([epoch, step, loss.item(), loss_sp])
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.1f}, Label Correction Loss: {:.1f}'
                      .format(epoch, Epochs, str(step).zfill(3), str(steps_per_epoch).zfill(3), 
                            #   scheduler.get_last_lr()[0], 
                              loss.item(), loss_sp))

        # try:
        #     torch.save(model.state_dict(), os.path.join(
        #         '/hdd8//dataset/coco/model/baseline', 'model-head-coco-{}-{}.ckpt'.format(epoch + 1, step + 1)))
        # except:
        #     pass
        if epoch>5:
            model.eval()
            mAP_ema_total, mAP_many_shot, mAP_median_shot,mAP_low_shot= validate_multi(train_loader.dataset.per_class_labels,val_loader, model)
            mAP_total_average=(mAP_many_shot+ mAP_median_shot+ mAP_low_shot)/3
            model.train()
            if mAP_total_average > highest_mAP:
                highest_mAP = mAP_total_average
                try:
                    torch.save(model.state_dict(), os.path.join(
                        'models/', 'model-balanced-voc-highest.ckpt'))
                except:
                    pass
            print('current_average_mAP = {:.2f}, highest_average_mAP = {:.2f}\n'.format(mAP_total_average, highest_mAP))


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
