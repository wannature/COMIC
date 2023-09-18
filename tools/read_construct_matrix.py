from matplotlib.ticker import MultipleLocator
from models.BalancedlNetwork import MLTModel
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from utils.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay,shot_mAP, mixup_data, mixup_criterion
from logger import Logger
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from utils.util import source_import, update, shot_acc
from utils import dataloader
from loss.ReflectiveLabelCorrectorLoss import SPLC
from torchvision.models import resnet50,resnet101
import yaml
from torch import nn
import copy
import numpy as np
from torch.autograd import Variable
#Losses
from loss../loss/PriorFocalModifierLoss.py import ./loss/PriorFocalModifierLoss.py
from loss.AsymmetricLoss import AsymmetricLoss
from loss.FocalLoss import FocalLoss
from loss.Cross_entropy_loss import CrossEntropyLoss
from loss.HillLoss import Hill




def load_model( model_path):

    ### get the parameters
    parser = argparse.ArgumentParser(description='PyTorch MLT_COCO Training')
    parser.add_argument('--cfg', default=None, type=str)
    args = parser.parse_args()
    with open(args.cfg) as f:
        config = yaml.load(f)
    training_opt = config['training_opt']
    num_classes=int(training_opt["num_classes"])
    model_dir = training_opt['log_dir']
    if not model_dir.endswith('.pth'):
        model_dir = os.path.join(model_dir,model_path)
    print('Loading model from %s' % (model_dir))
    checkpoint = torch.load(model_dir)          
    model_state = checkpoint['state_dict_best']
    model_weights_feat_model = copy.deepcopy(model_state['feat_model'])
    model_weights_additive_attention = copy.deepcopy(model_state['additive_attention'])
    model_weights_classifier= copy.deepcopy(model_state['classifier'])

    ### load the model and parameters
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
                        if k not in ['type', 'def_file']}}
    train_dataloader,val_dataloader = dataloader.load_data(training_opt,sampler_dic)
    training_model = MLTModel(config, train_dataloader, val_dataloader, test=False)
    training_model.balanced_networks['feat_model'].load_state_dict(model_weights_feat_model)
    training_model.balanced_networks['additive_attention'].load_state_dict(model_weights_additive_attention)
    training_model.balanced_networks['classifier'].load_state_dict(model_weights_classifier)
    print("Have loaded the parameters for the model")

    return training_model, num_classes



def eval(self,  epoch, phase='val', save_feat=False):
        print_str = ['Phase: %s' % (phase)]
        torch.cuda.empty_cache()
        Sig = torch.nn.Sigmoid()
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.balanced_networks.values():
            model.eval()
        head_preds= []
        balanced_preds= []
        tail_preds= []
        targets = []
        # Iterate over dataset
        for inputs, labels, indexes in tqdm(self.val_dataloader):
            paths_image=self.val_dataloader.dataset.get_image_path(np.array(indexes).reshape(1,-1)[0])
            inputs, labels = inputs.cuda(), labels.cuda()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                self.head_features = self.head_networks['feat_model'](inputs)
                self.tail_features = self.tail_networks['feat_model'](inputs)
                self.balanced_features = self.balanced_networks['feat_model'](inputs)
                self.keys=torch.cat([self.head_features.unsqueeze(1), self.tail_features.unsqueeze(1)],1)
                self.balanced_features=(self.balanced_features+ 0.1*self.balanced_networks['additive_attention'](self.balanced_features.unsqueeze(1),self.keys,self.keys).squeeze(1))
                self.logits_balanced,_= self.balanced_networks['classifier'](self.balanced_features, labels, self.balanced_embed_mean)
                output = Sig(self.logits_balanced).cpu()
                balanced_preds.append(output.cpu().detach())
                flag=False
                targets.append(labels.cpu().detach())
                for i in range(len(indexes)):
                    temp_pred=output[i,:].cpu()>0.5
                    temp_target=labels[i,:].cpu()>0.5
                    temp_true_index_target=np.where(temp_target==True)[0]
                    temp_true_index_pred=np.where(temp_pred==True)[0]
                    if(len(np.intersect1d(temp_true_index_target, temp_true_index_pred))==len(temp_true_index_target) and flag==False):
                        string=""
                        # for j in range(len(temp_true_index_target)):
                        #     string+=", "+ str(temp_true_index_target[j])
                        for j in range(len(output[i,:][temp_true_index_target])):
                            string+=", "+ str(output[i,:][temp_true_index_target][j].item())
                        if("000000105014" in str(paths_image[i]) or "000000437331" in str(paths_image[i]) or "000000265518" in str(paths_image[i]) or "000000262895" in str(paths_image[i]) or "000000015497" in str(paths_image[i]) or "000000545100" in str(paths_image[i]) or "000000015597" in str(paths_image[i]) or "000000303908" in str(paths_image[i]) or "000000458255" in str(paths_image[i]) or "000000251572" in str(paths_image[i])):
                            print(paths_image[i] + string)
        print ("balanced classifier performance.")
        ap= shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(balanced_preds).numpy(), self.training_opt["head_class_number"], self.training_opt["tail_class_number"])

        return ap



def shot_mAP(per_class_number, targs, preds, many_shot_thr=100, low_shot_thr=20):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    many_shot = []
    median_shot = []
    low_shot = []
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # print (scores)
            
        # compute average precision
        ap[k] = average_precision(scores, targets)
        if per_class_number[k]>=many_shot_thr:
            many_shot.append(ap[k])
        elif per_class_number[k]<low_shot_thr:
            low_shot.append(ap[k])
        else:
            median_shot.append(ap[k])
    return ap

def average_precision(output, target):
    epsilon = 1e-8

    
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))
    # print(111111111)
    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    # print(np.logical_not(ind))
    # print(pos_count_[np.logical_not(ind)])
    # print(pos_count_)
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

### for coco
model, class_number=load_model("final_model_balanced_coco_5508_checkpoint.pth")

### for voc
# model, class_number=load_model("final_model_balanced_voc_81564_checkpoint.pth")

# After every epoch, validation
class_AP = eval(model, 0,phase='val')
x_axis=range(1,class_number+1)
class_AP_sorted=pd.DataFrame(class_AP.reshape(1,class_number),columns=x_axis).sort_values(by=[0],axis=1,ascending=False)
y=class_AP_sorted.to_numpy()[0]
x=class_AP_sorted.columns.to_numpy().astype(str)

### for coco
# plt.figure()
# fig, ax1=plt.subplots(figsize=(16,4))
# bwith=1
# width = 0.2
# frame=plt.gca()
# frame.spines["bottom"].set_linewidth(bwith)
# frame.spines["left"].set_linewidth(bwith)
# color1="#4472c4"
# color2="#eb7d32"
# cm = plt.get_cmap('RdYlGn')
# no_points=len(x_axis)
# bar1=ax1.bar(x,y,color=[cm(0.822-0.272*(i/(no_points))) for i in range(no_points)],edgecolor="black",linewidth=0)
# ax1.set_ylabel("AP",color="black", fontsize = 15)
# ax1.set_xlabel("Class Index",color="black", fontsize = 15)
# ax1.tick_params(axis="x",labelrotation=270)
# # frame.spines["left"].set_color(color2)
# # frame.spines["bottom"].set_color(color2)
# for axis in [ax1.xaxis]:
#     axis.set_major_locator(MultipleLocator(2))

# for axis in [ax1.xaxis]:
#     axis.set_major_locator(MaxNLocator(integer=False))
# ax1.set_xticks(x)

# # ax1.set_xticks(index)
# for label in ax1.get_yticklabels():
# 	label.set_fontsize(10)
# for label in ax1.get_xticklabels():
#     label.set_fontsize(10)
# plt.ylim(0.0,1.0)
# plt.savefig("test_5.pdf",format="pdf",dpi=600,bbox_inches='tight')
# plt.close()


### for voc
plt.figure()
fig, ax1=plt.subplots(figsize=(6,4))
bwith=1
width = 0.2
frame=plt.gca()
frame.spines["bottom"].set_linewidth(bwith)
frame.spines["left"].set_linewidth(bwith)
color1="#4472c4"
color2="#eb7d32"
cm = plt.get_cmap('RdYlGn')
no_points=len(x_axis)
bar1=ax1.bar(x,y,color=[cm(0.822-0.272*(i/(no_points))) for i in range(no_points)],edgecolor="black",linewidth=0)
# bar1=ax1.bar(x,y,color=color1,edgecolor="black",linewidth=0)
ax1.set_ylabel("AP",color="black", fontsize = 12)
ax1.set_xlabel("Class Index",color="black", fontsize = 12)
# ax1.tick_params(axis="x",labelrotation=270)
# frame.spines["left"].set_color(color2)
# frame.spines["bottom"].set_color(color2)
for axis in [ax1.xaxis]:
    axis.set_major_locator(MultipleLocator(2))

for axis in [ax1.xaxis]:
    axis.set_major_locator(MaxNLocator(integer=False))
ax1.set_xticks(x)

# ax1.set_xticks(index)
for label in ax1.get_yticklabels():
	label.set_fontsize(8)
for label in ax1.get_xticklabels():
    label.set_fontsize(8)
plt.ylim(0.0,1.0)
plt.savefig("test_6.pdf",format="pdf",dpi=600,bbox_inches='tight')
plt.close()


# /hdd8//dataset/coco/images/val2017/000000105014.jpg ['fork' 'bowl' 'broccoli' 'carrot' 'diningtable'] 0.6984739303588867, 0.7644844651222229, 0.8020085692405701, 0.7355707883834839, 0.8707544207572937
# /hdd8//dataset/coco/images/val2017/000000437331.jpg  ['person' 'surfboard']  0.9187266230583191, 0.7935048341751099
# /hdd8//dataset/coco/images/val2017/000000265518.jpg, ['cup' 'fork' 'sandwich' 'broccoli']  0.8025175929069519, 0.8183766007423401, 0.5088521242141724, 0.6595704555511475
# /hdd8//dataset/coco/images/val2017/000000015497.jpg, ['cat' 'sofa' 'mouse']  0.9330663681030273, 0.6684539318084717, 0.624189019203186
# /hdd8//dataset/coco/images/val2017/000000545100.jpg, ['person' 'car' 'traffic_light' 'cell_phone'] 0.9451642632484436, 0.867201566696167, 0.7369737029075623, 0.6299712657928467
# /hdd8//dataset/coco/images/val2017/000000303908.jpg, ['car' 'boat' 'bench'] 0.6625676155090332, 0.5548955202102661, 0.8564484119415283
# /hdd8//dataset/coco/images/val2017/000000458255.jpg, ['person' 'cat' 'bed' 'book'] , 0.7797839045524597, 0.7348338961601257, 0.8646310567855835, 0.6706752777099609
# /hdd8//dataset/coco/images/val2017/000000251572.jpg, ['person' 'dog' 'sofa'] 0.8033306002616882, 0.7213786244392395, 0.6395539045333862