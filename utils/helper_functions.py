import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
       

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


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
        # if k ==0:
        #     print("person")
        #     print (ap[k])
        # print (ap[k])
        if per_class_number[k]>=many_shot_thr:
            many_shot.append(ap[k])
        elif per_class_number[k]<low_shot_thr:
            low_shot.append(ap[k])
        else:
            median_shot.append(ap[k])
    # print ("many_shot'number is " +str(len (many_shot)))
    # print ("median_shot'number is " +str(len (median_shot)))
    # print ("low_shot'number is " +str(len (low_shot)))
    return 100 * ap.mean(), 100 *np.mean(many_shot), 100 *np.mean(median_shot), 100 *np.mean(low_shot)


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        # self.ids= list(self.coco.imgToAnns.keys())
        self.ids =  [int(x.lstrip('0')[0:-4]) for x in os.listdir(self.root)]
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]



def plot_class_freq(stat_orig, stat_sim):
    plt.figure()
    plt.plot(stat_orig, label="original")
    plt.plot(stat_sim, label="simulated")
    plt.xlabel("Class index")
    plt.ylabel("Class frequency")

    path_dest = "./outputs"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    plt.savefig(os.path.join(path_dest, "class_freq.png"))


def simulate_coco(args, dataset_train, mode="fix_per_class", param=1000):
    ''' Possible modes:
        random_per_sample (param: remove percentage, 0.1, 0.2,...)
        fix_per_class (param: number of pos/neg samples per class, 1000, 2000,...)
    '''

    # Parameters
    mode = args.simulate_partial_type
    param = args.simulate_partial_param
    save_class_frequencies = False

    targets_vec = dataset_train.targets_all
    S = np.array([y.numpy() for x, y in targets_vec.items()])
    # S = np.array([y.max(dim=0)[0].numpy() for x, y in targets_vec.items()])
    img_ids = list(dataset_train.targets_all.keys())

    # Original samples
    num_samples = S.sum(axis=0)
    stat_orig = num_samples / S.shape[0]
    print("Original stat:", stat_orig[:10])

    if mode == "fix_per_class" or mode == "fpc":
        print("Simulate coco. Mode: %s. Param: %f" % (mode, param))

        max_pos = int(param)
        max_neg = int(param)
        add_one_label = False
        Sout = -np.ones_like(S)
        for c in range(S.shape[1]):
            s = S[:, c]
            idx_pos = np.where(s == 1)[0]
            idx_neg = np.where(s == 0)[0]
            idx_select_pos = np.random.choice(idx_pos, np.minimum(max_pos, len(idx_pos)), replace=False)
            idx_select_neg = np.random.choice(idx_neg, np.minimum(max_neg, len(idx_neg)), replace=False)
            Sout[idx_select_pos, c] = 1
            Sout[idx_select_neg, c] = 0

        if add_one_label:
            # Add one positive label in case of no-positive labels found in sample (the same for negative)
            for i, x in enumerate(Sout):
                if not np.any(x == 1):
                    idx_pos = np.where(S[i] == 1)[0]
                    idx_select_pos = np.random.choice(idx_pos, 1)
                    Sout[i, idx_select_pos] = 1
                if not np.any(x == 0):
                    idx_neg = np.where(S[i] == 0)[0]
                    idx_select_neg = np.random.choice(idx_neg, 1)
                    Sout[i, idx_select_neg] = 0
        S = Sout

    elif mode == "random_per_sample" or mode == "rps":
        print("Simulate coco. Mode: %s. Param: %f" % (mode, param))

        idx_random = np.random.random((S.shape)) < param
        S[idx_random] = -1

    # Assign in sampler
    targets_all = dict(zip(img_ids, S))
    dataset_train.targets_all = targets_all

    # Simulated class frequencies
    num_samples = (S == 1).sum(axis=0)
    stat_simulate = num_samples / S.shape[0]
    print("Simulated stat:", stat_simulate[: 5])

    return dataset_train
