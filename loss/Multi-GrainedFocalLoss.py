import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class MultiGrainedFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, gamma_class_ng=1.2, clip=0.05, eps=1e-8, \
            disable_torch_grad_focal_loss=True, distribution_path=None, co_occurrence_matrix=None):
        super(MultiGrainedFocalLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_class_ng=gamma_class_ng
        self.gamma_class_pos=1
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.distribution_path=distribution_path

        
        

        # self.spls_loss=SPLC(batch_size=32)
    def create_weight(self, distribution_path):
        with open(self.distribution_path) as f:
            for line in f:
                list_temp=line.replace(" ","").split(",")
        list_distribution=list(map(int,list_temp))
        num = sum(list_distribution)
        prob = [i/num for i in list_distribution]
        prob = torch.FloatTensor(prob)
        max_prob = prob.max().item()
        prob = prob / max_prob
        weight = pow(- prob.log() + 1, 1/6)
        self.weight=weight.cuda()

    def create_co_occurrence_matrix(self, co_occurrence_matrix):
        co_occurrence_matrix=torch.tensor(np.load(co_occurrence_matrix)).cuda()
        self.co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)

    def forward(self, x, y):
        # postive -
        x_sigmoid = torch.pow(torch.sigmoid(x),1) 
        gamma_class_pos=1
        xs_pos = x_sigmoid*gamma_class_pos
        xs_neg = 1 - x_sigmoid
        

        # negtive +
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss*=self.weight
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = (self.gamma_pos)* y + (self.gamma_neg+self.weight)* (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        loss=-loss.sum()
        return loss

def create_loss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True):
    return MultiGrainedFocalLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

