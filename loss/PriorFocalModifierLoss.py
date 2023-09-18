import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from loss.SpccLoss import SPLC 
class PriorFocalModifierLoss(nn.Module):
    def __init__(self, gamma_neg=3, gamma_pos=1, gamma_class_ng=1.2, clip=0.05, eps=1e-8, \
            disable_torch_grad_focal_loss=True, distribution_path=None, co_occurrence_matrix=None):
        super(PriorFocalModifierLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_class_ng=gamma_class_ng
        self.gamma_class_pos=1
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.distribution_path=distribution_path

        
    
    def create_weight(self, distribution_path):
        with open(self.distribution_path) as f:
            for line in f:
                list_temp=line.replace(" ","").split(",")
                print (list_temp)
        list_distribution=list(map(int,list_temp))
        num = sum(list_distribution)
        prob = [i/num for i in list_distribution]
        prob = torch.FloatTensor(prob)
        max_prob = prob.max().item()
        prob = prob / max_prob
        weight = pow(- prob.log() + 1, 1/3)
        self.weight=weight.cuda()

    def create_co_occurrence_matrix(self, co_occurrence_matrix):
        co_occurrence_matrix=torch.tensor(np.load(co_occurrence_matrix)).cuda()
        self.co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)

    def forward(self, x, y):

        attention_scores_total=[]
        for k in range(y.shape[0]): 
            attention_scores=self.co_occurrence_matrix[y[k]==1].mean(dim=0)
            attention_scores=attention_scores/attention_scores.sum()
            attention_scores_total.append(attention_scores)
        final_attention_scores=torch.stack(attention_scores_total,0)
        # print (final_attention_scores)
        
        # postive -
        x_sigmoid = torch.pow(torch.sigmoid(x),1) 
        # gamma_class_pos=self.gamma_class_pos
        # print (gamma_class_pos)
        gamma_class_pos=self.gamma_class_pos-final_attention_scores
        # gamma_class_pos=1
        xs_pos = x_sigmoid*gamma_class_pos
        xs_neg = 1 - x_sigmoid
        

        # negtive +
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        xs_neg=torch.where(final_attention_scores==0, xs_neg, xs_neg*self.gamma_class_ng).clamp(max=1)

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
            # print (self.weight)
            one_sided_gamma = (self.gamma_pos)* y + (self.gamma_neg+self.weight)* (1 - y)
            # print (one_sided_gamma)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        loss=-loss.sum()
        # loss+=self.spls_loss(x,y,epoch)
        return loss

def create_loss(gamma_neg=3, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True):
    return PriorFocalModifierLoss(gamma_neg=3, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

