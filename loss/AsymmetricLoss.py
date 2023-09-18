import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, distribution_path=None):
        super(AsymmetricLoss, self).__init__()
        self.distribution_path=distribution_path
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
        with open('/home//project/NLT-multi-label-classification/dataset/coco/longtail2017/distribution.txt') as f:
            for line in f:
                list_temp=line.replace(" ","").split(",")
        list_distribution=list(map(int,list_temp))
        num = sum(list_distribution)
        prob = [i/num for i in list_distribution]
        prob = torch.FloatTensor(prob)
        # # normalization
        max_prob = prob.max().item()
        prob = prob / max_prob
        # # class reweight
        weight = pow(- prob.log() + 1, 1/3)
        # print (weight)
        self.weight=weight.cuda()


    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.pow(torch.sigmoid(x),1) 
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # loss=loss*self.weight
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
def create_loss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, distribution_path=None):
    return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, distribution_path=None)

