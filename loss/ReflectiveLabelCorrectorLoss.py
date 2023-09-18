import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ReflectiveLabelCorrectorLoss(nn.Module):
    def __init__(self,
                 tau: float = 0.7,
                 compute_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 batch_size: int = 32,
                #  distribution_path: str = '/home//project/NLT-multi-label-classification/dataset/coco/longtail2017/distribution.txt',
                 distribution_path: str = None,
                 reduction: str = 'sum') -> None:
        super(ReflectiveLabelCorrectorLoss, self).__init__()
        self.tau = tau
        self.compute_epoch = compute_epoch
        self.margin = margin
        self.gamma = gamma
        self.batch_size=batch_size
        self.clip =0.05
        self.reduction = reduction
        self.disable_torch_grad_focal_loss=True
        self.distribution_path=distribution_path
        with open(str(self.distribution_path)) as f:
            for line in f:
                list_temp=line.replace(" ","").split(",")
        list_distribution=torch.tensor(list(map(int,list_temp)))
        self.distribution=list_distribution
        self.original_distribution=list_distribution.clone()
        self.num = sum(list_distribution)
        self.epoch=0
        self.total_TP=0
        self.total_FP=0
        prob = [i/self.num for i in list_distribution]
        prob = torch.FloatTensor(prob)
        self.eps=1e-8
        self.gamma_pos=0
        self.gamma_neg=3
        # # normalization
        max_prob = prob.max().item()
        prob = prob / max_prob
        # # class reweight
        weight = pow(- prob.log() + 1, 1/6)
        self.weight=weight.cuda()
        self.update_distribution=list_distribution

    def update_weight(self, distribution, targets):
       
        targets=targets.cpu().detach()
        for temp_list in targets:
            distribution+=temp_list
        prob = [i/self.num for i in distribution]
        prob = torch.FloatTensor(prob)
        # # normalization
        max_prob = prob.max().item()
        prob = prob / max_prob
        # # class reweight
        weight = pow(- prob.log() + 1, 1/6)
        self.weight=weight.cuda()
        self.update_distribution=distribution
        return distribution

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch=None, ground_truth=None) -> torch.Tensor:
        # Subtract margin for positive logits
        original_targets=targets
        logits = torch.where(targets == 1, logits-self.margin, logits)
        targets_number=(targets==True).sum()
        prediction=torch.sigmoid(logits)
        average_prediction=prediction.mean(dim=0)
        average_prediction=average_prediction.repeat(prediction.shape[0], 1)
        if epoch >= self.compute_epoch:
            targets = torch.where(
                (prediction > self.tau) & (torch.gt(prediction, average_prediction)),
                torch.tensor(1).cuda(), targets)
            sum_targets=(targets==1).sum()
            missing_targets=targets-original_targets
        ## get the TN and TP
        ## get the TN and TP of each classs and then get them
        ## get the labels of the samples in the batch
        # compute TP
            # temp_ground_truth=np.where(ground_truth.reshape(1,-1)==1)[1]
            # temp_preds=np.where(targets.cpu().reshape(1,-1)==1)[1]
            # sum=0
            # for i in range(len(temp_preds)):
            #     if(temp_preds[i] in temp_ground_truth):
            #         sum+=1
            # TP=sum
            # temp_ground_truth=np.where(ground_truth.reshape(1,-1)==0)[1]
            # temp_preds=np.where(targets.cpu().reshape(1,-1)==1)[1]
            # sum=0
            # for i in range(len(temp_preds)):
            #     if(temp_preds[i] in temp_ground_truth):
            #         sum+=1
            # FP=sum
            # # print("epoch %d : Tp is %f, FP is %f" %(epoch, TP, FP))
            # TP=TP-targets_number
            # FP=FP
            if self.epoch<epoch:
                self.epoch=epoch
                # self.total_TP=TP
                # self.total_FP=FP
                self.distribution=self.original_distribution.clone()
            # else:
            #     self.total_TP+=TP
            #     self.total_FP+=FP

            
            self.update_weight(self.distribution, missing_targets)
            xs_pos = prediction
            xs_neg = 1 - prediction
            if self.clip is not None and self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1)
            los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
            los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
            loss = los_pos + los_neg
            loss*=self.weight
            if self.gamma_neg > 0 or self.gamma_pos > 0:
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(False)
                pt0 = xs_pos * targets
                pt1 = xs_neg * (1 - targets) 
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos)* targets + (self.gamma_neg+self.weight)* (1 - targets)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(True)
                loss *= one_sided_w
                loss *=(self.batch_size/sum_targets)
                loss=-loss.sum()
                return loss
            else:
                return 0
        return 0
