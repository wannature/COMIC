import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helper_functions import shot_mAP

class HeadTailBalancerLoss(nn.Module):
    def __init__(self, gamma):
        super(HeadTailBalancerLoss, self).__init__()
        self.PFM = None
        self.gamma = 2

    def forward(self, head, tail, balance, labels):
        # calculate weight
        head_acc= self.PFM(head, labels)
        tail_acc = self.PFM(tail, labels)
        head_acc=torch.pow(head_acc,self.gamma)
        tail_acc=torch.pow(tail_acc,self.gamma)
        head_weight = head_acc/(head_acc + tail_acc)
        tail_weight= tail_acc/(head_acc + tail_acc)     
        #head loss
        prob_head = F.softmax(head, -1).clone().detach()
        prob_balance = F.softmax(balance, -1)
        head_loss=self.PFM(prob_head*prob_balance, labels)


        
        # #tail loss
        prob_tail = F.softmax(tail, -1).clone().detach()
        tail_loss=self.PFM(prob_tail*prob_balance, labels)

        loss = (head_weight*head_loss).mean() + (tail_weight*tail_loss).mean()
        return loss
