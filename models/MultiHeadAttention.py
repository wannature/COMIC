import torch.nn as nn
from utils import *
from os import path

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim=1024*3, num_heads=3, *args):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        

    def forward(self, query, key, value, *args):
        x, attention_score = self.multihead_attn(query, key, value)
        return x, attention_score

def create_model(embed_dim=3072,  num_heads=3, dataset=None, test=False,*args):
    return  MultiHeadAttention(embed_dim, num_heads,*args)