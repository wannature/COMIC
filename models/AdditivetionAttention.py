from torch import nn
import math
import torch

def masked_softmax(X, valid_lens):
    if(valid_lens is None):
        return nn.functional.softmax(X, dim=-1)
    else:
        shape=X.shape
        if(valid_lens.dim())==1:
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
        X=d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return  nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditivetionAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditivetionAttention, self).__init__()
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # Q :1 × dk，K:n × dk，V:n × dv
        queries, keys =self.W_q(queries), self.W_k(keys)
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        scores=self.w_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

def create_model(key_size, query_size, num_hiddens, dropout=0.1, **kwargs):
    return  AdditivetionAttention(key_size, query_size, num_hiddens, dropout=0.1, **kwargs)