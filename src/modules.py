'''
Author: your name
Date: 2022-04-17 10:23:07
LastEditTime: 2022-10-11 02:29:17
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /useGeneration2Drug/generate2Drug/baseline_model/set_transformer/modules.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None):
        Q = self.fc_q(Q) 
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        ############################################################
        attn_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if attn_mask is not None:
            attn_mask = attn_mask.view_as(attn_score)
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_score += attn_mask
        ############################################################

        A = torch.softmax(attn_score, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, input):
        X, attn_mask = input
        return self.mab(X, X, attn_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.FloatTensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.num_inds = num_inds

    def forward(self, input):
        X, attn_mask = input
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attn_mask) # [B*V, num_inds, dim]
        attn_mask = attn_mask.transpose(-2, -1)
        return self.mab1(X, H, attn_mask)
    # def forward(self, X):
    #     H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
    #     return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim)) # [1, K, dim]
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X) #[batch_size*visit_len, K, dim]
