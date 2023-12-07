import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.modules.linear import Linear
# from data.processing import process_visit_lg2

from layers import SelfAttend
from layers import GraphConvolution

from modules import *
from block_recurrent_transformer.transformer import BlockRecurrentAttention

class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class policy_network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(policy_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class SHAPE(nn.Module):
    """这个将DDI表示放在输出层并，增加一个DDI loss"""
    def __init__(self, voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=128, device=torch.device('cpu:0'), num_inds=32, dim_hidden=128, num_heads=2, ln=False, isab_num=2, kgloss_alpha=0.001):
        super(SHAPE, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = num_heads
        self.ddi_adj = ddi_adj
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2]+2      # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2

        self.isab_num = isab_num
        self.num_inds = num_inds

        # num_outputs = k seed vector
        num_outputs = voc_size[2]
        dim_output = 1
        # num_outputs = 1
        # dim_output = voc_size[2]

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        # dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0]+3, emb_dim, self.DIAG_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1]+3, emb_dim, self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # 添加padding_idx，表示取0向量
            nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # 这里采用的set-encoder模块
        self.isab = ISAB(emb_dim, dim_hidden, num_heads, num_inds, ln=ln)
        self.diag_enc = nn.Sequential(
                ISAB(emb_dim, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.proc_enc = nn.Sequential(
                ISAB(emb_dim, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.med_enc = nn.Sequential(
                ISAB(emb_dim, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        # 这里使用recurrent-transformer来编码visit-level的时序信息
        self.dim_hidden = dim_hidden
        self.recurrent_attn = BlockRecurrentAttention(dim_hidden*3, dim_hidden*3)
        # self.gru = nn.GRU(dim_hidden*3, dim_hidden*3, batch_first=True, num_layers=2)
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden*3, num_heads, num_outputs, ln=ln),
        #         SAB(dim_hidden*3, dim_hidden*3, num_heads, ln=ln),
        #         SAB(dim_hidden*3, dim_hidden*3, num_heads, ln=ln),
        #         nn.Linear(dim_hidden*3, dim_output))
                #
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(dim_hidden*3, voc_size[2])
        
        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], voc_size[2], False)        

        # 加入DDI图
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.kgloss_alpha = kgloss_alpha
    
    def set_encoder(self, input, attn_mask, rep=2):
        attn_mask = attn_mask.unsqueeze(2).repeat(1,1,self.num_inds, 1)
        for i in range(rep):
            input = self.isab([input, attn_mask])
        return input


    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        # 1. 首先计算code-level的embedding
        diag_emb = self.diag_embedding(diseases) # [batch_size, diag_code_len, emb]
        proc_emb = self.proc_embedding(procedures) # [batch_size, proc_code_len, emb]
        
        # 2. 由于medication需要增加一个padding的visit记录。构造一个new_medication，表示上一个visit的记录，然后与[0,t-1]时刻的medication进行拼接
        new_medication = torch.full((batch_size, 1, max_med_num), self.MED_PAD_TOKEN).to(device)
        new_medication = torch.cat([new_medication, medications[:, :-1, :]], dim=1) # new_medication.shape=[b,max_seq_len, max_med_num]
        # m_mask_matrix 同样也需要移动
        new_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device) # 这里用较大负值，避免softmax之后分走了概率
        new_m_mask = torch.cat([new_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        med_emb = self.med_embedding(new_medication)

        # # 3. 在code-level采用set-encoder进行编码
        # # 3.1 diagnosis 的set-encoders
        d_enc_mask_matrix = d_mask_matrix.view(batch_size*max_seq_length, max_diag_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_num, nhead, diag_len]
        diag_enc_input = diag_emb.view(batch_size*max_seq_length, max_diag_num, -1)
        diag_encode = self.set_encoder(diag_enc_input, d_enc_mask_matrix) # [batch_size*visit_len, diag_len, hdm]
        # 3.2 procedure的set-encoders
        p_enc_mask_matrix = p_mask_matrix.view(batch_size*max_seq_length, max_proc_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_num, nhead, proc_len]
        proc_enc_input = proc_emb.view(batch_size*max_seq_length, max_proc_num, -1)
        proc_encode = self.set_encoder(proc_enc_input, p_enc_mask_matrix)  # [batch_size*visit_len, proc_len, hdm]
        # 3.3 medication的set-encoders
        m_enc_mask_matrix = new_m_mask.view(batch_size*max_seq_length, max_med_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_len, nhead, med_len]
               
        # 3.4. 得到ehr graph 和 DDI graph的表示
        ehr_embedding, ddi_embedding = self.gcn() # [vocab_size, hdm]
        drug_memory = ehr_embedding - ddi_embedding * self.inter # 采用共现图-DDI的图表示来
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float() # 特殊字符
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)# [vocab_size, hdm]

        # # 3.5 直接将两种medication code进行相加
        # med_memory_emb = drug_memory[new_medication] # [batch_size, max_seq_length, med_code_len, hdm]
        # med_emb = med_emb + med_memory_emb
        
        m_enc_input = med_emb.view(batch_size*max_seq_length, max_med_num, -1)
        med_encode = self.set_encoder(m_enc_input, m_enc_mask_matrix) # [batch_size, max_seq_length, med_code_len, hdm]
        
        # 4.将三种code分别aggregate，转化为visit-level
        diag_enc = torch.sum(diag_encode, dim=1).view(batch_size, max_seq_length, -1)
        proc_enc = torch.sum(proc_encode, dim=1).view(batch_size, max_seq_length, -1)
        med_enc = torch.sum(med_encode, dim=1).view(batch_size, max_seq_length, -1)
        visit_enc = torch.cat([diag_enc, proc_enc, med_enc], dim=-1) # [batch_size, max_seq_length, 3*hdm]

        # 5. 将visit-level经过recurrent-transformer
        visit_mask = torch.full((batch_size, max_seq_length), 0).to(device)
        state = torch.zeros((batch_size, max_seq_length, self.dim_hidden*3)).to(device)
        for i, v_l in enumerate(seq_length):
            visit_mask[i, :v_l] = 1
        output, state = self.recurrent_attn(visit_enc, state, mask=visit_mask.bool()) # 
        # output, state = self.gru(visit_enc)
        decoder_output = self.output_layer(output) # [batch_size, max_seq_length, vocab_size]
        decoder_output = decoder_output * visit_mask.unsqueeze(-1)

        # 6. 这里计算预测的药物序列的DDI然后以已知的DDI矩阵进行约束
        sigmoid_output = torch.sigmoid(decoder_output)
        sigmoid_output_ddi = torch.matmul(sigmoid_output.unsqueeze(2).transpose(-1, -2),sigmoid_output.unsqueeze(2)) # [batch_size, max_seq_length, vocab_size, vocab_size]
        kg_ddi = torch.from_numpy(self.ddi_adj).to(sigmoid_output.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_seq_length, 1, 1) # [batch_size, max_seq_length, vocab_siz, vocab_size]
        kg_ddi_score = 0.001 * self.kgloss_alpha * torch.sum(kg_ddi * sigmoid_output_ddi, dim=[-1,-2]).mean()

        return decoder_output, kg_ddi_score                
         
   