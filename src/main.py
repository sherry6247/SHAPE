import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import  torch.optim as optim
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace

import sys
sys.path.append("..")
from shape_model import SHAPE
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result
from recommend import eval, test

# torch.manual_seed(1203)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name = 'Set_GMed'
# resume_path = ''

# if not os.path.exists(os.path.join("saved", model_name)):
#     os.makedirs(os.path.join("saved", model_name))

"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power, current_length):
    # ratio_length = 1 - (float(current_length) / 30) 
    # iter = iter + ratio_length
    iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))#+ (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

def adjust_learning_rate(optimizer, i_iter, args, current_length):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power, current_length)
    optimizer.param_groups[0]['lr'] = np.min(np.around(lr,8))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default='SHAPE', help="model name")
parser.add_argument('--resume_path', type=str, default='', help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--threshold', type=float, default=0.3, help='the threshold of prediction')
parser.add_argument('--ln', type=int, default=0, help='layer normlization')
parser.add_argument('--seed', type=int, default=1203)
parser.add_argument('--device', type=str, default='1', help='Choose GPU device')
parser.add_argument('--kgloss', type=float, default=0.001, help='Choose GPU device')



args = parser.parse_args()

def main(args):
    # load data
    data_path = "../data/records_final.pkl"
    voc_path = "../data/voc_final.pkl"

    # ehr_adj_path = '../data/weighted_ehr_adj_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    # device = torch.device('cuda')
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda')
    args.device = device
    torch.manual_seed(args.seed)

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    # # sorted data according the visit length
    # def sorted_data(data):
    #     # data_len = [len(i) for i in data]
    #     data = sorted(data, key=lambda x:len(x))
    #     return data
    # data_train = sorted_data(data_train)
    # data_eval = sorted_data(data_eval)
    # data_test = sorted_data(data_test)

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)
    total_dataset = mimic_data(data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)
    total_dataloader = DataLoader(total_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    if args.model_name == 'SHAPE':
        model = SHAPE(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.emb_dim, device=device, dim_hidden=args.emb_dim, ln=args.ln, kgloss_alpha=args.kgloss)
    else:
        print("MODEL NAME ERROR!!!")
    
    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        smm_record, ja, prauc, precision, recall, f1, med_num = test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, args)
        # smm_record, ja, prauc, precision, recall, f1, med_num = test(model, total_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, args)
        result = []
        for _ in range(10):
            data_num = len(ja)
            final_length = int(0.8 * data_num)
            idx_list = list(range(data_num))
            random.shuffle(idx_list)
            idx_list = idx_list[:final_length]
            avg_ja = np.mean([ja[i] for i in idx_list])
            avg_prauc = np.mean([prauc[i] for i in idx_list])
            avg_precision = np.mean([precision[i] for i in idx_list])
            avg_recall = np.mean([recall[i] for i in idx_list])
            avg_f1 = np.mean([f1[i] for i in idx_list])
            avg_med = np.mean([med_num[i] for i in idx_list])
            cur_smm_record = [smm_record[i] for i in idx_list]
            ddi_rate = ddi_rate_score(cur_smm_record, path=ddi_adj_path)
            # ddi_rate = 0.
            result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
            llprint('\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                    ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} + {:.4f}\t".format(m, s)

        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr) #, weight_decay=0.01

    args.power = 0.9
    args.num_steps = 50000 #50000
    args.weight_decay = 0.0005
    args.momentum = 0.9
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 使用SGD作为优化function
    # optimizer = optim.SGD(model.parameters(),
    #                       args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 200
    cu_iter = 0
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} seed : {} model : {} --------------------------'.format(epoch, args.seed, args.model_name))
        model.train()
        for idx, data in enumerate(train_dataloader):
            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            # medications = medications.to(device)
            medications = pad_num_replace(medications, -1, MED_PAD_TOKEN).to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)

            """
            下面增加adjust_learning_rate的代码
            """
            cu_iter +=1
            adjust_learning_rate(optimizer, cu_iter, args, np.sum(np.array(seq_length))/args.batch_size)
            output_logits, kgddi_loss = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            
            # 需要对输出部分pad部分处理掉
            # labels, predictions = output_flatten(target_list, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, max_len=args.max_len)
            bce_target = np.zeros([medications.shape[0], medications.shape[1], voc_size[2]])
            for b_i, med in enumerate(target_list):
                for v_i, m in enumerate(med):
                    bce_target[b_i, v_i, m] = 1
            labels = torch.Tensor(bce_target).to(device)
            loss = F.binary_cross_entropy_with_logits(output_logits, labels)
            loss += kgddi_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            llprint('\rtraining step: {} / {} loss:{:.4f} lr:{} sample_len:{}'.format(idx, len(train_dataloader), loss.item(), optimizer.param_groups[0]['lr'], diseases.shape))

        print ()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, epoch, device, TOKENS, args)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print (' Med: {}, Ja: {}, F1: {} PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        saved_path = os.path.join("./saved", args.model_name, 'seed_{}_lr_{}_hs_{}_ln_{}'.format(args.seed, args.lr, args.emb_dim, int(args.ln)))
        
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            torch.save(model.state_dict(), open(os.path.join(saved_path, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

        # dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))
        dill.dump(history, open(os.path.join(saved_path, 'history_{}.pkl'.format(args.model_name)), 'wb'))
        if epoch - best_epoch > 10:
            break


if __name__ == '__main__':
    main(args)
