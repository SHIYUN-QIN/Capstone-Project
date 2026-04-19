import argparse
import os
# import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
# from sklearn.metrics import roc_curve
from typing import List, Tuple
from torch_geometric.loader import DataLoader
# import matplotlib.pyplot as plt
from function import bond_angle_graph_data,KIBA_graph_data,case_graph_data,davis_graph_data

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 防止指数爆炸，减去最大值
    return exp_x / np.sum(exp_x, axis=0)

root = './dataset'
# bond_angle_graph_list = bond_angle_graph_data(root='./dataset/bond_angle/',t_1D_path='dataset/bond_angle/t_feature.npy',
#                                   d_2D_path='dataset/bond_angle/d_feature.npy',
#                                   d_3D_path='dataset/bond_angle/d_3D_feature.npy',
#                                   label_file='dataset/total_cid_unid_csv.csv',
#                                   metadata_path='dataset/bond_angle/metadata.csv')
# data_list = KIBA_graph_data('./dataset/KIBA/',
#                                 d_2D_path='dataset/KIBA/KIBA_d_2d_feature.npy',
#                                 t_1D_path = 'dataset/KIBA/KIBA_t_1D_feature.npy',
#                                 d_3D_path = 'dataset/KIBA/KIBA_d_3d_feature.npy',
#                                 label_file='dataset/KIBA/KIBA_total_cid_unid.csv',
#                                 metadata_path='dataset/KIBA/KIBA_total_cid_unid.csv')
# data_list = davis_graph_data('./dataset/davis/',
#                              t_1D_path='dataset/davis/davis_t_feature.npy',
#                              d_2D_path='dataset/davis/davis_d_2D_features.npy',
#                              d_3D_path='dataset/davis/davis_d_3d_feature.npy',
#                              label_file='dataset/davis/davis_total_cid_unid.csv',
#                              metadata_path='dataset/davis/davis_total_cid_unid.csv')
data_list = case_graph_data('./dataset/case_drugbank/',
                                d_2D_path='dataset/case_drugbank/drugbank_case_2d_emb.npy',
                                t_1D_path = 'dataset/case_drugbank/drugbank_protein_case_emb.npy',
                                d_3D_path = 'dataset/case_drugbank/drugbank_case_3D_emb.npy',
                                label_file='dataset/case_drugbank/drugbank_total.csv',
                                metadata_path='dataset/case_drugbank/drugbank_total.csv')
graph_dict = {}
# np.random.seed(42)
shuffled_indices = np.random.permutation(len(data_list))
# train_idx = shuffled_indices[:int(0.8 * len(data_list))]
# val_idx = shuffled_indices[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
test_idx = shuffled_indices
# train_loader = DataLoader(data_list, batch_size = 32, drop_last = True,
#                           sampler = SubsetRandomSampler(train_idx))
# val_loader = DataLoader(data_list, batch_size = 32, drop_last=False,
#                         sampler=SubsetRandomSampler(val_idx))
test_loader = DataLoader(data_list, batch_size = 32, drop_last=False,
                        sampler=SubsetRandomSampler(test_idx))
print('load data finish')

# model = LightAttention(embeddings_dim=1024)
# checkpoint = torch.load(os.path.join('D:\\xyt\\DTI_drugbank', 'checkpoint.pth'),
#                                     map_location='cuda:1')
# model.load_state_dict(checkpoint['model_state_dict'])
model = torch.load(os.path.join('./runs/drugbank_model',
                                'checkpoint.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Needs "from torch.optim import *" and "from models import *" to work
# solver = Solver(model,eval=True)
# solver.train(train_loader, val_loader)
# params = list(model.named_parameters())
# print(params[0])

preds = []
labels = []
t_id = []
d_id = []
att_list = []
for i, batch in enumerate(test_loader):
    metadata = batch.metadata
    # get label
    label_org = batch.l
    label_org = label_org.to("cuda")

    sequence_lengths = metadata['length'][:, None].to('cuda')  # [batchsize, 1]
    # frequencies = metadata['frequencies'].to(self.device)  # [batchsize, 25]

    # create mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
    mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,
                                                             None]  # [batchsize, seq_len]
    # batch_preds, att_AA = model(batch, mask=mask.to('cuda'),
    #                          sequence_lengths=sequence_lengths)  # [batchsize, 2]
    batch_preds = model(batch, mask=mask.to('cuda'),
                        sequence_lengths=sequence_lengths)  # [batchsize, 2]
    batch_preds = batch_preds.tolist()
    label_org = label_org.tolist()
    # att_AA = att_AA.tolist()
    id1 = metadata['id']
    id2 = metadata['d_id']
    preds.extend(batch_preds)
    labels.extend(label_org)
    t_id.extend(id1)
    d_id.extend(id2)
    # att_list.extend(att_AA)

    p = []
    c = []
    var = []
    ev = []
    bk_list = []
    prob_list = []

    for i in range(len(preds)):
        num_classes = 2

        alphas = preds[i]  # shape=(num_tasks * num_classes)
        num_tasks = len(alphas) // num_classes

        alphas = np.reshape(alphas, (num_tasks, num_classes))
        evidence = alphas - 1
        bk = alphas - 1 / np.sum(alphas, axis=-1)

        #证据深度学习使用
        probs = alphas / np.sum(alphas, axis=-1).reshape(num_tasks, 1)

        # 非证据深度学习使用
        # probs = np.array(preds[i])
        # softmax_output = softmax(probs)

        # final probability is just the prob of being active in
        # this task
        # probs = probs[:, 1]

        probs = np.squeeze(probs)
        if probs[0] >= probs[1]:
            pred = 0
        else:
            pred = 1

        # pred = np.max(probs[..., -2:], dim=1)[1]  # get indices of the highest value for sol


        p.append(np.stack([pred, labels[i]]))

        # 证据深度学习使用
        prob_list.append(probs[1])

        # 非证据深度学习使用
        # prob_list.append(softmax_output[1])

        conf = num_classes / np.sum(alphas, axis=-1)
        c.append(conf[0])
        ev.append(evidence)
        bk_list.append(bk)
        # TODO: std not implemented here
        var.append(conf[0])

# print(p,c,var)
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, \
    precision_score, accuracy_score, roc_auc_score, precision_recall_curve,auc,roc_curve
val_results = np.squeeze(p)
val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
c_m = confusion_matrix(val_results[:, 1], val_results[:, 0])
val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
val_auc = roc_auc_score(val_results[:, 1], prob_list)
val_recall = recall_score(val_results[:, 1], val_results[:, 0])
val_pre = precision_score(val_results[:, 1], val_results[:, 0])
fpr, tpr, tresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
precision, recall, _thresholds = precision_recall_curve(val_results[:, 1], prob_list)
val_prauc = auc(recall, precision)
print(val_acc,val_recall, val_pre, val_mcc,val_f1, val_auc, val_prauc,c_m)
# print(val_acc,val_recall, val_pre, val_mcc,val_f1, val_prauc,c_m)
lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.5f)' % val_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
result_id = []
result_id.append(t_id)
result_id.append(d_id)
result_id.append(prob_list)
result_id.append(p)
result_id.append(c)
result_id.append(var)
result_id.append(ev)
result_id.append(bk_list)
result_csv = pd.DataFrame(result_id)
result_csv.to_csv('./runs/drugbank_model/result_test.csv')
# att_csv = pd.DataFrame(att_list)
# att_csv.to_csv('/fs1/home/wangll/software/dti/DTI_drugbank/runs/la_test1_21-11_17-11-51/case_att_result.csv')
