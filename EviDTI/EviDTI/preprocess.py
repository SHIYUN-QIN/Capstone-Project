"""
根据文件顺序读取蛋白质与小分子表征，构建蛋白小分子输入文件
"""
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
# path！！！
interaction_file = 'dataset/bisai/bisai_HY-QS_64_info.csv'
interaction_list = pd.read_csv(interaction_file)
t_list = interaction_list['uid'].tolist()
d_list = interaction_list['cid'].tolist()
labels = interaction_list['label'].tolist()
smile_list = interaction_list['SMILES'].tolist()

# label_list = np.save('labels.npy', labels)

# 靶点信息写入
def read_target_info(t_list):
    # path！！！
    t_feature = np.load('dataset/bisai/protein_emb.npy', allow_pickle=True).item()
    # t_feature = t_feature.item()
    # t_feature = t_feature.squeeze()
    t_feature_list = []
    for id in t_list:
    # for id in range(202):
        t_feature_list.append(t_feature[id])
        # print(t_feature_list)
    # print(len(t_feature_list))
    t_feature_list = np.array(t_feature_list,dtype=object)
    # path！！！
    np.save('dataset/bisai/protein_feature.npy', t_feature_list)
    # 检查文件是否正确写入
    # t_feature = np.load('dataset/case_drugbank/protein_1d_feature', allow_pickle=True)
    # print(t_feature.shape)

# # 药物2D信息写入
def read_drug_2D_info():
    d_feature_list = []
    # path！！！
    d_feature = np.load('dataset/bisai/bisai_HY_QS_64_2d_embeddings.npy', allow_pickle=True).item()
    for id in d_list:
        d_feature_list.append(d_feature[id])
    # d_name_list = pd.read_csv('dataset/shunxu_cid.csv')
    # d_name_list = d_name_list['cid'].tolist()
    # d_feature_last = []
    # for d in tqdm(d_list):
    #     # index = d_name_list.index(d)
    #     d_feature_last.append(d_feature_list[index])
    # for
    # path！！！
    np.save('dataset/bisai/HY_QA_64_drug_2d_feature', d_feature_list)
    # print(d_feature_list.shape)
    print(len(d_feature_list))

# 药物3D信息写入
def read_drug_3D_info():
    metadata_file = smile_list
    # smiles_list = metadata_file['smile'].tolist()
    d_feature_list = []
    # path！！！
    d_feature = np.load('dataset/bisai/bisai_HY_QS_3D_emb_64.npy', allow_pickle=True)
    # d_name_list = pd.read_csv('dataset/shunxu_cid.csv')
    # d_name_list = d_name_list['cid'].tolist()
    d_feature_last = []
    for smile_1 in tqdm(metadata_file):
        mark = 0
        for smile_2 in d_feature:
            if smile_2!=None:
                if smile_1==smile_2['smiles']:
                    d_feature_last.append(smile_2)
                    mark = 1
                    break
        # if mark==0:
        #     print(smile_1)
    # for
    # path！！！
    np.save('dataset/bisai/HY_QA_64_drug_3d_feature', d_feature_last)
    print(len(d_feature_last))

# # 写元数据文件
def prepare_metadata():
    d_info_list = pd.read_csv('dataset/shunxu_cid.csv')
    d_id_list = d_info_list['cid'].tolist()
    d_smile_list = d_info_list['SMILES'].tolist()
    t_info_list = pd.read_csv('dataset/DTI_seq_info.csv')
    t_id_list = t_info_list['id'].tolist()
    t_seq_list = t_info_list['input'].tolist()
    dti_index_file = 'dataset/total_cid_unid_csv.csv'
    dti_info = pd.read_csv(dti_index_file)
    dti_info = np.array(dti_info)
    # dti_d_id = dti_info['cid'].tolist()
    # dti_t_id = dti_info['Uniprot ID'].tolist()
    metadata_list = []
    for dti in tqdm(range(len(dti_info))):
        index_d = d_id_list.index(dti_info[dti][0])
        index_t = t_id_list.index(dti_info[dti][1])
        temp = [d_id_list[index_d], d_smile_list[index_d], t_id_list[index_t], t_seq_list[index_t]]
        metadata_list.append(temp)
    metadata_csv = pd.DataFrame(metadata_list)
    metadata_csv.to_csv('metadata.csv')

# 根据uniprotid筛选结构
def coll_seq():
    path = 'D:\\DTI数据\\uni\\uni\\'
    seq_path = 'dataset/DTI_seq_info.csv'
    seq_list = pd.read_csv(seq_path)
    seq_list = seq_list['id'].tolist()
    str_list = []

    for filename in os.listdir(r'D:\\DTI数据\\uni\\uni\\'):
        str_list.append(filename)
    i = 0
    for id in str_list:
        if id.split('.')[0] not in seq_list:
            os.remove(path + str(id).rstrip("\n"))
            print(id)
            i+=1
    print(i)

# coll_seq()
# read_drug_3D_info()
# read_target_info(t_list)
read_drug_2D_info()


def read_case_drug_info():
    d2_feature = np.load('dataset/case_study/ACE_d_2d_feature.npy', allow_pickle=True)
    d3_feature = np.load('dataset/case_study/ACE_d_3d_feature.npy', allow_pickle=True)
    metadata_file = smile_list
    d2_feature_list_temp = []
    d2_feature_list = []
    d3_feature_list = []
    for i in d2_feature:
        for j in i:
            d2_feature_list_temp.append(j)
    k = 0
    for smile_1 in metadata_file:
        if len(smile_1)<300:
            for smile_2 in d3_feature:
                if smile_2!=None:
                    if smile_1==smile_2['smiles']:
                        d3_feature_list.append(smile_2)
                        d2_feature_list.append(d2_feature_list_temp[k])
                        k += 1
                        break
    np.save('dataset/case_study/ACE_d_2d_feature_processed.npy', d2_feature_list)
    np.save('dataset/case_study/ACE_d_3d_feature_processed.npy', d3_feature_list)

# read_case_drug_info()