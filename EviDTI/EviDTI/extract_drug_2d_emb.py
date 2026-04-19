import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from dataset_dti import Graph_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import  PredictModel,BertModel


#import ipdb

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


embeddings = tf.zeros([32,256])
def main(seed):
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['BBB', 'FDAMDD',  'Pgp_sub']

    #task = 'FDAMDD'
    #task = 'ours'
    #task = 'hesGroup-process'
    #task = 'KIBA-test'
    task = 'HY_QS_64_drug_info'
    print(task)

    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
    medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
    large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights','addH':True}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    train_dataset, test_dataset , val_dataset = Graph_Classification_Dataset('data/clf/{}.csv'.format(task), smiles_field='SMILES',
                                                               label_field='Label',addH=True).get_data()
    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    # print(x)
    #ipdb.set_trace()
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.5)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')

    y_true = []
    y_preds = []
    preds_list = []
    drug_feature_dic = {}
    id = pd.read_csv('./data/clf/HY_QS_64_drug_info.csv')['ID']
    for x, adjoin_matrix, y in val_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        # print(x)
        #ipdb.set_trace()
        preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
        # np.asarray(preds)
        preds_list.append(preds)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    np.asarray(preds_list)
    y_true = np.concatenate(y_true,axis=0).reshape(-1)
    index = 0
    for i, pred in enumerate(preds_list):
        for j, p in enumerate(pred):
            drug_feature_dic[id.values[index]] = p
            index+=1
    np.save('bisai_HY_QS_64_2d_embeddings.npy', drug_feature_dic)


if __name__ == '__main__':

    auc_list = []
    #for seed in [7,17,27,37,47,57,67,77,87,97]:
    #    print(seed)
    #    auc = main(seed)
    #    auc_list.append(auc)
    seed = 5
    print(seed)
    auc = main(seed)
    auc_list.append(auc)

    print(auc_list)



