import os
from typing import Tuple

import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, \
            accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

from utils import dirichlet_loss, cal_top_hit_ratio


class Solver():
    def __init__(self, model,cfg, device, optim=torch.optim.Adam, loss_func=dirichlet_loss, eval=False):
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = cfg.SOLVER['BATCH_SIZE']
        self.epoch = cfg.SOLVER['MAX_EPOCH']
        self.lr = cfg.SOLVER['LR']
        self.weight_decay = cfg.SOLVER['WEIGHT_DECAY']
        self.loss_func_name = cfg.SOLVER['LOSS_FUNCTION']
        if self.loss_func_name == 'dirichlet_loss':
            self.loss_func = dirichlet_loss
        self.optim = optim(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=20,gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10)
        if eval:
            checkpoint = torch.load(os.path.join(''),
                                    map_location=self.device)
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format('la', 'test1',
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))
            # with open(os.path.join(self.writer.log_dir, 'epoch.txt'), "r") as f:  # last epoch not the best epoch
            #     self.start_epoch = int(f.read()) + 1
            self.max_val_acc = checkpoint['maximum_accuracy']
            # self.weight = checkpoint['weight'].to(self.device)

        if not eval:
            self.start_epoch = 0
            self.max_val_acc = 0  # running accuracy to decide whether or not a new model should be saved
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format('la', 'test1',
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))
            # self.weight = weight.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_data=None):
        """
        Train and simultaneously evaluate on the val_loader and then estimate the stderr on eval_data if it is provided
        Args:
            train_loader: For training
            val_loader: For validation during training
            eval_data: For evaluation and estimating stderr after training

        Returns:

        """
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        min_train_acc = 95
        max_train_acc = 5
        for epoch in range(self.start_epoch, self.epoch):  # loop over the dataset multiple times
            self.model.train()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                _,_,val_results,c,var,prob_list,_,_ = self.predict_val(val_loader, epoch + 1)

            val_results = np.array(val_results)
            val_results = np.squeeze(val_results)
            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            with warnings.catch_warnings():  # because sklearns mcc implementation is a little dim
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                train_mcc = matthews_corrcoef(train_results[:, 1], train_results[:, 0])
                val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
                val_auc = roc_auc_score(val_results[:, 1], prob_list)

            print('[Epoch %d] val accuracy: %.4f%% train accuracy: %.4f%%' % (epoch + 1, val_acc, train_acc))

            self.writer.add_scalars('Acc', {'train': train_acc, 'val': val_acc}, epoch + 1)
            self.writer.add_scalars('MCC', {'train': train_mcc, 'val': val_mcc}, epoch + 1)

            if val_acc >= self.max_val_acc:  # save the model with the best accuracy
                epochs_no_improve = 0
                self.max_val_acc = val_acc
                self.save_checkpoint(epoch + 1)
            else:
                epochs_no_improve += 1


            with open(os.path.join(self.writer.log_dir, 'epoch.txt'), 'w') as file:  # save what the last epoch is
                file.write(str(epoch))

            if train_acc >= max_train_acc:
                max_train_acc = train_acc
            if epochs_no_improve >= 30 and max_train_acc >= min_train_acc:  # stopping criterion
                break

        if eval_data:  # do evaluation on the test data if a eval_data is provided
            # load checkpoint of best model to do evaluation
            checkpoint = torch.load(os.path.join(self.writer.log_dir, 'checkpoint.pth'))
            self.model = checkpoint
            self.evaluation(eval_data, filename='val_data_after_training')

    # train
    def predict(self, data_loader: DataLoader, epoch: int = None, optim: torch.optim.Optimizer = None) -> \
            Tuple[float, np.ndarray]:

        results = []  # prediction and corresponding localization
        running_loss = 0
        for i, batch in enumerate(data_loader):
            metadata = batch.metadata
            # get label
            label_org = batch.l

            sequence_lengths = metadata['length'][:, None].to(self.device)  # [batchsize, 1]
            # frequencies = metadata['frequencies'].to(self.device)  # [batchsize, 25]

            # create mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,
                                                                         None]  # [batchsize, seq_len]
            label_org = label_org.to(self.device)
            label = torch.nn.functional.one_hot(label_org)
            label = label.to(torch.float16) # [batchsize, 2]
            prediction = self.model(batch, mask=mask.to(self.device), sequence_lengths=sequence_lengths) #[batchsize, 2]
            loss = self.loss_func(label, alphas=prediction, lam=0.1)
            # loss = loss * class_weights * mask
            loss = loss.sum()
            if optim:  # run backpropagation if an optimizer is provided
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            pred = torch.max(prediction[..., -2:], dim=1)[1]  # get indices of the highest value for sol

            results.append(torch.stack((pred,label_org),dim=1).detach().cpu().numpy())
            loss_item = loss.item()
            running_loss += loss_item

            if i % 100 == 99:  # log every log_iterations
                if epoch:
                    print('Epoch %d ' % (epoch), end=' ')
                print('[Iter %5d/%5d] %s: loss: %.7f, accuracy: %.4f%%' % (
                    i + 1, len(data_loader), 'Train' if optim else 'Val', loss_item,
                    100 * (pred==label_org).sum().item() / self.batch_size))

        running_loss /= len(data_loader)
        return running_loss, np.concatenate(results)  # [n_train_proteins, 2] pred and loc


    def predict_val(self, data_loader: DataLoader, epoch: int = None):
        preds = []
        labels = []
        t_id = []
        d_id = []
        for i, batch in enumerate(data_loader):
            metadata = batch.metadata
            # get label
            label_org = batch.l
            label_org = label_org.to(self.device)

            sequence_lengths = metadata['length'][:, None].to(self.device)  # [batchsize, 1]
            # frequencies = metadata['frequencies'].to(self.device)  # [batchsize, 25]

            # create mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,
                                                                         None]  # [batchsize, seq_len]
            batch_preds = self.model(batch, mask=mask.to(self.device),
                                    sequence_lengths=sequence_lengths)  # [batchsize, 2]
            batch_preds = batch_preds.tolist()
            label_org = label_org.tolist()
            preds.extend(batch_preds)
            labels.extend(label_org)
            id1 = metadata['id']
            id2 = metadata['d_id']
            t_id.extend(id1)
            d_id.extend(id2)

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
            bk = alphas -1 / np.sum(alphas, axis=-1)
            probs = alphas / np.sum(alphas, axis=-1).reshape(num_tasks, 1)

            probs = np.squeeze(probs)
            if probs[0] >= probs[1]:
                pred = 0
            else:
                pred = 1

            p.append(np.stack([pred,labels[i]]))
            # p.append(probs)
            prob_list.append(probs[1])

            conf = num_classes / np.sum(alphas, axis=-1)
            c.append(conf)
            ev.append(evidence)
            bk_list.append(bk)
            # TODO: std not implemented here
            var.append(conf)
        # mark = cal_top_hit_ratio(p,c)
        # return t_id,d_id,p, c, var,prob_list,ev,bk_list,mark
        return t_id, d_id, p, c, var, prob_list, ev, bk_list


    def evaluation(self, eval_dataset, filename: str = '', lookup_dataset: Dataset = None):

        self.model.eval()
        with torch.no_grad():
            t_id,d_id,p, c, var,prob_list,ev,bk_list = self.predict_val(eval_dataset)
        # print(p,c,var)
        val_results = np.squeeze(p)
        val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
        val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
        val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
        val_auc = roc_auc_score(val_results[:, 1], prob_list)
        val_recall = recall_score(val_results[:, 1], val_results[:, 0])
        val_pre = precision_score(val_results[:, 1], val_results[:, 0])
        fpr, tpr, tresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
        precision, recall, _thresholds = precision_recall_curve(val_results[:, 1], prob_list)
        val_prauc = auc(recall, precision)
        print(val_acc, val_recall, val_pre, val_mcc, val_f1, val_auc, val_prauc)
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % val_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
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
        result_csv.to_csv(os.path.join(self.writer.log_dir,'result.csv'))
        matrixs = [val_acc, val_recall, val_pre, val_mcc, val_f1, val_auc, val_prauc]
        with open('matrix_result.csv', 'a') as f:
            f.write('\t'.join(map(str, matrixs)) + '\n')


    def save_checkpoint(self, epoch: int):
        """
        Saves checkpoint of model in the logdir of the summarywriter/ in the used rundir
        Args:
            epoch: current epoch from which the run will be continued if it is loaded

        Returns:

        """
        run_dir = self.writer.log_dir
        torch.save(self.model, os.path.join(run_dir, 'checkpoint.pth'))


