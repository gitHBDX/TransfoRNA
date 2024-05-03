import os

import numpy as np
import skorch
import torch
from sklearn.metrics import confusion_matrix, make_scorer
from skorch.callbacks import BatchScoring
from skorch.callbacks.scoring import ScoringBase, _cache_net_forward_iter
from skorch.callbacks.training import Checkpoint

from .LRCallback import LearningRateDecayCallback

writer = None

def accuracy_score(y_true, y_pred: torch.tensor,task:str=None,mirna_flag:bool = False):
    #sample 
    
    # premirna
    if task == "premirna":
        y_pred = y_pred[:,:-1]
        miRNA_idx = np.where(y_true.squeeze()==mirna_flag)
        correct = torch.max(y_pred,1).indices.cpu().numpy()[miRNA_idx] == mirna_flag
        return sum(correct) 

    # sncrna
    if task == "sncrna":
        y_pred = y_pred[:,:-1]
        # correct is of [samples], where each entry is true if it was found in top k
        correct = torch.max(y_pred,1).indices.cpu().numpy() == y_true.squeeze()

        return sum(correct) / y_pred.shape[0]


def accuracy_score_tcga(y_true, y_pred):
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.clone().detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.clone().detach().cpu().numpy()
    
    #y pred contains logits | samples weights
    sample_weight = y_pred[:,-1]
    y_pred = np.argmax(y_pred[:,:-1],axis=1)

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score

def score_callbacks(cfg):

    acc_scorer = make_scorer(accuracy_score,task=cfg["task"])
    if cfg['task'] == 'tcga':
        acc_scorer = make_scorer(accuracy_score_tcga)


    if cfg["task"] == "premirna":
        acc_scorer_mirna = make_scorer(accuracy_score,task=cfg["task"],mirna_flag = True)
        
        val_score_callback_mirna = BatchScoringPremirna( mirna_flag=True,
            scoring = acc_scorer_mirna, lower_is_better=False, name="val_acc_mirna")

        train_score_callback_mirna = BatchScoringPremirna(mirna_flag=True,
            scoring = acc_scorer_mirna, on_train=True, lower_is_better=False, name="train_acc_mirna")
        
        val_score_callback = BatchScoringPremirna(mirna_flag=False,
            scoring = acc_scorer, lower_is_better=False, name="val_acc")

        train_score_callback = BatchScoringPremirna(mirna_flag=False,
            scoring = acc_scorer, on_train=True, lower_is_better=False, name="train_acc")


        scoring_callbacks = [
                train_score_callback,
                train_score_callback_mirna
                            ]
        if cfg["train_split"]:
            scoring_callbacks.extend([val_score_callback_mirna,val_score_callback])
    
    if cfg["task"] in ["sncrna", "tcga"]:

        val_score_callback = BatchScoring(acc_scorer, lower_is_better=False, name="val_acc")
        train_score_callback = BatchScoring(
            acc_scorer, on_train=True, lower_is_better=False, name="train_acc"
        )
        scoring_callbacks = [train_score_callback]

        #tcga dataset has a predifined valid split, so train_split is false, but still valid metric is required
        #TODO: remove predifined valid from tcga from prepare_data_tcga
        if cfg["train_split"] or cfg['task'] == 'tcga':
            scoring_callbacks.append(val_score_callback)

    return scoring_callbacks

def get_callbacks(path,cfg):

    callback_list = [("lrcallback", LearningRateDecayCallback)]
    if cfg['tensorboard'] == True:
        from .tbWriter import writer
        callback_list.append(MetricsVizualization)

    if (cfg["train_split"] or cfg['task'] == 'tcga') and cfg['inference'] == False:
        monitor = "val_acc_best"
        if cfg['trained_on'] == 'full':
            monitor = 'train_acc_best'
        ckpt_path = path+"/ckpt/"
        try:
            os.mkdir(ckpt_path)
        except:
            pass
        model_name = f'model_params_{cfg["task"]}.pt'
        callback_list.append(Checkpoint(monitor=monitor, dirname=ckpt_path,f_params=model_name))

    scoring_callbacks = score_callbacks(cfg)
    #TODO: For some reason scoring callbaks have to be inserted before checpoint and metrics viz callbacks
    #otherwise NeuralNet notify function throws an exception
    callback_list[1:1] = scoring_callbacks

    return callback_list


class MetricsVizualization(skorch.callbacks.Callback):
    def __init__(self, batch_idx=0) -> None:
        super().__init__()
        self.batch_idx = batch_idx

    # TODO: Change to display metrics at epoch ends
    def on_batch_end(self, net, training, **kwargs):
        # validation batch
        if not training:
            # log val accuracy. accessing net.history:[ epoch ,batches, last batch,column in batch]
            writer.add_scalar(
                "Accuracy/val_acc",
                net.history[-1, "batches", -1, "val_acc"],
                self.batch_idx,
            )
            # log val loss
            writer.add_scalar(
                "Loss/val_loss",
                net.history[-1, "batches", -1, "valid_loss"],
                self.batch_idx,
            )
            # update batch idx after validation on batch is computed
        # train batch
        else:
            # log lr
            writer.add_scalar("Metrics/lr", net.lr, self.batch_idx)
            # log train accuracy
            writer.add_scalar(
                "Accuracy/train_acc",
                net.history[-1, "batches", -1, "train_acc"],
                self.batch_idx,
            )
            # log train loss
            writer.add_scalar(
                "Loss/train_loss",
                net.history[-1, "batches", -1, "train_loss"],
                self.batch_idx,
            )
            self.batch_idx += 1

class BatchScoringPremirna(ScoringBase):
    def __init__(self,mirna_flag:bool = False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self.total_num_samples = total_num_samples
        self.total_num_samples = 0
        self.mirna_flag = mirna_flag
        self.first_batch_flag = True
    def on_batch_end(self, net, X, y, training, **kwargs):
        if training != self.on_train:
            return

        y_preds = [kwargs['y_pred']]
        #only for the first batch: get no. of samples belonging to same class samples
        if self.first_batch_flag:
            self.total_num_samples += sum(kwargs["batch"][1] == self.mirna_flag).detach().cpu().numpy()[0]

        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            y = None if y is None else self.target_extractor(y)
            try:
                score = self._scoring(cached_net, X, y)
                cached_net.history.record_batch(self.name_, score)
            except KeyError:
                pass
    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name_]]))
        #score_avg = np.average(scores, weights=weights)
        score_avg = sum(scores)/self.total_num_samples
        return score_avg

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        self.first_batch_flag = False
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', bool(is_best))
