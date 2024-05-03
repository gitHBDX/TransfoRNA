import logging
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import skorch
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from ..utils.file import save

logger = logging.getLogger(__name__)

def load_pkl(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def infere_additional_test_data(net,data):
    '''
    The premirna task has an additional dataset containing premirna from different species
    This function computes the accuracy score on this additional test set
    All samples in the additional test data are precurosr mirnas
    '''
    for dataset_idx in range(len(data)):
        predictions = net.predict(data[dataset_idx])
        correct = sum(torch.max(predictions,1).indices)
        total = len(torch.max(predictions,1).indices)
        logger.info(f'The prediction on the {dataset_idx} dataset is {correct} out of {total}')

def get_rna_seqs(seq, model_config):
    rna_seqs = []
    if model_config.tokenizer == "no_overlap":
        for _, row in seq.iterrows():
            rna_seqs.append("".join(x for x in row if x != "pad"))
    else:
        rna_seqs_overlap = []
        for _, row in seq.iterrows():
            # remove the paddings
            rna_seqs_overlap.append([x for x in row if x != "pad"])
            # join the beg of each char in rna_seqs_overlap
            rna_seqs.append("".join(x[0] for x in rna_seqs_overlap[-1]))
            # append the last token w/o its first char
            rna_seqs[-1] = "".join(rna_seqs[-1] + rna_seqs_overlap[-1][-1][1:])

    return rna_seqs

def save_embedds(net,path:str,rna_seq,split:str='train',labels:pd.DataFrame=None,model_config = None,logits=None):
    #reconstruct seqs
    # join sequence and remove pads
    rna_seqs = get_rna_seqs(rna_seq, model_config)

     # create pandas dataframe  of sequences
    iterables = [["RNA Sequences"], np.arange(1, dtype=int)]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    rna_seqs_df = pd.DataFrame(columns=index, data=np.vstack(rna_seqs))

    data=np.vstack(net.gene_embedds)
     # create pandas dataframe  for token ids of sequences
    iterables = [["RNA Embedds"], np.arange((data.shape[1]), dtype=int)]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    gene_embedd_df = pd.DataFrame(columns=index, data=data)

    if 'baseline' not in model_config.model_input:
        data = np.vstack(net.second_input_embedds)
        iterables = [["SI Embedds"], np.arange(data.shape[1], dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        exp_embedd_df = pd.DataFrame(columns=index, data=data)
    else:
        exp_embedd_df = []

    iterables = [["Labels"], np.arange(1, dtype=int)]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    labels_df = pd.DataFrame(columns=index, data=labels.values)

    if logits:
        iterables = [["Logits"], model_config.class_mappings]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        logits_df = pd.DataFrame(columns=index, data=np.array(logits))

        final_csv = rna_seqs_df.join(gene_embedd_df).join(exp_embedd_df).join(labels_df).join(logits_df)
    else:
        final_csv = rna_seqs_df.join(gene_embedd_df).join(exp_embedd_df).join(labels_df)

    save(data=final_csv,path =f'{path}{split}_embedds')
    

def infer_from_model(net,split_data:torch.Tensor):
    batch_size = 100
    predicted_labels_str = []
    soft = nn.Softmax()
    logits = []
    attn_scores_first_list = []
    attn_scores_second_list = []
    #this dict will be used to convert between neumeric predictions and string labels
    labels_mapping_dict = net.labels_mapping_dict
    #switch labels and str_labels
    labels_mapping_dict = {y:x for x,y in labels_mapping_dict.items()}
    for idx,batch in enumerate(torch.split(split_data, batch_size)):
        predictions = net.predict(batch)
        attn_scores_first,attn_scores_second = net.get_attention_scores(batch)
        predictions = predictions[:,:-1]

        max_ids_tensor = torch.max(predictions,1).indices
        if max_ids_tensor.is_cuda:
            max_ids_tensor = max_ids_tensor.cpu().numpy()
        predicted_labels_str.extend([labels_mapping_dict[x] for x in max_ids_tensor.tolist()])

        logits.extend(soft(predictions).detach().cpu().numpy())
        
        attn_scores_first_list.extend(attn_scores_first)
        if attn_scores_second is not None:
            attn_scores_second_list.extend(attn_scores_second)

    return predicted_labels_str,logits,attn_scores_first_list,attn_scores_second_list

def get_split_score(net,split_data:torch.Tensor,split_labels:torch.Tensor,split:str,scoring_function:Dict,task:str=None,log_split_str_labels:bool=False,mirna_flag:bool = None):
    split_acc = []
    batch_size = 100
    predicted_labels_str = []
    true_labels_str = []
    #this dict will be used to convert between neumeric predictions and string labels
    labels_mapping_dict = net.labels_mapping_dict
    #switch labels and str_labels
    labels_mapping_dict = {y:x for x,y in labels_mapping_dict.items()}
    for idx,batch in enumerate(torch.split(split_data, batch_size)):
        predictions = net.predict(batch)
        if split_labels is not None:
            true_labels = torch.split(split_labels,batch_size)[idx]
            if mirna_flag is not None:
                batch_score = scoring_function(true_labels.numpy(), predictions,task=task,mirna_flag=mirna_flag)
                batch_score /= sum(true_labels.numpy().squeeze() == mirna_flag)
            else:
                batch_score = scoring_function(true_labels.numpy(), predictions,task=task)
            split_acc.append(batch_score)

        if log_split_str_labels:
            #save true labels
            if split_labels is not None:
                true_labels_str.extend([labels_mapping_dict[x[0]] for x in true_labels.numpy().tolist()])
        predicted_labels_str.extend([labels_mapping_dict[x] for x in torch.max(predictions[:,:-1],1).indices.cpu().numpy().tolist()])
    
    if log_split_str_labels:
        #save all true and predicted labels to compute metrics on
        if split_labels is not None:
            with open(f"true_labels_{split}.pkl", "wb") as fp:  
                pickle.dump(true_labels_str, fp)

        with open(f"predicted_labels_{split}.pkl", "wb") as fp: 
            pickle.dump(predicted_labels_str, fp)

    
    if split_labels is not None:
        split_score = sum(split_acc)/len(split_acc)
        if mirna_flag is not None:
            logger.info(f"{split} accuracy score is {split_score} for mirna: {mirna_flag}")
    else:
        #only for inference
        split_score = None
            
        logger.info(f"{split} accuracy score is {split_score}")
    
    return split_score,predicted_labels_str

def generate_embedding(net, path:str,accuracy_sore,data, data_seq,labels,labels_numeric,split,model_config=None,train_config=None,log_embedds:bool=False):

    predictions_per_split = []   
    accuracy = []
    logits = []
    weights_per_batch = []
    data = torch.cat((data.T,labels_numeric.unsqueeze(1).T)).T
    for batch in torch.split(data, train_config.batch_size):
        weights_per_batch.append(batch.shape[0])
        predictions = net.predict(batch[:,:-1])
        soft = nn.Softmax(dim=1)
        logits.extend(list(soft(predictions[:,:-1]).detach().cpu().tolist()))

        accuracy.append(accuracy_sore(batch[:,-1], predictions))
        
        #drop sample weights
        predictions = predictions[:,:-1]

        predictions = torch.argmax(predictions,axis=1)
        predictions_per_split.extend(predictions.tolist())

    if split == 'test':
        matrix = confusion_matrix(labels_numeric.tolist(), predictions_per_split)
        #get the worst predicted classes
        worst_predicted_classes = np.argsort(matrix.diagonal())[:40]
        best_predicted_classes = np.argsort(matrix.diagonal())[-40:]
        #first get the mapping dict from labels_numeric tensor and labels containing string labels
        mapping_dict = {}
        for idx,label in enumerate(labels_numeric.tolist()):
            mapping_dict[label] = labels.values[idx][0]
        #convert worst_predicted_classes to string labels
        worst_predicted_classes = [mapping_dict[x] for x in worst_predicted_classes]
        #save worst predicted classes as csv
        pd.DataFrame(worst_predicted_classes).to_csv(f"{path}worst_predicted_classes.csv")

        #check how many files in path start with confusion_matrix
        num_confusion_matrix = len([name for name in os.listdir(path) if name.startswith("confusion_matrix")])
        #save confusion matrix
        cf = pd.DataFrame(matrix)
        #rename cf columns to be the labels by first ordering the mapping dict by the keys
        cf.columns = [mapping_dict[x] for x in sorted(mapping_dict.keys())]
        cf.index = cf.columns
        cf.to_csv(f"{path}confusion_matrix_{num_confusion_matrix}.csv")


    score_avg = 0
    if split in ['train','valid','test']:
        score_avg = np.average(accuracy,weights = weights_per_batch)
        logger.info(f"total accuracy score on {split} is {np.round(score_avg,4)}")

    
    if log_embedds:
        logger.debug(f"logging embedds for {split} set")
        save_embedds(net,path,data_seq,split,labels,model_config,logits)

    return score_avg



def compute_score_tcga(
    net, all_data, path,cfg:Dict
):
    task = cfg['task']
    net.load_params(f_params=f'{path}/ckpt/model_params_{task}.pt')
    net.save_embedding = True

    #create path for embedds and confusion matrix
    embedds_path = path+"/embedds/"
    if not os.path.exists(embedds_path):
        os.mkdir(embedds_path)
    
    #get scoring function
    for cb in net.callbacks:
        if type(cb) == skorch.callbacks.scoring.BatchScoring:
            scoring_function = cb.scoring._score_func
            break

    splits = ['train','valid','test','ood','no_annotation','artificial']

    test_score = 0
    #log all splits
    for split in splits:
        # reset tensors
        net.gene_embedds = []
        net.second_input_embedds = []
        try:
            score = generate_embedding(net,embedds_path,scoring_function,all_data[f"{split}_data"],all_data[f"{split}_rna_seq"],\
                                all_data[f"{split}_labels"],all_data[f"{split}_labels_numeric"],f'{split}',\
                                    cfg['model_config'],cfg['train_config'],cfg['log_embedds'])
            if split == 'test':
                test_score = score
        except:
            trained_on = cfg['trained_on']
            logger.info(f'Skipping {split} split, as split does not exist for models trained on {trained_on}!')
            
        
            
    return test_score
        




def compute_score_benchmark(
    net, path,all_data,scoring_function:Dict, cfg:Dict
):
    task = cfg['task']
    net.load_params(f_params=f'{path}/ckpt/model_params_{task}.pt')
    net.save_embedding = True
    # reset tensors
    net.gene_embedds = []
    net.second_input_embedds = []

    if task == 'premirna':
        get_split_score(net,all_data["train_data"],all_data["train_labels_numeric"],'train',scoring_function,task,mirna_flag = 0)
        get_split_score(net,all_data["train_data"],all_data["train_labels_numeric"],'train',scoring_function,task,mirna_flag = 1)
    else:
        get_split_score(net,all_data["train_data"],all_data["train_labels_numeric"],'train',scoring_function,task)

    embedds_path = path+"/embedds/"
    if not os.path.exists(embedds_path):
        os.mkdir(embedds_path)
    if cfg['log_embedds']:
        torch.save(torch.vstack(net.gene_embedds), embedds_path+"train_gene_embedds.pt")
        torch.save(torch.vstack(net.second_input_embedds), embedds_path+"train_gene_exp_embedds.pt")
        all_data["train_rna_seq"].to_pickle(embedds_path+"train_rna_seq.pkl")

    # reset tensors
    net.gene_embedds = []
    net.second_input_embedds = []
    if task == 'premirna':
        test_score_0,_ = get_split_score(net,all_data["test_data"],all_data["test_labels_numeric"],'test',scoring_function,task,mirna_flag = 0)
        test_score_1,_ = get_split_score(net,all_data["test_data"],all_data["test_labels_numeric"],'test',scoring_function,task,mirna_flag = 1)
        test_score = (test_score_0+test_score_1)/2
    else:
        test_score,_ = get_split_score(net,all_data["test_data"],all_data["test_labels_numeric"],'test',scoring_function,task)

    if cfg['log_embedds']:
        torch.save(torch.vstack(net.gene_embedds), embedds_path+"test_gene_embedds.pt")
        torch.save(torch.vstack(net.second_input_embedds), embedds_path+"test_gene_exp_embedds.pt")
        all_data["test_rna_seq"].to_pickle(embedds_path+"test_rna_seq.pkl")
    return test_score



def infer_testset(net,cfg,all_data,accuracy_score):
    if cfg["task"] == 'premirna':
            split_score,predicted_labels_str = get_split_score(net,all_data["test_data"],all_data["test_labels"],'test',accuracy_score,cfg["task"],mirna_flag = 0)
            split_score,predicted_labels_str = get_split_score(net,all_data["test_data"],all_data["test_labels"],'test',accuracy_score,cfg["task"],mirna_flag = 1)
    else:
        split_score,predicted_labels_str = get_split_score(net,all_data["test_data"],all_data["test_labels_numeric"],'test',\
        accuracy_score,cfg["task"],log_split_str_labels = True)
        #only for premirna
    if "additional_testset" in all_data:
        infere_additional_test_data(net,all_data["additional_testset"])