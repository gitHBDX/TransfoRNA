
import math
import os
import random
from contextlib import redirect_stdout
from pathlib import Path
from random import randint
from typing import Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from Levenshtein import distance
from omegaconf import DictConfig, OmegaConf
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from ..callbacks.metrics import get_callbacks
from ..dataset.seq_tokenizer import SeqTokenizer
from ..score.score import infer_from_model
from ..utils.file import save
from ..utils.tcga_post_analysis_utils import Results_Handler
from .energy import *
from .file import load


def get_fused_seqs(seqs,num_sequences:int=1,max_len:int=30):
    '''
    fuse num_sequences sequences from seqs
    '''
    fused_seqs = []
    while len(fused_seqs) < num_sequences:
        #get two random sequences
        seq1 = random.choice(seqs)[:max_len]
        seq2 = random.choice(seqs)[:max_len]
        
        #select indeex to tuncate seq1 at between 60 to 70% of its length
        idx = random.randint(math.floor(len(seq1)*0.3),math.floor(len(seq1)*0.7))
        len_to_be_added_from_seq2 = len(seq1)-idx
        #truncate seq1 at idx
        seq1 = seq1[:idx]
        #get the rest from the beg of seq2
        seq2 = seq2[:len_to_be_added_from_seq2]
        #fuse seq1 and seq2
        fused_seq = seq1+seq2

        if fused_seq not in fused_seqs and fused_seq not in seqs:
            fused_seqs.append(fused_seq)

    return fused_seqs

def get_lev_dist(seqs_a_list:List[str],seqs_b_list:List[str]):
    '''
    compute levenstein distance between two lists of sequences and normalize by the length of the longest sequence
    The lev distance is computed between seqs_a_list[i] and seqs_b_list[i]
    '''
    lev_dist = []
    for i in range(len(seqs_a_list)):
        dist = distance(seqs_a_list[i],seqs_b_list[i])
        #normalize
        dist = dist/max(len(seqs_a_list[i]),len(seqs_b_list[i]))
        lev_dist.append(dist)
    return lev_dist

def get_closest_neighbors(results:Results_Handler,query_embedds:np.ndarray,num_neighbors:int=1):
    '''
    get the closest neighbors to the query embedds using the knn model in results
    The closest neighbors are to be found in the training set
    '''
    #norm infer embedds
    query_embedds = query_embedds/np.linalg.norm(query_embedds,axis=1)[:,None]
    #get top 1 seqs
    distances, indices = results.knn_model.kneighbors(query_embedds)
    distances = distances[:,:num_neighbors].flatten()
    #flatten distances

    indices = indices[:,:num_neighbors]

    top_n_seqs = np.array(results.knn_seqs)[indices][:,:num_neighbors]
    top_n_seqs = [seq[0] for sublist in top_n_seqs for seq in sublist]
    top_n_labels = np.array(results.knn_labels)[indices][:,:num_neighbors]
    top_n_labels = [label[0] for sublist in top_n_labels for label in sublist]
    
    return top_n_seqs,top_n_labels,distances

def get_closest_ngbr_per_split(results:Results_Handler,split:str,num_neighbors:int=1):
    '''
    compute levenstein distance between the sequences in split and their closest neighbors in the training set
    '''
    split_df = results.splits_df_dict[f'{split}_df']
    #log
    print(f'number of sequences in {split} is {split_df.shape[0]}')
    #accomodate for multi-index df or single index
    try:
        split_seqs = split_df[results.seq_col].values[:,0]
    except:
        split_seqs = split_df[results.seq_col].values
    try:
        split_labels = split_df[results.label_col].values[:,0]
    except:
        split_labels = None
    #get embedds
    embedds = split_df[results.embedds_cols].values
    
    top_n_seqs,top_n_labels,distances = get_closest_neighbors(results,embedds,num_neighbors)
    #get levenstein distance
    #for each split_seqs duplicate it num_neighbors times
    split_seqs = [seq for seq in split_seqs for _ in range(num_neighbors)]
    lev_dist = get_lev_dist(split_seqs,top_n_seqs)
    return split_seqs,split_labels,top_n_seqs,top_n_labels,distances,lev_dist

def update_config_with_inference_params(config:DictConfig,mc_or_sc:str='sc',trained_on:str = 'id',path_to_models:str = 'models/tcga/') -> DictConfig:
    inference_config = config.copy()
    model = config['model_name']
    model = "-".join([word.capitalize() for word in model.split("-")])
    transforna_folder = "TransfoRNA_ID"
    if trained_on == "full":
        transforna_folder = "TransfoRNA_FULL"
    if mc_or_sc == 'sc':
        target = 'sub_class'
    else:
        target = 'major_class'

    inference_config['inference_settings']["model_path"] = f'{path_to_models}{transforna_folder}/{target}/{model}/ckpt/model_params_tcga.pt'
    inference_config["inference"] = True
    inference_config["log_logits"] = False


    inference_config = DictConfig(inference_config)
    #train and model config should be fetched from teh inference model
    train_cfg_path = get_hp_setting(inference_config, "train_config")
    model_cfg_path = get_hp_setting(inference_config, "model_config")
    train_config = instantiate(train_cfg_path)
    model_config = instantiate(model_cfg_path)
    # prepare configs as structured dicts
    train_config = OmegaConf.structured(train_config)
    model_config = OmegaConf.structured(model_config)
    # update model config with the name of the model
    model_config["model_input"] = inference_config["model_name"]
    inference_config = OmegaConf.merge({"train_config": train_config, "model_config": model_config}, inference_config)
    return inference_config

def predict_transforna_na(config:DictConfig = None) -> Tuple:
    inference_config = update_config_with_inference_params(config)
    
    #path should be infer_cfg["model_path"] - 2 level + embedds
    path = '/'.join(inference_config['inference_settings']["model_path"].split('/')[:-2])+'/embedds'
    #read threshold
    results:Results_Handler = Results_Handler(path=path,splits=['train','na'])
    results.get_knn_model()
    threshold = load(results.analysis_path+"/novelty_model_coef")["Threshold"]
    sequences = results.splits_df_dict['na_df'][results.seq_col].values[:,0]
    with redirect_stdout(None):
        root_dir = Path(__file__).parents[3].absolute()
        inference_config, net = get_model(inference_config, root_dir)
        infer_pd = pd.Series(sequences, name="Sequences").to_frame()
        print(f'predicting sub classes for the NA set by the ID models')
        predicted_labels, logits,gene_embedds_df, attn_scores_pd,all_data, max_len, net = infer_from_pd(inference_config, net, infer_pd, SeqTokenizer)


    prepare_inference_results_tcga(inference_config, predicted_labels, logits, all_data, max_len)
    infer_pd = all_data["infere_rna_seq"]

    
    #compute lev distance for embedds and 
    print('computing levenstein distance for the NA set by the ID models')
    _,_,_,_,_,lev_dist = get_closest_ngbr_per_split(results,'na')
    
    print(f'num of hico based on entropy novelty prediction is {sum(infer_pd["Is Familiar?"])}')
    infer_pd['Is Familiar?'] = [True if lv<threshold else False for lv in lev_dist]
    infer_pd['Threshold'] = threshold
    print(f'num of new hico based on levenstein distance is {np.sum(infer_pd["Is Familiar?"])}')
    return infer_pd.rename_axis("Sequence").reset_index()
    
def set_seed_and_device(seed:int = 0,device_no:int=0):
    # set seed
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device_no)
    #CUDA_LAUNCH_BLOCKING=1 #for debugging

def sync_skorch_with_config(skorch_cfg: DictConfig,cfg:DictConfig):
    '''
    skorch config contains duplicate params to the train and model configs
    values of skorch config should be populated by those in the trian and 
    model config 
    '''

    #populate skorch params with params in train or model config if exists
    for key in skorch_cfg:
        if key in cfg["train_config"]:
            skorch_cfg[key] = cfg["train_config"][key]
        if key in cfg["model_config"]:
            skorch_cfg[key] = cfg["model_config"][key]

    return 

def instantiate_predictor(skorch_cfg: DictConfig,cfg:DictConfig,path: str=None):
    # convert config to omegaconf container
    predictor_config = OmegaConf.to_container(skorch_cfg)
    # Patch model device argument from the run config:
    if "device" in predictor_config:
        predictor_config["device"] = skorch_cfg["device"]
    for key, val in predictor_config.items():
        try:
            predictor_config[key] = _locate(val)
        except:
            continue
    #add callbacks to list of params    
    predictor_config["callbacks"] = get_callbacks(path,cfg)
    
    
    #save callbacks as instantiate changes the lrcallback from tuple to list,
    #then skorch's instantiate_callback throws an error
    callbacks_list = predictor_config["callbacks"]
    predictor_config["callbacks"] = "disable"

    #remove model from the cfg otherwise intantiate will throw an error as
    #models' scoring doesnt recieve input params
    predictor_config["module__main_config"] = \
        {key:cfg[key] for key in cfg if key not in ["model"]}
    #in case of tcga task, remove dataset at it its already instantiated        
    if 'dataset' in predictor_config['module__main_config']:
        del predictor_config['module__main_config']['dataset']

    #set train split to false in skorch model
    if not cfg['train_split']:
        predictor_config['train_split'] = False
    net = instantiate(predictor_config)
    #restore callback and instantiate it
    net.callbacks = callbacks_list
    net.task = cfg['task']
    net.initialize_callbacks()
    #prevents double initialization
    net.initialized_=True
    return net

def revert_seq_tokenization(sequences,configs):
    window = configs["model_config"].window
    if configs["model_config"].tokenizer != "overlap":
        print("Sequences are not reverse tokenized")
        return sequences
    
    #currently only overlap tokenizer can be reverted
    seqs_concat = []
    for seq in sequences.values:
        seqs_concat.append(''.join(seq[seq!='pad'])[::window]+seq[seq!='pad'][-1][window-1])
    
    return pd.DataFrame(seqs_concat,columns=["Sequences"])

def prepare_split(split_data_df,configs):
    '''
    This function returns tokens, token ids and labels for a given dataframes' split.
    It also moves tokens and labels to device
    '''

    model_input_cols = ['tokens_id','second_input','seqs_length']
    #token_ids
    split_data = torch.tensor(
        np.array(split_data_df[model_input_cols].values, dtype=float),
        dtype=torch.float,
    )
    split_weights = torch.tensor(compute_sample_weight('balanced',split_data_df['Labels']))
    split_data = torch.cat([split_data,split_weights[:,None]],dim=1)
    #tokens (chars)
    split_rna_seq = revert_seq_tokenization(split_data_df["tokens"],configs)

    #labels
    split_labels = torch.tensor(
        np.array(split_data_df["Labels"], dtype=int),
        dtype=torch.long,
    )
    return split_data, split_rna_seq, split_labels

def get_add_test_set(dataset_class,dataset_path):
    all_added_test_set = []
    #get paths of all files in mirbase and mirgene
    paths_mirbase = dataset_path+"mirbase/"
    files_mirbase = os.listdir(paths_mirbase)
    for f_idx,_ in enumerate(files_mirbase):
        files_mirbase[f_idx] = paths_mirbase+files_mirbase[f_idx]
    
    paths_mirgene = dataset_path + "mirgene/"
    files_mirgene = os.listdir(paths_mirgene)
    for f_idx,_ in enumerate(files_mirgene):
        files_mirgene[f_idx] = paths_mirgene+files_mirgene[f_idx]
    files = files_mirbase+files_mirgene
    for f in files:
        #tokenize test set
        test_pd = load(f)
        test_pd = test_pd.drop(columns='Unnamed: 0')
        test_pd["Sequences"] = test_pd["Sequences"].astype(object)
        test_pd["Secondary"] = test_pd["Secondary"].astype(object)
        #convert dataframe to anndata
        test_pd["Labels"] = 1

        dataset_class.seqs_dot_bracket_labels = test_pd
        dataset_class.limit_seqs_to_range()
        all_added_test_set.append(dataset_class.get_preprocessed_data_df())
    return all_added_test_set

def update_config_with_dataset_params_benchmark(train_data_df,configs):
    '''
    After tokenizing the dataset, some features in the config needs to be updated as they will be used 
    later by sub modules
    '''
    # set feedforward input dimension and vocab size
    #ss_tokens_id and tokens_id are the same
    configs["model_config"].second_input_token_len = train_data_df["second_input"].shape[1]
    configs["model_config"].tokens_len = train_data_df["tokens_id"].shape[1]
    #set batch per epoch (number of batches). This will be used later by both the criterion and the LR
    configs["train_config"].batch_per_epoch = train_data_df["tokens_id"].shape[0]/configs["train_config"].batch_size
    return 

def tokenize_set(dataset_class,test_ad,inference:bool=False):
    test_ad.var["Secondary"] = test_ad.var["Secondary"].astype(object)
    test_ad.var["Sequences"] = test_ad.var["Sequences"].astype(object)

    #reassign the sequences to test
    dataset_class.ad = test_ad
    #prevent sequences with len < min lenght from being deleted
    dataset_class.min_length = 0
    dataset_class.limit_seqs_to_range()
    return  dataset_class.get_preprocessed_data_df(inference)

def stratify(train_data,train_labels,valid_size):
    return train_test_split(train_data, train_labels,
                                                    stratify=train_labels, 
                                                    test_size=valid_size)

def convert_to_tensor(in_arr,convert_type,device):
    tensor_dtype = torch.long if convert_type == int else torch.float
    return torch.tensor(
        np.array(in_arr, dtype=convert_type),
        dtype=tensor_dtype,
    ).to(device=device)

        
def prepare_data_benchmark(dataset_class,test_ad, configs):
    """
    This function recieves anddata and prepares the anndata in a format suitable for training
    It also set default parameters in the config that cannot be known until preprocessing step
    is done.
    all_data_df is heirarchical pandas dataframe, so can be accessed  [AA,AT,..,AC ]
    """
    ###get tokenized train set
    train_data_df = dataset_class.get_preprocessed_data_df()
    
    ### update config with data specific params
    update_config_with_dataset_params_benchmark(train_data_df,configs)

    ###tokenize test set
    test_data_df = tokenize_set(dataset_class,test_ad)

    ### get tokens(on device), seqs and labels(on device)
    train_data, train_rna_seq, train_labels =  prepare_split(train_data_df,configs)
    test_data, test_rna_seq, test_labels =  prepare_split(test_data_df,configs)

    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(train_labels.flatten()),y=train_labels.flatten().numpy())

    
    #omegaconfig does not support float64 as datatype so conversion to str is done 
    # and reconversion is done in criterion
    configs['model_config'].class_weights = [str(x) for x in list(class_weights)]

    if configs["train_split"]:
        #stratify train to get valid
        train_data,valid_data,train_labels,valid_labels = stratify(train_data,train_labels,configs["valid_size"])
        valid_ds = Dataset(valid_data,valid_labels)
        valid_ds=predefined_split(valid_ds)
    else:
        valid_ds = None

    all_data= {"train_data":train_data, 
               "valid_ds":valid_ds,
               "test_data":test_data, 
               "train_rna_seq":train_rna_seq,
               "test_rna_seq":test_rna_seq,
               "train_labels_numeric":train_labels,
               "test_labels_numeric":test_labels}

    if configs["task"] == "premirna":
        generalization_test_set = get_add_test_set(dataset_class,\
            dataset_path=configs["train_config"].datset_path_additional_testset)
    

    #get all vocab from both test and train set
    configs["model_config"].vocab_size = len(dataset_class.seq_tokens_ids_dict.keys())
    configs["model_config"].second_input_vocab_size = len(dataset_class.second_input_tokens_ids_dict.keys())
    configs["model_config"].tokens_mapping_dict = dataset_class.seq_tokens_ids_dict

    
    if configs["task"] == "premirna":
        generalization_test_data = []
        for test_df in generalization_test_set:
            #no need to read the labels as they are all one
            test_data_extra, _, _ =  prepare_split(test_df,configs)
            generalization_test_data.append(test_data_extra)
        all_data["additional_testset"] = generalization_test_data

    #get inference dataset
    # if do inference and inference datasert path exists
    get_inference_data(configs,dataset_class,all_data)

    return all_data

def convert_pd_to_ad(data_pd,cfg):
    #convert infer_data from pd to ad
    infer_ad = anndata.AnnData(X= np.zeros((cfg["model_config"]["ff_input_dim"],len(data_pd.index))))
    infer_ad.var["Sequences"]= data_pd["Sequences"].values
    infer_ad.var["Secondary"] = data_pd["Secondary"].values
    infer_ad.var["Labels"] = data_pd["Labels"].values
    return infer_ad

def get_inference_data(configs,dataset_class,all_data):

    if configs["inference"]==True and configs["inference_settings"]["sequences_path"] is not None:
        inference_file = configs["inference_settings"]["sequences_path"]
        inference_path = Path(__file__).parent.parent.parent.absolute() / f"{inference_file}"

        infer_data = load(inference_path)
        #check if infer_data has secondary structure
        if "Secondary" not in infer_data:
            infer_data['Secondary'] = dataset_class.get_secondary_structure(infer_data["Sequences"])
        if "Labels" not in infer_data:
            infer_data["Labels"] = [0]*len(infer_data["Sequences"].values)
        
        dataset_class.seqs_dot_bracket_labels = infer_data


        dataset_class.min_length = 0
        dataset_class.limit_seqs_to_range()
        infere_data_df = dataset_class.get_preprocessed_data_df(inference=True)
        infere_data,infere_rna_seq,_ = prepare_split(infere_data_df,configs)

        all_data["infere_data"] = infere_data
        all_data["infere_rna_seq"] = infere_rna_seq

def remove_fewer_samples(min_num_samples,selected_classes_df):
    counts = selected_classes_df['Labels'].value_counts()
    fewer_class_ids = counts[counts < min_num_samples].index
    fewer_class_labels = [i[0]  for i in fewer_class_ids]
    fewer_samples_per_class_df = selected_classes_df.loc[selected_classes_df['Labels'].isin(fewer_class_labels).values, :]
    fewer_ids = selected_classes_df.index.isin(fewer_samples_per_class_df.index)
    selected_classes_df = selected_classes_df[~fewer_ids]
    return fewer_samples_per_class_df,selected_classes_df


def split_tcga_data_keep_all_sc(selected_classes_df,configs):
    #remove artificial_affix
    ood_df = selected_classes_df.loc[selected_classes_df['Labels'][0].isin(['random','recombined','artificial_affix'])]
    art_affix_ids = selected_classes_df.index.isin(ood_df.index)
    selected_classes_df = selected_classes_df[~art_affix_ids]
    selected_classes_df = selected_classes_df.reset_index(drop=True)

    #remove no annotations
    no_annotaton_df = selected_classes_df.loc[selected_classes_df['Labels'].isnull().values]

    n_a_ids = selected_classes_df.index.isin(no_annotaton_df.index)
    selected_classes_df = selected_classes_df[~n_a_ids]
    #reset ids
    selected_classes_df = selected_classes_df.reset_index(drop=True)
    no_annotaton_df = no_annotaton_df.reset_index(drop=True)
    #get quantity of each class and append it as a column
    selected_classes_df["Quantity",'0'] = selected_classes_df["Labels"].groupby([0])[0].transform("count")
    frequent_samples_df = selected_classes_df[selected_classes_df["Quantity",'0'] >= 8].reset_index(drop=True)
    fewer_samples_df = selected_classes_df[selected_classes_df["Quantity",'0'] < 8].reset_index(drop=True)
    unique_fewer_samples_df = fewer_samples_df.drop_duplicates(subset=[('Labels',0)], keep="last")
    unique_fewer_samples_df['Quantity','0'] -= 8
    unique_fewer_samples_df['Quantity','0'] = unique_fewer_samples_df['Quantity','0'].abs()
    repeated_fewer_samples_df = unique_fewer_samples_df.loc[unique_fewer_samples_df.index.repeat(unique_fewer_samples_df.Quantity['0'])]
    repeated_fewer_samples_df = repeated_fewer_samples_df.reset_index(drop=True)
    selected_classes_df = frequent_samples_df.append(repeated_fewer_samples_df).append(fewer_samples_df).reset_index(drop=True)

    
    train_df,valid_test_df = train_test_split(selected_classes_df,stratify=selected_classes_df["Labels"],train_size=0.7,random_state=configs.seed)
    
    valid_df,test_df =train_test_split(valid_test_df,stratify=valid_test_df["Labels"],train_size=0.5,random_state=configs.seed)
    
    train_labels = train_df['Labels'].values.tolist()
    train_labels = [i[0] for i in train_labels]
    valid_labels = valid_df['Labels'].values.tolist()
    valid_labels = [i[0] for i in valid_labels]

    ood_3_labels = list(set(train_labels).difference(set(valid_labels)))
    ood_3_df = train_df[train_df['Labels'][0].isin(ood_3_labels)]
    ood_3_ids = train_df.index.isin(ood_3_df.index)
    train_df = train_df[~ood_3_ids]



    #train_df is the selected_classes_df in case of full as no train/val/test split should take place
    #sample 1 sample per each label in valid_df and test_df
    valid_df = valid_df.groupby(('Labels',0)).apply(lambda x: x.iloc[np.random.randint(x.shape[0]),:] if len(x)>0 else None).reset_index(drop=True).dropna(subset=[('Labels',0)])
    test_df = test_df.groupby(('Labels',0)).apply(lambda x: x.iloc[np.random.randint(x.shape[0]),:] if len(x)>0 else None).reset_index(drop=True).dropna(subset=[('Labels',0)])
    splits_df_dict = {"train_df":selected_classes_df,"valid_df":valid_df,"test_df":test_df,"ood_df":ood_df,"no_annotaiton_df":no_annotaton_df}
    return splits_df_dict

def split_tcga_data(selected_classes_df,configs):
    #remove artificial_affix
    ood_0_df = selected_classes_df.loc[selected_classes_df['Labels'][0] == 'artificial_affix']
    art_affix_ids = selected_classes_df.index.isin(ood_0_df.index)
    selected_classes_df = selected_classes_df[~art_affix_ids]
    selected_classes_df = selected_classes_df.reset_index(drop=True)

    #remove no annotations
    no_annotaton_df = selected_classes_df.loc[selected_classes_df['Labels'].isnull().values]

    n_a_ids = selected_classes_df.index.isin(no_annotaton_df.index)
    selected_classes_df = selected_classes_df[~n_a_ids]
    #reset ids
    selected_classes_df = selected_classes_df.reset_index(drop=True)
    no_annotaton_df = no_annotaton_df.reset_index(drop=True)
    #remove classes that have only one sample
    min_num_samples = 2
    ood_1_df,selected_classes_df = remove_fewer_samples(min_num_samples,selected_classes_df)
    #reset index
    selected_classes_df = selected_classes_df.reset_index(drop=True)
    
    #split data
    train_df,valid_test_df = train_test_split(selected_classes_df,stratify=selected_classes_df["Labels"],train_size=0.8,random_state=configs.seed)
    
    #remove ones from valid test df
    min_num_samples = 2
    ood_2_df,valid_test_df = remove_fewer_samples(min_num_samples,valid_test_df)
    valid_df,test_df =train_test_split(valid_test_df,stratify=valid_test_df["Labels"],train_size=0.5,random_state=configs.seed)
    
    train_labels = train_df['Labels'].values.tolist()
    train_labels = [i[0] for i in train_labels]
    valid_labels = valid_df['Labels'].values.tolist()
    valid_labels = [i[0] for i in valid_labels]

    ood_3_labels = list(set(train_labels).difference(set(valid_labels)))
    ood_3_df = train_df[train_df['Labels'][0].isin(ood_3_labels)]
    ood_3_ids = train_df.index.isin(ood_3_df.index)
    train_df = train_df[~ood_3_ids]
    #create ood set but selecting classes in train but not in valid or test
    # PLUS appending the oo_1_df
    ood_df = ood_0_df.append(ood_1_df).append(ood_2_df).append(ood_3_df)


    splits_df_dict = {"train_df":train_df,"valid_df":valid_df,"test_df":test_df,"ood_df":ood_df,"no_annotaiton_df":no_annotaton_df}
    return splits_df_dict

def update_config_with_dataset_params_tcga(dataset_class,all_data_df,configs):
    configs["model_config"].ff_input_dim = all_data_df['second_input'].shape[1]
    configs["model_config"].vocab_size = len(dataset_class.seq_tokens_ids_dict.keys())
    configs["model_config"].second_input_vocab_size = len(dataset_class.second_input_tokens_ids_dict.keys())
    configs["model_config"].tokens_len = dataset_class.tokens_len
    configs["model_config"].second_input_token_len = dataset_class.tokens_len

    if configs["model_name"] == "seq-seq":
        configs["model_config"].tokens_len = math.ceil(dataset_class.tokens_len/2)
        configs["model_config"].second_input_token_len = math.ceil(dataset_class.tokens_len/2)
    
    '''
    The expression profile has a length of 481. 
    1 is ommitted, so 480 is the new expression profile length, this number is then divided by the num_embedd_hidden which in turn
      deciedes the length of the token embedding on which Self attention would be applied; 480/30 = 16.
      check forward function of RNATransformer
    '''
    if configs["model_name"] == "seq-exp":
        configs["model_config"]["num_embed_hidden"] = 30 

def append_sample_weights(splits_df_dict,splits_features_dict,device):

    train_weights = convert_to_tensor(compute_sample_weight('balanced',splits_df_dict['train_df']['Labels'][0]),convert_type=float,device=device)
    valid_weights = convert_to_tensor(compute_sample_weight('balanced',splits_df_dict['valid_df']['Labels'][0]),convert_type=float,device=device)
    test_weights =  convert_to_tensor(compute_sample_weight('balanced',splits_df_dict['test_df']['Labels'][0]),convert_type=float,device=device)
    ood_weights = convert_to_tensor(np.ones(splits_df_dict['ood_df'].shape[0]),convert_type=float,device=device)
    na_weights =  convert_to_tensor(np.ones(splits_df_dict['no_annotaiton_df'].shape[0]),convert_type=float,device=device)

    splits_features_dict['train_data'] = torch.cat([splits_features_dict['train_data'],train_weights[:,None]],dim=1)
    splits_features_dict['valid_data'] = torch.cat([splits_features_dict['valid_data'],valid_weights[:,None]],dim=1)
    splits_features_dict['test_data'] = torch.cat([splits_features_dict['test_data'],test_weights[:,None]],dim=1)
    splits_features_dict['ood_data'] = torch.cat([splits_features_dict['ood_data'],ood_weights[:,None]],dim=1)
    splits_features_dict['na_data'] = torch.cat([splits_features_dict['na_data'],na_weights[:,None]],dim=1)

    return

def get_features_per_split(splits_df_dict,device):
    model_input_cols = ['tokens_id','second_input','seqs_length']
    #get data and labels for each of the five splits
    train_data = convert_to_tensor(splits_df_dict["train_df"][model_input_cols].values,convert_type=float,device=device)
    valid_data = convert_to_tensor(splits_df_dict["valid_df"][model_input_cols].values,convert_type=float,device=device)
    test_data = convert_to_tensor(splits_df_dict["test_df"][model_input_cols].values,convert_type=float,device=device)
    ood_data = convert_to_tensor(splits_df_dict["ood_df"][model_input_cols].values,convert_type=float,device=device)
    na_data = convert_to_tensor(splits_df_dict["no_annotaiton_df"][model_input_cols].values,convert_type=float,device=device)
    
    return {"train_data":train_data,"valid_data":valid_data,"test_data":test_data,"ood_data":ood_data,"na_data":na_data}

def get_labels_per_split(splits_df_dict,configs,device):
    #obtain labels
    train_labels = splits_df_dict["train_df"]['Labels']
    valid_labels =splits_df_dict["valid_df"]['Labels']
    test_labels = splits_df_dict["test_df"]['Labels']
    ood_labels = splits_df_dict["ood_df"]['Labels']
    na_labels = splits_df_dict["no_annotaiton_df"]['Labels']

    
    #encode labels 
    enc = LabelEncoder()
    enc.fit(splits_df_dict["train_df"]['Labels'])
    #save mapping dict to config
    configs["model_config"].class_mappings = enc.classes_.tolist()
    train_labels_numeric = convert_to_tensor(enc.transform(train_labels), convert_type=int,device=device)
    valid_labels_numeric =convert_to_tensor(enc.transform(valid_labels), convert_type=int,device=device)
    test_labels_numeric =convert_to_tensor(enc.transform(test_labels), convert_type=int,device=device)
    ood_labels_numeric =convert_to_tensor(np.zeros((ood_labels.shape[0])), convert_type=int,device=device)
    na_labels_numeric =convert_to_tensor(np.zeros((na_labels.shape[0])), convert_type=int,device=device)
    
    #compute class weight
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(train_labels),y=train_labels[0].values)
    
    #omegaconfig does not support float64 as datatype so conversion to str is done 
    # and reconversion is done in criterion
    configs['model_config'].class_weights = [str(x) for x in list(class_weights)]


    return {"train_labels":train_labels,
            "valid_labels":valid_labels,
            "test_labels":test_labels,
            "ood_labels":ood_labels,
            "na_labels":na_labels,

            "train_labels_numeric":train_labels_numeric,
            "valid_labels_numeric":valid_labels_numeric,
            "test_labels_numeric":test_labels_numeric,
            "ood_labels_numeric":ood_labels_numeric,
            "na_labels_numeric":na_labels_numeric
    }

def get_seqs_per_split(splits_df_dict,configs):
    train_rna_seq = revert_seq_tokenization(splits_df_dict["train_df"]["tokens"],configs)
    valid_rna_seq = revert_seq_tokenization(splits_df_dict["valid_df"]["tokens"],configs)
    test_rna_seq = revert_seq_tokenization(splits_df_dict["test_df"]["tokens"],configs)
    ood_rna_seq = revert_seq_tokenization(splits_df_dict["ood_df"]["tokens"],configs)
    na_rna_seq = revert_seq_tokenization(splits_df_dict["no_annotaiton_df"]["tokens"],configs)

    return {"train_rna_seq":train_rna_seq,
            "valid_rna_seq":valid_rna_seq,
            "test_rna_seq":test_rna_seq,
            "ood_rna_seq":ood_rna_seq,
            "na_rna_seq":na_rna_seq}

def create_artificial_dataset(size):
    ntds = ['A','C','G','T']
    max_len = size["seq_len_dist"].index.max()

    samples_ntds = np.random.choice(ntds,(size["num_seqs"],max_len))
    artificial_seqs = np.array([''.join(samples_ntds[i,:]) for i in range(samples_ntds.shape[0])])

    possible_seq_lengths = size["seq_len_dist"].index.to_numpy()
    prob_per_length = size["seq_len_dist"].values/sum(size["seq_len_dist"].values)

    artificial_seqs_length = np.random.choice(possible_seq_lengths,size = size["num_seqs"],p = prob_per_length)

    #truncate seqs according to the seq_len distribution
    for seq_idx,seq in enumerate(artificial_seqs):
        artificial_seqs[seq_idx] = seq[0:artificial_seqs_length[seq_idx]]
   

    artificial_secondary = fold_sequences(artificial_seqs,temperature=37)[f'structure_37'].values

    artificial_df = pd.DataFrame({'Sequences':artificial_seqs,'Secondary':artificial_secondary,'Labels':0})
    return artificial_df

def prepare_artificial_data(dataset_class,size,device):
    #create artificial dataset based on uniform sampling of nucleotides
    artificial_df = create_artificial_dataset(size)
    dataset_class.seqs_dot_bracket_labels = artificial_df
    processed_artificial_df = dataset_class.get_preprocessed_data_df().sample(frac=1)
    model_input_cols = ['tokens_id','second_input','seqs_length']
    artificial_data = convert_to_tensor(processed_artificial_df[model_input_cols].values,convert_type=float,device=device)
    art_weights =  convert_to_tensor(np.ones(artificial_data.shape[0]),convert_type=float,device=device)
    artificial_data = torch.cat([artificial_data,art_weights[:,None]],dim=1)

    artificial_labels_numeric =convert_to_tensor(np.zeros((processed_artificial_df['Labels'].shape[0])), convert_type=int,device=device)
    artificial_labels = processed_artificial_df['Labels']
    artificial_rna_seq = artificial_df[["Sequences"]]
    return {"artificial_data":artificial_data,
            "artificial_labels":artificial_labels,
            "artificial_labels_numeric":artificial_labels_numeric,
            "artificial_rna_seq":artificial_rna_seq}

def append_na_to_train(all_data):
    all_data["train_data"] = torch.cat((all_data["train_data"],all_data["na_data"]))
    all_data["train_labels_numeric"] = torch.cat((all_data["train_labels_numeric"],all_data["na_labels_numeric"]))
    all_data["train_labels"] = all_data['train_labels'].append(all_data['na_labels']).reset_index(drop=True)
    all_data["train_rna_seq"] = all_data['train_rna_seq'].append(all_data['na_rna_seq'])

    entries_to_remove = ('na_data', 'na_labels','na_labels_numeric','na_rna_seq')
    for k in entries_to_remove:
        all_data.pop(k, None)
    return all_data

def prepare_data_tcga(dataset_class, configs):
    """
    This function recieves anddata and prepares the anndata in a format suitable for training
    It also set default parameters in the config that cannot be known until preprocessing step
    is done.
    all_data_df is heirarchical pandas dataframe, so can be accessed  [AA,AT,..,AC ]
    """
    device = configs["train_config"].device

    all_data_df = dataset_class.get_preprocessed_data_df().sample(frac=1)

    #split data 
    if configs['trained_on'] == 'full':
        splits_df_dict =  split_tcga_data_keep_all_sc(all_data_df,configs)
    elif configs['trained_on'] == 'id':
        splits_df_dict =  split_tcga_data(all_data_df,configs)
    
    train_val_counts = splits_df_dict['train_df'].Labels.value_counts()[splits_df_dict['train_df'].Labels.value_counts()>0]
    train_num_samples = sum(train_val_counts)
    num_scs = len(train_val_counts)
    #log
    print(f'Training with {num_scs} sub classes and {train_num_samples} samples')

    #get features, labels, and seqs per split
    splits_features_dict = get_features_per_split(splits_df_dict,device)
    append_sample_weights(splits_df_dict,splits_features_dict,device)
    splits_labels_dict = get_labels_per_split(splits_df_dict,configs,device)
    splits_seqs_dict = get_seqs_per_split(splits_df_dict,configs)

    #get desired artificial data split
    size_dict = {
                "seq_len_dist":dataset_class.seq_len_dist,
                "num_seqs":200}
    artificial_data_dict = prepare_artificial_data(dataset_class,size_dict,device)

    
    #prepare validation set for skorch
    valid_ds = Dataset(splits_features_dict["valid_data"],splits_labels_dict["valid_labels_numeric"])
    valid_ds = predefined_split(valid_ds)

    #combine all dicts
    all_data = splits_features_dict | splits_labels_dict | splits_seqs_dict | \
        {"valid_ds":valid_ds} | artificial_data_dict

    ###update configs
    update_config_with_dataset_params_tcga(dataset_class,all_data_df,configs)
    configs["model_config"].num_classes = len(all_data['train_labels'][0].unique())
    configs["train_config"].batch_per_epoch = int(all_data["train_data"].shape[0]\
        /configs["train_config"].batch_size)

    get_inference_data(configs,dataset_class,all_data)

    #save token dicts
    save(data = dataset_class.second_input_tokens_ids_dict,path = os.getcwd()+'/second_input_tokens_ids_dict')
    save(data = dataset_class.seq_tokens_ids_dict,path = os.getcwd()+'/seq_tokens_ids_dict')
    #save token dicts
    save(data = dataset_class.second_input_tokens_ids_dict,path = os.getcwd()+'/second_input_tokens_ids_dict')
    save(data = dataset_class.seq_tokens_ids_dict,path = os.getcwd()+'/seq_tokens_ids_dict')
    return all_data

def introduce_mismatches(seq, n_mismatches):
    seq = list(seq)
    for i in range(n_mismatches):
        rand_nt = randint(0,len(seq)-1)
        seq[rand_nt] = ['A','G','C','T'][randint(0,3)]
    return ''.join(seq)


def prepare_inference_results_benchmarck(net,cfg,predicted_labels,logits,all_data):
    iterables = [["Sequences"], np.arange(1, dtype=int)]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    rna_seqs_df = pd.DataFrame(columns=index, data=np.vstack(all_data["infere_rna_seq"]["Sequences"].values))

    iterables = [["Logits"], list(net.labels_mapping_dict.keys())]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    logits_df = pd.DataFrame(columns=index, data=np.array(logits))

    #add Labels,entropy to df
    all_data["infere_rna_seq"]["Labels",'0'] = predicted_labels
    all_data["infere_rna_seq"].set_index("Sequences",inplace=True)

    #log logits if required
    if cfg["log_logits"]:
        seq_logits_df = logits_df.join(rna_seqs_df).set_index(("Sequences",0))
        all_data["infere_rna_seq"] = all_data["infere_rna_seq"].join(seq_logits_df)
    else:
        all_data["infere_rna_seq"].columns = ['Labels']

    return 

def prepare_inference_results_tcga(cfg,predicted_labels,logits,all_data,max_len):

    logits_clf = load('/'.join(cfg["inference_settings"]["model_path"].split('/')[:-2])\
        +'/analysis/logits_model_coef.yaml')
    threshold = round(logits_clf['Threshold'],2)


    iterables = [["Sequences"], np.arange(1, dtype=int)]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    rna_seqs_df = pd.DataFrame(columns=index, data=np.vstack(all_data["infere_rna_seq"]["Sequences"].values))

    iterables = [["Logits"], cfg['model_config'].class_mappings]
    index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
    logits_df = pd.DataFrame(columns=index, data=np.array(logits))

    #add Labels,novelty to df
    all_data["infere_rna_seq"]["Net-Label"] = predicted_labels
    all_data["infere_rna_seq"]["Is Familiar?"] = entropy(logits,axis=1) <= threshold

    all_data["infere_rna_seq"].set_index("Sequences",inplace=True)

    #log logits if required
    if cfg["log_logits"]:
        seq_logits_df = logits_df.join(rna_seqs_df).set_index(("Sequences",0))
        all_data["infere_rna_seq"] = all_data["infere_rna_seq"].join(seq_logits_df)
       
    all_data["infere_rna_seq"].index.name = f'Sequences, Max Length={max_len}'


    return 

def get_tokenization_dicts(cfg):
    tokenization_path='/'.join(cfg['inference_settings']['model_path'].split('/')[:-2])
    seq_token_dict = load(tokenization_path+'/seq_tokens_ids_dict')
    ss_token_dict = load(tokenization_path+'/second_input_tokens_ids_dict')
    return seq_token_dict,ss_token_dict

def get_hp_setting(cfg,setting):
    try:
        model_parent_path='/'.join(cfg['inference_settings']['model_path'].split('/')[:-2])
        return load(model_parent_path+'/meta/hp_settings')[setting]
    except: 
        root_path = str(Path(__file__).parents[3])
        cfg['inference_settings']['model_path'] = root_path+'/'+cfg['inference_settings']['model_path']
        model_parent_path='/'.join(cfg['inference_settings']['model_path'].split('/')[:-2])
        return load(model_parent_path+'/meta/hp_settings')[setting]

def add_ss_and_labels(infer_data):
    #check if infer_data has secondary structure
    if "Secondary" not in infer_data:
        infer_data["Secondary"] = fold_sequences(infer_data["Sequences"].tolist())['structure_37'].values
    if "Labels" not in infer_data:
        infer_data["Labels"] = [0]*len(infer_data["Sequences"].values)
    return infer_data

def prepare_model_inference(cfg,path):
    # instantiate skorch model
    net = instantiate_predictor(cfg["model"]["skorch_model"], cfg,path)
    net.initialize()
    net.load_params(f_params=f'{cfg["inference_settings"]["model_path"]}')
    net.labels_mapping_dict = dict(zip(cfg["model_config"].class_mappings,list(np.arange(cfg["model_config"].num_classes))))
    #save embeddings
    if cfg['log_embedds']:
        net.save_embedding=True
        net.gene_embedds = []
        net.second_input_embedds = []
    return net

def update_dataclass_inference(cfg,dataset_class):
    seq_token_dict,ss_token_dict = get_tokenization_dicts(cfg)
    dataset_class.seq_tokens_ids_dict = seq_token_dict
    dataset_class.second_input_tokens_ids_dict = ss_token_dict
    dataset_class.tokens_len =cfg["model_config"].tokens_len
    return dataset_class

def prepare_inference_data(cfg,ad,dataset_class):
    #tokenize sequences
    infere_data_df =   tokenize_set(dataset_class,ad,inference=True)
    infere_data,infere_rna_seq,_ = prepare_split(infere_data_df,cfg)

    all_data = {}
    all_data["infere_data"] = infere_data
    all_data["infere_rna_seq"] = infere_rna_seq
    return all_data

def get_model(cfg,path):

    cfg["model_config"] = get_hp_setting(cfg,'model_config')

    #set seed and update skorch config
    #set_seed_and_device(cfg["seed"],cfg["device_number"])
    sync_skorch_with_config(cfg["model"]["skorch_model"],cfg)
    cfg['model_config']['model_input'] = cfg['model_name']
    net = prepare_model_inference(cfg,path)
    return cfg,net

def chunkstring_overlap(string, window):
        return (
            string[0 + i : window + i] for i in range(0, len(string) - window + 1, 1)
        )

def create_short_seqs_from_long(df,max_len):
    long_seqs = df['Sequences'][df['Sequences'].str.len()>max_len].values
    short_seqs_pd = df[df['Sequences'].str.len()<=max_len]
    feature_tokens_gen = list(
            chunkstring_overlap(feature, max_len)
            for feature in long_seqs
        )
    seqs_shortend = [list(seq) for seq in feature_tokens_gen][0]
    shortened_df = pd.DataFrame(data=seqs_shortend,columns=['Sequences'])
    df = shortened_df.append(short_seqs_pd).reset_index(drop=True)
    return df

def infer_from_pd(cfg,net,infer_pd,DataClass,attention_flag:bool=False):
    try:
        max_len = net.module_.transformer_layers.pos_encoder.pe.shape[1]+1
    except:
        max_len = 30#for baseline models

    if cfg['model_name'] == 'seq-seq':
        max_len = max_len*2 - 1

    if len(infer_pd['Sequences'][infer_pd['Sequences'].str.len()>max_len].values)>0:
        infer_pd = create_short_seqs_from_long(infer_pd,max_len)
    infer_pd = add_ss_and_labels(infer_pd)
    infer_ad = convert_pd_to_ad(infer_pd,cfg)
    if cfg['model_name'] == 'seq-seq':
        cfg['model_config']['tokens_len'] *=2 
        cfg['model_config']['second_input_token_len'] *=2 
        
        
    #create dataclass to tokenize infer sequences
    dataset_class = DataClass(infer_ad.var,cfg)
    #update datasetclass with tokenization dicts and tokens_len
    dataset_class = update_dataclass_inference(cfg,dataset_class)
    #tokenize sequences
    all_data = prepare_inference_data(cfg,infer_ad,dataset_class)
    
    #inference on custom data
    predicted_labels,logits,attn_scores_first_list,attn_scores_second_list = infer_from_model(net,all_data["infere_data"])  
    if attention_flag:
        #in case of baseline or seq models
        if not attn_scores_second_list:
            attn_scores_second_list = attn_scores_first_list
            
        attn_scores_first = np.array(attn_scores_first_list)
        seq_lengths = all_data['infere_rna_seq']['Sequences'].str.len().values
        #get attention scores for each sequence
        attn_scores_list = [attn_scores_first[i,:seq_lengths[i],:seq_lengths[i]].flatten().tolist() for i in range(len(seq_lengths))]
        attn_scores_first_df = pd.DataFrame(data = {'attention_first':attn_scores_list})
        attn_scores_first_df.index = all_data['infere_rna_seq']['Sequences'].values

        attn_scores_second = np.array(attn_scores_second_list)
        attn_scores_list = [attn_scores_second[i,:seq_lengths[i],:seq_lengths[i]].flatten().tolist() for i in range(len(seq_lengths))]
        attn_scores_second_df = pd.DataFrame(data = {'attention_second':attn_scores_list})
        attn_scores_second_df.index = all_data['infere_rna_seq']['Sequences'].values

        attn_scores_df = attn_scores_first_df.join(attn_scores_second_df)
        attn_scores_df['Secondary'] = infer_ad.var["Secondary"].values
    else:
        attn_scores_df = None
    
    gene_embedds_df = None
    #net.gene_embedds is a list of tensors. convert them to a numpy array
    if cfg['log_embedds']:
        gene_embedds = np.vstack(net.gene_embedds)
        if cfg['model_name'] not in ['baseline']:
            second_input_embedds = np.vstack(net.second_input_embedds)
            gene_embedds = np.concatenate((gene_embedds,second_input_embedds),axis=1)
        gene_embedds_df = pd.DataFrame(data=gene_embedds)
        gene_embedds_df.index = all_data['infere_rna_seq']['Sequences'].values
        gene_embedds_df.columns = ['gene_embedds_'+str(i) for i in range(gene_embedds_df.shape[1])]

    return predicted_labels,logits,gene_embedds_df,attn_scores_df,all_data,max_len,net

def log_embedds(cfg,net,seqs_df):
    gene_embedds = np.vstack(net.gene_embedds)
    if not cfg['model_name'] in ['seq','baseline']:
        second_input_embedds = np.vstack(net.second_input_embedds)
        gene_embedds = np.concatenate((gene_embedds,second_input_embedds),axis=1)
    
    return seqs_df.join(pd.DataFrame(data=gene_embedds))
