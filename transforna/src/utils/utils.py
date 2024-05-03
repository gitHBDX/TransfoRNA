
import logging
import math
import os
import random
from pathlib import Path
from random import randint
from typing import List

import numpy as np
import pandas as pd
import torch
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from ..callbacks.metrics import get_callbacks
from ..score.score import infer_from_model
from .energy import *
from .file import load

logger = logging.getLogger(__name__)

def update_config_with_inference_params(config:DictConfig,mc_or_sc:str='sub_class',trained_on:str = 'id',path_to_id_models:str = 'models/tcga/') -> DictConfig:
    inference_config = config.copy()
    model = config['model_name']
    model = "-".join([word.capitalize() for word in model.split("-")])
    transforna_folder = "TransfoRNA_ID"
    if trained_on == "full":
        transforna_folder = "TransfoRNA_FULL"

    inference_config['inference_settings']["model_path"] = f'{path_to_id_models}{transforna_folder}/{mc_or_sc}/{model}/ckpt/model_params_tcga.pt'
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

def update_config_with_dataset_params_tcga(dataset_class,all_data_df,configs):
    configs["model_config"].ff_input_dim = all_data_df['second_input'].shape[1]
    configs["model_config"].vocab_size = len(dataset_class.seq_tokens_ids_dict.keys())
    configs["model_config"].second_input_vocab_size = len(dataset_class.second_input_tokens_ids_dict.keys())
    configs["model_config"].tokens_len = dataset_class.tokens_len
    configs["model_config"].second_input_token_len = dataset_class.tokens_len

    if configs["model_name"] == "seq-seq":
        configs["model_config"].tokens_len = math.ceil(dataset_class.tokens_len/2)
        configs["model_config"].second_input_token_len = math.ceil(dataset_class.tokens_len/2)
    

def update_dataclass_inference(cfg,dataset_class):
    seq_token_dict,ss_token_dict = get_tokenization_dicts(cfg)
    dataset_class.seq_tokens_ids_dict = seq_token_dict
    dataset_class.second_input_tokens_ids_dict = ss_token_dict
    dataset_class.tokens_len =cfg["model_config"].tokens_len
    dataset_class.max_length = get_hp_setting(cfg,'max_length')
    dataset_class.min_length = get_hp_setting(cfg,'min_length')
    return dataset_class

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

def revert_seq_tokenization(tokenized_seqs,configs):
        window = configs["model_config"].window
        if configs["model_config"].tokenizer != "overlap":
            logger.error("Sequences are not reverse tokenized")
            return tokenized_seqs
        
        #currently only overlap tokenizer can be reverted
        seqs_concat = []
        for seq in tokenized_seqs.values:
            seqs_concat.append(''.join(seq[seq!='pad'])[::window]+seq[seq!='pad'][-1][window-1])
        
        return pd.DataFrame(seqs_concat,columns=["Sequences"])

def introduce_mismatches(seq, n_mismatches):
    seq = list(seq)
    for i in range(n_mismatches):
        rand_nt = randint(0,len(seq)-1)
        seq[rand_nt] = ['A','G','C','T'][randint(0,3)]
    return ''.join(seq)

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

def prepare_data_benchmark(tokenizer,test_ad, configs):
    """
    This function recieves anddata and prepares the anndata in a format suitable for training
    It also set default parameters in the config that cannot be known until preprocessing step
    is done.
    all_data_df is heirarchical pandas dataframe, so can be accessed  [AA,AT,..,AC ]
    """
    ###get tokenized train set
    train_data_df = tokenizer.get_tokenized_data()
    
    ### update config with data specific params
    update_config_with_dataset_params_benchmark(train_data_df,configs)

    ###tokenize test set
    test_data_df = tokenize_set(tokenizer,test_ad.var)

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
        generalization_test_set = get_add_test_set(tokenizer,\
            dataset_path=configs["train_config"].datset_path_additional_testset)
    

    #get all vocab from both test and train set
    configs["model_config"].vocab_size = len(tokenizer.seq_tokens_ids_dict.keys())
    configs["model_config"].second_input_vocab_size = len(tokenizer.second_input_tokens_ids_dict.keys())
    configs["model_config"].tokens_mapping_dict = tokenizer.seq_tokens_ids_dict

    
    if configs["task"] == "premirna":
        generalization_test_data = []
        for test_df in generalization_test_set:
            #no need to read the labels as they are all one
            test_data_extra, _, _ =  prepare_split(test_df,configs)
            generalization_test_data.append(test_data_extra)
        all_data["additional_testset"] = generalization_test_data

    #get inference dataset
    # if do inference and inference datasert path exists
    get_inference_data(configs,tokenizer,all_data)

    return all_data

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

def prepare_inference_data(cfg,infer_pd,dataset_class):
    #tokenize sequences
    infere_data_df = tokenize_set(dataset_class,infer_pd,inference=True)
    infere_data,infere_rna_seq,_ = prepare_split(infere_data_df,cfg)

    all_data = {}
    all_data["infere_data"] = infere_data
    all_data["infere_rna_seq"] = infere_rna_seq
    return all_data

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
        dataset_class.limit_seqs_to_range(logger)
        infere_data_df = dataset_class.get_tokenized_data(inference=True)
        infere_data,infere_rna_seq,_ = prepare_split(infere_data_df,configs)

        all_data["infere_data"] = infere_data
        all_data["infere_rna_seq"] = infere_rna_seq

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
        all_added_test_set.append(dataset_class.get_tokenized_data())
    return all_added_test_set

def get_tokenization_dicts(cfg):
    tokenization_path='/'.join(cfg['inference_settings']['model_path'].split('/')[:-2])
    seq_token_dict = load(tokenization_path+'/seq_tokens_ids_dict')
    ss_token_dict = load(tokenization_path+'/second_input_tokens_ids_dict')
    return seq_token_dict,ss_token_dict

def get_hp_setting(cfg,hp_param):
    model_parent_path=Path('/'.join(cfg['inference_settings']['model_path'].split('/')[:-2]))
    hp_settings = load(model_parent_path/'meta/hp_settings.yaml')
    
    #hp_param could be in hp_settings .keyes or in a key of a key
    hp_val = hp_settings.get(hp_param)
    if not hp_val:
        for key in hp_settings.keys():
            try:
                hp_val = hp_settings[key].get(hp_param)
            except:
                pass
            if hp_val != None:
                break
    if hp_val == None:
        raise ValueError(f"hp_param {hp_param} not found in hp_settings")

    return hp_val

def get_model(cfg,path):

    cfg["model_config"] = get_hp_setting(cfg,'model_config')

    #set seed and update skorch config
    #set_seed_and_device(cfg["seed"],cfg["device_number"])
    sync_skorch_with_config(cfg["model"]["skorch_model"],cfg)
    cfg['model_config']['model_input'] = cfg['model_name']
    net = prepare_model_inference(cfg,path)
    return cfg,net

def stratify(train_data,train_labels,valid_size):
    return train_test_split(train_data, train_labels,
                                                    stratify=train_labels, 
                                                    test_size=valid_size)
 
def tokenize_set(dataset_class,test_pd,inference:bool=False):
    #reassign the sequences to test
    dataset_class.seqs_dot_bracket_labels = test_pd
    #prevent sequences with len < min lenght from being deleted
    dataset_class.limit_seqs_to_range()
    return  dataset_class.get_tokenized_data(inference)

def add_ss_and_labels(infer_data):
    #check if infer_data has secondary structure
    if "Secondary" not in infer_data:
        infer_data["Secondary"] = fold_sequences(infer_data["Sequences"].tolist())['structure_37'].values
    if "Labels" not in infer_data:
        infer_data["Labels"] = [0]*len(infer_data["Sequences"].values)
    return infer_data

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
    if cfg['model_name'] == 'seq-seq':
        cfg['model_config']['tokens_len'] *=2 
        cfg['model_config']['second_input_token_len'] *=2 
        
        
    #create dataclass to tokenize infer sequences
    dataset_class = DataClass(infer_pd,cfg)
    #update datasetclass with tokenization dicts and tokens_len
    dataset_class = update_dataclass_inference(cfg,dataset_class)
    #tokenize sequences
    all_data = prepare_inference_data(cfg,infer_pd,dataset_class)
    
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
        attn_scores_df['Secondary'] = infer_pd["Secondary"].values
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
